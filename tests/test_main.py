import pytest

from rag_pipeline import main as cli
from rag_pipeline import pipeline as pipeline_module
from tests.conftest import FakeEmbeddings, FakeLLM


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Runs the CLI against a temporary index with fake embeddings, so no model is downloaded."""
    monkeypatch.setenv("RAG_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(pipeline_module, "get_embedding_model", lambda name: FakeEmbeddings())


def use_llm(monkeypatch, llm):
    monkeypatch.setattr(cli, "get_llm", lambda provider, model=None: llm)


def test_retrieval_only_mode_prints_the_matching_code(codebase, offline, capsys):
    assert cli.main(["--path", str(codebase), "--provider", "none", "-q", "How are passwords hashed?"]) == 0

    out = capsys.readouterr().out
    assert "Index ready: 5 files" in out
    assert "LLM: none (retrieval-only mode" in out
    assert "[1] app/auth.py" in out
    assert "def hash_password" in out


def test_answers_a_question_with_sources(codebase, offline, monkeypatch, capsys):
    use_llm(monkeypatch, FakeLLM("It uses PBKDF2 [1]."))

    assert cli.main(["--path", str(codebase), "--provider", "ollama", "-q", "How are passwords hashed?"]) == 0

    out = capsys.readouterr().out
    assert "LLM: ollama / fake-model" in out
    assert "It uses PBKDF2 [1]." in out
    assert "Sources:\n  [1] app/auth.py" in out


def test_rebuild_reindexes_everything(codebase, offline, capsys):
    cli.main(["--path", str(codebase), "--provider", "none", "-q", "cart"])
    cli.main(["--path", str(codebase), "--provider", "none", "--rebuild", "-q", "cart"])

    second_run = capsys.readouterr().out.split("Index ready")[-1]
    assert "new 5, changed 0, removed 0, unchanged 0" in second_run


def test_provider_errors_are_reported_without_a_traceback(codebase, offline, monkeypatch, capsys):
    class CrashingLLM(FakeLLM):
        def stream(self, *args, **kwargs):
            raise RuntimeError("Ollama could not load gpt-oss:20b")
            yield

    use_llm(monkeypatch, CrashingLLM())

    assert cli.main(["--path", str(codebase), "--provider", "ollama", "-q", "anything"]) == 1
    assert "The ollama request failed: Ollama could not load gpt-oss:20b" in capsys.readouterr().err


def test_interactive_mode_keeps_history_until_exit(codebase, offline, monkeypatch, capsys):
    llm = FakeLLM("Answer [1].")
    use_llm(monkeypatch, llm)
    questions = iter(["How are passwords hashed?", "", "and the cart?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(questions))

    assert cli.main(["--path", str(codebase), "--provider", "ollama"]) == 0

    out = capsys.readouterr().out
    assert out.count("Sources:") == 2
    # answer 1, follow-up rewrite, answer 2 (which receives the first exchange as history)
    assert len(llm.calls) == 3
    assert llm.calls[2]["messages"][0] == {"role": "user", "content": "How are passwords hashed?"}


def test_api_mode_indexes_and_answers_through_the_api(codebase, make_api_client, monkeypatch, capsys):
    make_api_client(llm=FakeLLM("It uses PBKDF2 [1]."))
    monkeypatch.setattr(pipeline_module, "get_embedding_model", lambda name: pytest.fail("loaded a model locally"))

    code = cli.main(
        [
            "--api",
            "http://api.test",
            "--path",
            str(codebase),
            "--provider",
            "ollama",
            "--oneshot",
            "-q",
            "How are passwords hashed?",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "API: http://api.test" in out
    assert "Index ready: 5 files" in out
    assert "It uses PBKDF2 [1]." in out
    assert "Sources:\n  [1] app/auth.py" in out


def test_api_mode_without_an_llm_prints_the_matching_code(codebase, make_api_client, capsys):
    make_api_client()

    assert cli.main(["--api", "http://api.test", "--path", str(codebase), "--provider", "none", "-q", "passwords"]) == 0

    out = capsys.readouterr().out
    assert "[1] app/auth.py" in out
    assert "def hash_password" in out


def test_api_mode_sends_the_history_of_an_interactive_session(codebase, make_api_client, monkeypatch, capsys):
    llm = FakeLLM("Answer [1].")
    make_api_client(llm=llm)
    questions = iter(["How are passwords hashed?", "and the cart?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(questions))

    assert cli.main(["--api", "http://api.test", "--path", str(codebase), "--provider", "ollama", "--oneshot"]) == 0

    assert capsys.readouterr().out.count("Sources:") == 2
    assert llm.calls[-1]["messages"][0] == {"role": "user", "content": "How are passwords hashed?"}


def test_api_mode_reports_a_failed_index(tmp_path, make_api_client, capsys):
    make_api_client()

    assert cli.main(["--api", "http://api.test", "--path", str(tmp_path / "missing"), "-q", "anything"]) == 1
    assert "Folder not found" in capsys.readouterr().err


@pytest.mark.parametrize("flag", [["--rebuild"], ["--mode", "dense"]])
def test_api_mode_rejects_options_for_the_local_index(flag, capsys):
    with pytest.raises(SystemExit):
        cli.main(["--api", "http://api.test", *flag])
    assert "--api" in capsys.readouterr().err
