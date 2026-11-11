# api/schemas.py

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Provider = Literal["huggingface", "anthropic", "ollama", "none"]


class RepoIn(BaseModel):
    path: str | None = Field(None, description="Folder on the API's machine")
    github_url: str | None = Field(None, description="Public repository, e.g. https://github.com/owner/repo")

    @model_validator(mode="after")
    def one_source(self):
        if (self.path is None) == (self.github_url is None):
            raise ValueError("Send either path or github_url")
        return self


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatIn(BaseModel):
    question: str = Field(min_length=1)
    history: list[Message] = []
    provider: Provider | None = Field(None, description="Omit to use the first provider configured on the server")
    model: str | None = None
    k: int = Field(6, ge=1, le=20, description="Passages to retrieve")
    rerank: bool = False
    rewrite_followups: bool = True
    mode: Literal["agent", "oneshot"] = Field(
        "agent", description="agent lets the model search and read files with tools; oneshot retrieves once"
    )


class SearchIn(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(6, ge=1, le=20)
