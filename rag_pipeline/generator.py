from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class FlanT5Generator:
    def __init__(self, model_name='google/flan-t5-base'):
        print(f"Loading model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def generate(self, question, context, max_new_tokens=256):
        prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {question}"
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer
