
import torch
from transformers import AutoModel, AutoTokenizer

from .config import TEXT_ENCODER_CONFIG

class TextEncoder:
    def __init__(self):
        self.model_name = TEXT_ENCODER_CONFIG['model_name']
        self.max_length = TEXT_ENCODER_CONFIG['max_length']
        self.batch_size = TEXT_ENCODER_CONFIG['batch_size']
        self.device = TEXT_ENCODER_CONFIG['device']
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    def encode_batch(self, texts):
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                batch_embeddings = torch.mean(last_hidden_state, dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def encode_single(self, text):
        return self.encode_batch([text])[0]

if __name__ == "__main__":
    encoder = TextEncoder()
    test_texts = ["This is a test sentence.", "这是一个中文测试句子。"]
    embeddings = encoder.encode_batch(test_texts)
    print(f"Encoded {len(embeddings)} texts")
    print(f"Embedding shape: {embeddings[0].shape}")
