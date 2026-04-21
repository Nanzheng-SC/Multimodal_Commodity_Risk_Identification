#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModel, AutoTokenizer

from .config import TEXT_ENCODER_CONFIG

class TextEncoder:
    def __init__(self):
        self.model_name = TEXT_ENCODER_CONFIG['model_name']
        self.max_length = TEXT_ENCODER_CONFIG['max_length']
        self.batch_size = TEXT_ENCODER_CONFIG['batch_size']
        self.device = TEXT_ENCODER_CONFIG['device']
        
        # 统一使用 AutoTokenizer / AutoModel，便于多语言模型切换。
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    def encode_batch(self, texts):
        """
        批量编码文本
        """
        embeddings = []
        
        # 分批次处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # 编码
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用最后一层 token 平均池化
                last_hidden_state = outputs.last_hidden_state
                # 计算平均池化
                batch_embeddings = torch.mean(last_hidden_state, dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def encode_single(self, text):
        """
        编码单个文本
        """
        return self.encode_batch([text])[0]

if __name__ == "__main__":
    encoder = TextEncoder()
    test_texts = ["This is a test sentence.", "这是一个中文测试句子。"]
    embeddings = encoder.encode_batch(test_texts)
    print(f"Encoded {len(embeddings)} texts")
    print(f"Embedding shape: {embeddings[0].shape}")
