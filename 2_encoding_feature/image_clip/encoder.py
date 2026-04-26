from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from .config import IMAGE_ENCODER_CONFIG

class ImageEncoder:
    def __init__(self):
        self.model_name = IMAGE_ENCODER_CONFIG['model_name']
        self.image_size = IMAGE_ENCODER_CONFIG['image_size']
        self.batch_size = IMAGE_ENCODER_CONFIG['batch_size']
        self.device = IMAGE_ENCODER_CONFIG['device']
        
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    def encode_batch(self, image_paths):
        embeddings = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            
            images = []
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    continue
            
            if not images:
                continue
            
            inputs = self.processor(images=images, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    batch_embeddings = outputs.last_hidden_state.cpu().numpy()
                else:
                    batch_embeddings = outputs.cpu().numpy()
                if batch_embeddings.ndim == 3:
                    batch_embeddings = np.mean(batch_embeddings, axis=1)
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def encode_single(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    embedding = outputs.last_hidden_state.cpu().numpy()[0]
                else:
                    embedding = outputs.cpu().numpy()[0]
                if embedding.ndim == 2:
                    embedding = np.mean(embedding, axis=0)
                return embedding
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
