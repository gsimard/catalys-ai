import torch
from transformers import AutoModel, AutoTokenizer
from typing import Union, List
import time

class BGEM3Encoder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto"):
        """Initialise l'encodeur avec gestion automatique du GPU"""
        self.device = self._get_device(device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()  # Mode inference
    
    def _get_device(self, device: str) -> torch.device:
        """Détermine le meilleur device disponible"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def encode(self, 
              texts: Union[str, List[str]], 
              batch_size: int = 32,
              max_length: int = 8192,
              pooling_method: str = "cls") -> torch.Tensor:
        """Encode les textes en embeddings avec différentes options"""
        
        # Gestion des entrées simples
        if isinstance(texts, str):
            texts = [texts]
            
        # Découpage en batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenization
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Inférence
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Pooling
            if pooling_method == "cls":
                batch_embeddings = outputs.last_hidden_state[:, 0]
            elif pooling_method == "mean":
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                raise ValueError(f"Méthode de pooling inconnue: {pooling_method}")
                
            embeddings.append(batch_embeddings.cpu())
            
        return torch.cat(embeddings, dim=0)

# Usage avec optimisation GPU
if __name__ == "__main__":
    encoder = BGEM3Encoder()
    
    texts = [
        "La révolution française de 1789",
        "量子コンピューティングの基本原理",
        "Theoretical foundations of deep learning",
        "¿Cómo funciona el mecanismo de atención en los transformers?"
    ]
    
    start_time = time.time()
    embeddings = encoder.encode(texts, pooling_method="cls")
    inference_time = time.time() - start_time
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Temps d'inférence: {inference_time:.2f}s")
    print(f"Memoire GPU utilisée: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
