import torch
from transformers import AutoModel, AutoTokenizer
from typing import Union, List, Dict

class BGEM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto"):
        self.device = self._determine_device(device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def _determine_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @torch.inference_mode()
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        max_length: int = 8192,
        pooling: str = "cls"
    ) -> torch.Tensor:
        """
        Génère des embeddings pour une liste de textes.
        
        Args:
            texts: Un texte ou une liste de textes à encoder
            batch_size: Taille des lots pour le traitement
            max_length: Longueur maximale des séquences
            pooling: Méthode de pooling ('cls', 'mean', 'max')
            
        Returns:
            Tensor contenant les embeddings normalisés
        """
        # Conversion en liste si un seul texte est fourni
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        # Traitement par lots
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenisation avec padding et truncation
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Génération des embeddings
            outputs = self.model(**inputs)
            
            # Différentes stratégies de pooling
            if pooling == "cls":
                embeddings = outputs.last_hidden_state[:, 0]  # Token [CLS]
            elif pooling == "mean":
                # Moyenne sur les tokens non-padding
                attention_mask = inputs["attention_mask"]
                embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            elif pooling == "max":
                # Max pooling sur les tokens non-padding
                attention_mask = inputs["attention_mask"]
                embeddings = self._max_pooling(outputs.last_hidden_state, attention_mask)
            else:
                raise ValueError(f"Méthode de pooling '{pooling}' non supportée")
            
            # Normalisation L2
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
            
        # Concaténation des embeddings de tous les lots
        return torch.cat(all_embeddings, dim=0)
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Calcule la moyenne des embeddings en ignorant les tokens de padding."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _max_pooling(self, token_embeddings, attention_mask):
        """Applique max pooling sur les embeddings en ignorant les tokens de padding."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
