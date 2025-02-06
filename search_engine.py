import faiss
import torch
import numpy as np
from Version3 import TwoTowerModel
from typing import List, Tuple

class SearchEngine:
    def __init__(self, model_path: str = 'checkpoints/checkpoint_epoch_1.pt'):
        # Load the model and move to eval mode
        print("Loading checkpoint...")
        checkpoint = torch.load(model_path)
        self.model = TwoTowerModel(
            vocab_size=len(checkpoint['vocab']),
            vocab=checkpoint['vocab']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize FAISS index
        self.dimension = 384  # OUTPUT_DIM from model
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store vocabulary for tokenization
        self.vocab = checkpoint['vocab']
        
        # Add max_length parameter
        self.max_length = 64  # Same as MAX_LENGTH in Version3.py
        
    def cache_documents(self, documents: List[str]):
        """Pre-compute and store document embeddings"""
        print("Caching document embeddings...")
        document_embeddings = []
        
        # Process documents in batches
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            # Add tokenization here
            batch_tokens = [self.tokenize(doc) for doc in batch]
            # Stack tokens into a single tensor
            batch_tokens = torch.stack(batch_tokens)
            
            # Encode documents using document tower
            with torch.no_grad():
                embeddings = self.model.document_tower(batch_tokens)
                embeddings = embeddings.detach().cpu().numpy()
                document_embeddings.extend(embeddings)
        
        # Convert to numpy array and add to FAISS index
        document_embeddings = np.array(document_embeddings)
        self.index.add(document_embeddings)
        print(f"Cached {len(documents)} documents in FAISS index")

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Search for k most similar documents to query"""
        # Tokenize and encode query
        query_tokens = self.tokenize(query)
        query_embedding = self.model.query_tower(query_tokens.unsqueeze(0))
        query_embedding = query_embedding.detach().cpu().numpy()
        
        # Reshape query embedding to 2D array (1, dimension)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        return list(zip(indices[0], distances[0]))

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text using the model's vocabulary"""
        # Split text into tokens
        tokens = text.lower().split()
        
        # Convert to indices using vocabulary
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<unk>'])
            
        # Convert to tensor and pad/truncate to MAX_LENGTH
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.vocab['<pad>']] * (self.max_length - len(token_ids)))
        
        return torch.tensor(token_ids)
