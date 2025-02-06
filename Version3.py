import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from MarcoDataset import MarcoDataset
import logging
from tqdm import tqdm
import gensim.downloader as gensim_downloader
import numpy as np
import os

# Configuration Parameters
EMBEDDING_DIM = 300    # Match word2vec dimensions
HIDDEN_DIM = 768      # BiGRU hidden dimension (384 each direction)
OUTPUT_DIM = 384      # Final embedding dimension
MAX_LENGTH = 64       # Maximum sequence length
BATCH_SIZE = 256      # Batch size
LEARNING_RATE = 0.00005
NUM_EPOCHS = 3        # Running 3 epochs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add a subset size limit for initial testing
SUBSET_SIZE = 50000    # Limit training data to 50k samples

# Optional: Add gradient clipping
CLIP_GRAD_NORM = 1.0    # Add gradient clipping

class TowerModel(nn.Module):
    """
    Tower with BiGRU and attention for encoding queries/documents
    """
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=768, output_dim=384):
        super().__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # Split for bidirectional
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, input_ids):
        # Embed input tokens
        x = self.embedding(input_ids)                  # [batch, seq_len, emb_dim]
        
        # Pass through BiGRU
        outputs, _ = self.gru(x)                       # [batch, seq_len, hidden_dim]
        
        # Compute attention weights
        weights = self.attention(outputs)              # [batch, seq_len, 1]
        weights = F.softmax(weights, dim=1)           # [batch, seq_len, 1]
        
        # Apply attention
        x = torch.bmm(weights.transpose(1, 2), outputs) # [batch, 1, hidden_dim]
        x = x.squeeze(1)                               # [batch, hidden_dim]
        
        # Final projection
        x = self.fc(x)                                # [batch, output_dim]
        x = self.norm(x)                              # [batch, output_dim]
        
        return x

class TwoTowerModel(nn.Module):
    """
    Dual encoder model with shared word embeddings
    """
    def __init__(self, vocab_size, vocab):
        super().__init__()
        
        # Store vocabulary for embedding initialization
        self.vocab = vocab
        
        # Initialize the two towers (sharing embedding weights)
        self.query_tower = TowerModel(vocab_size)
        self.document_tower = TowerModel(vocab_size)
        
        # Ensure embedding layers share weights
        self.document_tower.embedding = self.query_tower.embedding
        
        # Load pre-trained embeddings
        self.load_pretrained_embeddings()
    
    def load_pretrained_embeddings(self):
        """Load and freeze pre-trained word2vec embeddings"""
        logging.info("Loading pre-trained word2vec embeddings...")
        
        # Load word vectors
        word2vec = gensim_downloader.load('word2vec-google-news-300')
        
        # Initialize embedding matrix
        weights = torch.randn(self.query_tower.embedding.weight.shape)
        
        # Copy pre-trained embeddings
        for i, word in enumerate(self.vocab.get_itos()):
            if word in word2vec:
                vector = torch.FloatTensor(word2vec[word])
                weights[i] = vector
        
        # Load embeddings into both towers (they share the embedding layer)
        self.query_tower.embedding.weight.data.copy_(weights)
        
        # Freeze embeddings
        self.query_tower.embedding.weight.requires_grad = False
        logging.info("Embeddings loaded and frozen")
    
    def forward(self, query, pos_doc, neg_doc):
        # Encode query
        query_encoding = self.query_tower(query)
        
        # Encode positive and negative documents
        pos_doc_encoding = self.document_tower(pos_doc)
        neg_doc_encoding = self.document_tower(neg_doc)
        
        return query_encoding, pos_doc_encoding, neg_doc_encoding

def compute_similarity(query_emb, doc_emb):
    """Compute cosine similarity between embeddings"""
    return F.cosine_similarity(query_emb, doc_emb)

def compute_loss(query_emb, pos_doc_emb, neg_doc_emb, temperature=1.0):
    # Cosine similarity with temperature scaling
    pos_sim = F.cosine_similarity(query_emb, pos_doc_emb) / temperature
    neg_sim = F.cosine_similarity(query_emb, neg_doc_emb) / temperature
    
    # InfoNCE loss
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(len(query_emb), dtype=torch.long, device=query_emb.device)
    return F.cross_entropy(logits, labels)

class TripletLossWithMetrics:
    def __init__(self, margin=0.3):
        self.margin = margin
    
    def __call__(self, query_enc, pos_doc_enc, neg_doc_enc):
        # Normalize embeddings
        query_enc = F.normalize(query_enc, p=2, dim=1)
        pos_doc_enc = F.normalize(pos_doc_enc, p=2, dim=1)
        neg_doc_enc = F.normalize(neg_doc_enc, p=2, dim=1)
        
        # Compute similarities
        pos_sim = compute_similarity(query_enc, pos_doc_enc)
        neg_sim = compute_similarity(query_enc, neg_doc_enc)
        
        # Compute loss with margin
        loss = F.relu(self.margin - pos_sim + neg_sim).mean()
        
        # Compute metrics
        avg_pos_sim = pos_sim.mean().item()
        avg_neg_sim = neg_sim.mean().item()
        triplet_accuracy = (pos_sim > neg_sim).float().mean().item()
        
        return loss, {
            'positive_similarity': avg_pos_sim,
            'negative_similarity': avg_neg_sim,
            'triplet_accuracy': triplet_accuracy
        }

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint(model, optimizer, epoch, metrics, filename='checkpoint.pt'):
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'vocab': model.vocab,  # Save vocabulary
    }, path)
    logging.info(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, filename='checkpoint.pt'):
    path = os.path.join('checkpoints', filename)
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Checkpoint loaded: {path}")
        return checkpoint['epoch'], checkpoint['metrics']
    return 0, None

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, early_stopping):
    """Train the two tower model with detailed metrics and early stopping"""
    model = model.to(DEVICE)
    best_val_metrics = {'loss': float('inf'), 'accuracy': 0}
    history = {'train': [], 'val': []}
    
    for epoch in range(num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_metrics = {
            'loss': 0,
            'positive_similarity': 0,
            'negative_similarity': 0,
            'triplet_accuracy': 0
        }
        
        train_batches = tqdm(train_loader, desc="Training")
        for batch in train_batches:
            # Move batch to device
            query = batch['query_input_ids'].to(DEVICE)
            pos_doc = batch['positive_doc_input_ids'].to(DEVICE)
            neg_doc = batch['negative_doc_input_ids'].squeeze(1).to(DEVICE)
            
            # Forward pass
            query_enc, pos_doc_enc, neg_doc_enc = model(query, pos_doc, neg_doc)
            
            # Compute loss and metrics
            loss, batch_metrics = criterion(query_enc, pos_doc_enc, neg_doc_enc)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping before optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
            
            # Update metrics
            train_metrics['loss'] += loss.item()
            for k, v in batch_metrics.items():
                train_metrics[k] += v
            
            # Update progress bar
            train_batches.set_postfix({
                'loss': loss.item(),
                'acc': batch_metrics['triplet_accuracy']
            })
        
        # Average metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_metrics = {
            'loss': 0,
            'positive_similarity': 0,
            'negative_similarity': 0,
            'triplet_accuracy': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                query = batch['query_input_ids'].to(DEVICE)
                pos_doc = batch['positive_doc_input_ids'].to(DEVICE)
                neg_doc = batch['negative_doc_input_ids'].squeeze(1).to(DEVICE)
                
                query_enc, pos_doc_enc, neg_doc_enc = model(query, pos_doc, neg_doc)
                loss, batch_metrics = criterion(query_enc, pos_doc_enc, neg_doc_enc)
                
                val_metrics['loss'] += loss.item()
                for k, v in batch_metrics.items():
                    val_metrics[k] += v
        
        # Average metrics
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        # Log metrics
        logging.info("\nTraining Metrics:")
        for k, v in train_metrics.items():
            logging.info(f"{k}: {v:.4f}")
        
        logging.info("\nValidation Metrics:")
        for k, v in val_metrics.items():
            logging.info(f"{k}: {v:.4f}")
        
        # Early stopping check
        early_stopping(val_metrics['loss'])
        if early_stopping.should_stop:
            logging.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best model
        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'early_stopped': early_stopping.should_stop
            }, 'best_model.pt')
        
        # Add at the end of each epoch:
        save_checkpoint(
            model, 
            optimizer, 
            epoch,
            {'train': train_metrics, 'val': val_metrics},
            f'checkpoint_epoch_{epoch+1}.pt'
        )
    
    return history

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load datasets
    logging.info("Loading datasets...")
    train_dataset = MarcoDataset(split="train", max_length=MAX_LENGTH)
    val_dataset = MarcoDataset(split="validation", max_length=MAX_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Initialize model with vocabulary
    model = TwoTowerModel(
        vocab_size=len(train_dataset.vocab),
        vocab=train_dataset.vocab  # Pass vocab to model
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=TripletLossWithMetrics(margin=0.3),
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        early_stopping=EarlyStopping(patience=2, min_delta=0.001)
    )
