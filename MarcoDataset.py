from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import random
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import logging
from tqdm import tqdm
import sys
import json
import os
from typing import List, Dict, Iterator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class MarcoDataset(Dataset):
    def __init__(self, split="train", max_length=64, data_dir="marco_data"):
        self.max_length = max_length
        self.data_dir = data_dir
        self.split = split
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load or download data
        self.data = self._load_data()
        
        # Build vocabulary
        self.vocab = self._build_vocab()

    def _load_data(self) -> List[Dict]:
        """Load or download the MS MARCO dataset"""
        filename = f"{self.split}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Data file not found at {filepath}. Please download the MS MARCO dataset "
                "and place it in the marco_data directory."
            )
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization"""
        return text.lower().split()

    def _build_vocab(self):
        """Build vocabulary with progress bar"""
        def yield_tokens():
            for item in tqdm(self.data, desc="Building vocab"):
                yield self._tokenize(item['query'])
                yield self._tokenize(item['positive_doc'])
                yield self._tokenize(item['negative_doc'])
        
        vocab = build_vocab_from_iterator(yield_tokens(), specials=['<pad>', '<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def _convert_to_ids(self, text: str) -> torch.Tensor:
        """Convert text to token IDs and pad/truncate to max_length"""
        tokens = self._tokenize(text)
        ids = [self.vocab[token] for token in tokens]
        
        # Pad or truncate
        if len(ids) < self.max_length:
            ids = ids + [self.vocab['<pad>']] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
            
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query_input_ids': self._convert_to_ids(item['query']),
            'positive_doc_input_ids': self._convert_to_ids(item['positive_doc']),
            'negative_doc_input_ids': self._convert_to_ids(item['negative_doc'])
        }

    def analyze_data(self):
        """Print statistics about the dataset"""
        queries = set()
        pos_docs = set()
        neg_docs = set()
        
        print(f"\nAnalyzing {self.split} dataset:")
        
        for item in self.data:
            queries.add(item['query'])
            pos_docs.add(item['positive_doc'])
            neg_docs.add(item['negative_doc'])
        
        print(f"Total samples: {len(self.data)}")
        print(f"Unique queries: {len(queries)}")
        print(f"Unique positive docs: {len(pos_docs)}")
        print(f"Unique negative docs: {len(neg_docs)}")
        
        # Sample a few examples
        print("\nSample pairs:")
        for i in range(min(3, len(self.data))):
            print(f"\nPair {i+1}:")
            print(f"Query: {self.data[i]['query']}")
            print(f"Positive: {self.data[i]['positive_doc'][:100]}...")
            print(f"Negative: {self.data[i]['negative_doc'][:100]}...")

def generate_sample_data(num_samples=5000):
    # Ensure correct positive pairs
    data = []
    for _ in range(num_samples):
        topic = random.choice(list(topics.keys()))
        query_idx = random.randrange(len(topics[topic]["queries"]))
        
        # Match query with its correct positive document
        query = topics[topic]["queries"][query_idx]
        positive_doc = topics[topic]["documents"][query_idx]
        
        # Select negative from different topic
        neg_topic = random.choice([t for t in topics.keys() if t != topic])
        negative_doc = random.choice(topics[neg_topic]["documents"])
        
        data.append({
            "query": query,
            "positive_doc": positive_doc,
            "negative_doc": negative_doc
        })

# Add this at the bottom of MarcoDataset.py
if __name__ == "__main__":
    # First verify data files exist
    print("Checking data files...")
    data_dir = "marco_data"
    required_files = ["train.json", "validation.json"]
    for file in required_files:
        path = os.path.join(data_dir, file)
        exists = os.path.exists(path)
        print(f"{path}: {'✓' if exists else '✗'}")

    # Try initializing dataset
    try:
        print("\nTrying to initialize training dataset...")
        train_dataset = MarcoDataset(split="train")
        print(f"Training samples: {len(train_dataset)}")

        # Check a sample
        sample = train_dataset[0]
        print("\nSample data structure:")
        for key, value in sample.items():
            print(f"{key}: shape {value.shape}")
            
        # Add data analysis
        print("\nAnalyzing datasets:")
        train_dataset.analyze_data()
        
    except Exception as e:
        print(f"Error initializing dataset: {str(e)}")
