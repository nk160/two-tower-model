import json
import os
from datasets import load_dataset
from tqdm import tqdm
import random

def download_marco_data():
    """Download and prepare full MS MARCO dataset with 10 negatives per positive"""
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v2.1")
    
    NUM_NEGATIVES = 10  # Use 10 negative examples per positive
    
    # Print dataset structure
    print("\nDataset structure:")
    print(f"Available splits: {dataset.keys()}")
    
    # Print example item structure
    print("\nExample item structure:")
    example = dataset['train'][0]
    print("Keys in training example:", example.keys())
    print("\nDetailed example:")
    for key, value in example.items():
        print(f"\n{key}: {type(value)}")
        if isinstance(value, dict):
            print("Dictionary keys:", value.keys())
            for k, v in value.items():
                print(f"  {k}: {type(v)}")
                if hasattr(v, '__len__'):
                    print(f"    Length: {len(v)}")
        elif hasattr(value, '__len__'):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element: {value[0]}")
    
    # Continue with data processing after we see the structure
    print("\nProcessing training data...")
    train_data = []
    for item in tqdm(dataset['train']):
        query = item['query']
        is_selected = item['passages']['is_selected']
        passage_texts = item['passages']['passage_text']
        
        # Get positive and negative passages
        positive_passages = [text for text, selected in zip(passage_texts, is_selected) if selected]
        negative_passages = [text for text, selected in zip(passage_texts, is_selected) if not selected]
        
        if positive_passages and negative_passages:
            for pos_doc in positive_passages:
                # Sample 10 negative passages (with replacement if needed)
                if len(negative_passages) >= NUM_NEGATIVES:
                    neg_docs = random.sample(negative_passages, NUM_NEGATIVES)
                else:
                    neg_docs = negative_passages + random.choices(negative_passages, 
                        k=NUM_NEGATIVES-len(negative_passages))
                
                # Add all negative examples
                for neg_doc in neg_docs:
                    train_data.append({
                        "query": query,
                        "positive_doc": pos_doc,
                        "negative_doc": neg_doc
                    })
    
    print("\nProcessing validation data...")
    val_data = []
    for item in tqdm(dataset['validation']):
        query = item['query']
        is_selected = item['passages']['is_selected']
        passage_texts = item['passages']['passage_text']
        
        positive_passages = [text for text, selected in zip(passage_texts, is_selected) if selected]
        negative_passages = [text for text, selected in zip(passage_texts, is_selected) if not selected]
        
        if positive_passages and negative_passages:
            for pos_doc in positive_passages:
                if len(negative_passages) >= NUM_NEGATIVES:
                    neg_docs = random.sample(negative_passages, NUM_NEGATIVES)
                else:
                    neg_docs = negative_passages + random.choices(negative_passages, 
                        k=NUM_NEGATIVES-len(negative_passages))
                
                for neg_doc in neg_docs:
                    val_data.append({
                        "query": query,
                        "positive_doc": pos_doc,
                        "negative_doc": neg_doc
                    })
    
    print("Saving data files...")
    os.makedirs("marco_data", exist_ok=True)
    
    with open("marco_data/train.json", "w") as f:
        json.dump(train_data, f)
    
    with open("marco_data/validation.json", "w") as f:
        json.dump(val_data, f)
    
    print(f"Saved {len(train_data)} training examples")
    print(f"Saved {len(val_data)} validation examples")

if __name__ == "__main__":
    download_marco_data() 