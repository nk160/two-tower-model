from search_engine import SearchEngine
from typing import List

def load_test_documents() -> List[str]:
    """Load some test documents"""
    return [
        "How to implement machine learning models in Python",
        "Best practices for deep learning in PyTorch",
        "Introduction to natural language processing",
        "Understanding neural networks and backpropagation",
        "Python programming basics for beginners"
    ]

def main():
    # Initialize search engine
    engine = SearchEngine()
    
    # Load and cache documents
    documents = load_test_documents()
    engine.cache_documents(documents)
    
    # Try a search
    query = "How do I learn PyTorch?"
    results = engine.search(query, k=3)
    
    # Print results
    print(f"\nQuery: {query}")
    print("\nTop 3 matches:")
    for idx, distance in results:
        print(f"Document: {documents[idx]}")
        print(f"Distance: {distance:.4f}\n")

if __name__ == "__main__":
    main() 