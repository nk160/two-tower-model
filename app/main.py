from fastapi import FastAPI, HTTPException
from search_engine import SearchEngine
import time
import os
from pydantic import BaseModel

app = FastAPI()
model_version = "1.0.0"
log_dir_path = "/var/log/app"
log_path = f"{log_dir_path}/V-{model_version}.log"

# Initialize search engine
search_engine = SearchEngine('checkpoints/checkpoint_epoch_1.pt')

# Add after initializing search engine
test_documents = [
    "How to implement machine learning models in Python",
    "Best practices for deep learning in PyTorch",
    "Introduction to natural language processing",
    "Understanding neural networks and backpropagation",
    "Python programming basics for beginners"
]
search_engine.cache_documents(test_documents)

class SearchQuery(BaseModel):
    query: str
    k: int = 5  # default value

@app.get("/ping")
def ping():
    return "ok"

@app.get("/version")
def version():
    return {"version": model_version}

@app.post("/search")
async def search(query: SearchQuery):
    try:
        start_time = time.time()
        results = search_engine.search(query.query, query.k)
        latency = (time.time() - start_time) * 1000
        
        # Convert results to a simpler format
        formatted_results = [
            {
                "index": int(idx),
                "distance": float(dist),
                "document": test_documents[int(idx)]  # Add the actual document text
            }
            for idx, dist in results
        ]
        
        return {
            "results": formatted_results,
            "query": query.query,
            "k": query.k,
            "latency_ms": float(latency)  # Ensure latency is float
        }
    except Exception as e:
        print(f"Error during search: {str(e)}")  # For debugging
        raise HTTPException(status_code=500, detail=str(e))
