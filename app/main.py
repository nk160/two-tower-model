from fastapi import FastAPI
from search_engine import SearchEngine
import time
import os

app = FastAPI()
model_version = "1.0.0"
log_dir_path = "/var/log/app"
log_path = f"{log_dir_path}/V-{model_version}.log"

# Initialize search engine
search_engine = SearchEngine('checkpoints/checkpoint_epoch_1.pt')

@app.get("/ping")
def ping():
    return "ok"

@app.get("/version")
def version():
    return {"version": model_version}

@app.post("/search")
async def search(query: str, k: int = 5):
    start_time = time.time()
    results = search_engine.search(query, k)
    latency = (time.time() - start_time) * 1000
    
    return {
        "results": results,
        "latency_ms": latency
    }
