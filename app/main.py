import os
import wandb
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import json
from typing import List, Dict
import torch
from Version4 import TwoTowerModel  # Import your model class

# Initialize FastAPI app
app = FastAPI()

# Model version
MODEL_VERSION = "1.0.0"

# Log directory setup
LOG_DIR_PATH = "/var/log/app"
LOG_PATH = f"{LOG_DIR_PATH}/V-{MODEL_VERSION}.log"

# Initialize model (we'll load it once when starting the server)
def load_model_from_wandb():
    wandb.login()
    
    run = wandb.init(project="two-tower-search")
    artifact = run.use_artifact('nigelkiernan-lpt-advisory/two-tower-search/model-checkpoint:v1')
    artifact_dir = artifact.download()
    
    model = TwoTowerModel()
    model.load_state_dict(torch.load(f"{artifact_dir}/model.pth"))
    model.eval()
    return model

model = load_model_from_wandb()

class Query(BaseModel):
    text: str

class SearchResponse(BaseModel):
    relevant_docs: List[Dict[str, any]]
    
@app.get("/ping")
def ping():
    return "ok"

@app.get("/version")
def version():
    return {"version": MODEL_VERSION}

@app.get("/logs")
def logs():
    return read_logs()

@app.post("/search")
async def search(query: Query):
    start_time = datetime.now()
    
    # Get prediction
    results = get_prediction(query.text)
    
    end_time = datetime.now()
    latency = (end_time - start_time).total_seconds() * 1000
    
    # Log the request
    log_request({
        "timestamp": start_time.isoformat(),
        "version": MODEL_VERSION,
        "input": query.text,
        "results": results,
        "latency_ms": latency
    })
    
    return {"relevant_docs": results}

def get_prediction(query_text: str):
    # Preprocess query similar to training
    # Use your model's tokenizer/preprocessing steps
    
    # Encode query using the query tower
    with torch.no_grad():
        query_embedding = model.encode_query(query_text)
    
    # Compare with pre-cached document embeddings
    # Return top k results
    # You'll need to implement document caching and similarity search
    
    return results  # Return list of relevant documents

def log_request(message: dict):
    os.makedirs(LOG_DIR_PATH, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(message) + "\n")

def read_logs():
    if not os.path.exists(LOG_PATH):
        return {"logs": []}
    
    with open(LOG_PATH, "r") as f:
        logs = [json.loads(line) for line in f]
    return {"logs": logs}
