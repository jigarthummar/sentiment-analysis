# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import uvicorn
import os
import tempfile
from google.cloud import storage
from typing import List, Optional
import functools

# Define the SentimentClassifier model
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # Binary classification
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = output.last_hidden_state
        # Use the first token ([CLS]) embedding as the sentence representation
        cls_embedding = last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU")
else:
    device = torch.device("cpu")
    print(f"Using CPU")
print(f"Using device: {device}")

# Initialize GCP storage client
storage_client = storage.Client()

# Constants
BUCKET_NAME = "distilbert-model-6010"  # Replace with your bucket name
MODEL_BLOB_PATH = "distilbert_model/model.pth"  # Path to model in bucket
TOKENIZER_BLOB_PATH = "distilbert_model/tokenizer"  # Path to tokenizer in bucket

@functools.lru_cache(maxsize=1)
def load_model_and_tokenizer():
    """
    Downloads and loads model and tokenizer from GCP bucket.
    Uses lru_cache to only do this once.
    """
    model = None
    tokenizer = None
    
    try:
        # Create temporary directories for model and tokenizer
        model_temp_dir = tempfile.mkdtemp()
        tokenizer_temp_dir = tempfile.mkdtemp()
        
        # Download model from bucket
        model_local_path = os.path.join(model_temp_dir, "model.pth")
        model_blob = storage_client.bucket(BUCKET_NAME).blob(MODEL_BLOB_PATH)
        model_blob.download_to_filename(model_local_path)
        print(f"Downloaded model to {model_local_path}")
        
        # Load the model
        model = torch.load(model_local_path, map_location=device, weights_only=False)
        model.eval()  # Set the model to evaluation mode
        
        # --- Patch to fix the _use_sdpa error ---
        if not hasattr(model.bert, '_use_sdpa'):
            model.bert._use_sdpa = False
        # -----------------------------------------
        
        # Download tokenizer files from bucket
        tokenizer_prefix = TOKENIZER_BLOB_PATH
        blobs = storage_client.bucket(BUCKET_NAME).list_blobs(prefix=tokenizer_prefix)
        
        # Create local tokenizer directory
        os.makedirs(tokenizer_temp_dir, exist_ok=True)
        
        # Download each tokenizer file
        for blob in blobs:
            filename = os.path.basename(blob.name)
            local_file_path = os.path.join(tokenizer_temp_dir, filename)
            blob.download_to_filename(local_file_path)
            print(f"Downloaded {blob.name} to {local_file_path}")
        
        # Initialize tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_temp_dir)
        
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
    
    return model, tokenizer

# Define the SentimentClassifierPipeline
class SentimentClassifierPipeline:
    def __init__(self, model, tokenizer, max_length=256, device=device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.model.eval()

    def preprocess(self, text):
        # Handle both string and list inputs
        if isinstance(text, list):
            text = text[0]
            
        encodings = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encodings

    def predict(self, text):
        encodings = self.preprocess(text)
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item() * 100
            
            # Get probabilities for both classes
            neg_prob = probs[0][0].item() * 100
            pos_prob = probs[0][1].item() * 100
            
        return {
            'class': pred_class, 
            'sentiment': 'Negative' if pred_class == 0 else 'Positive',
            'confidence': confidence,
            'text': text,
            'probabilities': {
                'negative': neg_prob,
                'positive': pos_prob
            }
        }

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using a DistilBERT model",
    version="1.0.0"
)

# Pydantic models for request and response
class SentimentRequest(BaseModel):
    text: str

class BatchSentimentRequest(BaseModel):
    texts: List[str]

class SentimentProbabilities(BaseModel):
    negative: float
    positive: float

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: SentimentProbabilities

# Lazily load model and tokenizer on first API call
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        model, tokenizer = load_model_and_tokenizer()
        if model is not None and tokenizer is not None:
            pipeline = SentimentClassifierPipeline(model, tokenizer, device=device)
    return pipeline

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API", "status": "active"}

@app.get("/health")
def health_check():
    pipeline = get_pipeline()
    if pipeline is None:
        return {"status": "error", "message": "Model or tokenizer failed to load"}
    return {"status": "healthy", "model": "DistilBERT for sentiment analysis"}

@app.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        result = pipeline.predict(request.text)
        return {
            "text": request.text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "probabilities": {
                "negative": result["probabilities"]["negative"],
                "positive": result["probabilities"]["positive"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/analyze-batch", response_model=List[SentimentResponse])
def analyze_batch(request: BatchSentimentRequest):
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    results = []
    for text in request.texts:
        try:
            result = pipeline.predict(text)
            results.append({
                "text": text,
                "sentiment": result["sentiment"],
                "confidence": result["confidence"],
                "probabilities": {
                    "negative": result["probabilities"]["negative"],
                    "positive": result["probabilities"]["positive"]
                }
            })
        except Exception as e:
            results.append({
                "text": text,
                "sentiment": "Error",
                "confidence": 0.0,
                "probabilities": {
                    "negative": 0.0,
                    "positive": 0.0
                }
            })
    
    return results

if __name__ == "__main__":
    # Use the PORT environment variable for Cloud Run
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)