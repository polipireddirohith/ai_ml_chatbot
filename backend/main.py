from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import time
import random

from . import models, database, ml_models

# Initialize Database
models.Base.metadata.create_all(bind=database.engine)

# Initialize ML Models (Train on startup)
ml_models.ml_manager.train_dummy_models()

app = FastAPI(title="NeuroChat API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model_id: str

class ChatResponse(BaseModel):
    response: str
    model_used: str
    timestamp: float

# Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Real ML/AI Processing Logic
def process_with_neural_net(message: str, model_type: str):
    """Uses the PyTorch Neural Network (MLP) to classify intent and generate response."""
    intent = ml_models.ml_manager.predict_nn(message)
    
    base_response = f"[{model_type.upper()} Analysis] "
    if intent == "greeting":
        return base_response + "Hello! I am your AI assistant. How can I help you with Neural Networks today?"
    elif intent == "question":
        return base_response + f"That's a good question about '{message}'. I'm processing it using my internal layers."
    elif intent == "tech_query":
        return base_response + "I detected a technical query. Accessing my knowledge base on AI/ML..."
    else:
        return base_response + f"I processed your input '{message}' through my hidden layers."

def process_with_svm(message: str):
    """Uses the SVM model."""
    intent = ml_models.ml_manager.predict_svm(message)
    return f"[SVM Boundary] Input classified as '{intent}'. The support vectors have determined this outcome."

def process_rag(message: str):
    """Uses the RAG system to retrieve knowledge."""
    retrieved_doc = ml_models.ml_manager.query_rag(message)
    if retrieved_doc:
        return f"[RAG Retrieval] Found relevant info: \"{retrieved_doc}\"\n\nBased on this, I can answer your query."
    else:
        return "[RAG Retrieval] No relevant documents found in the knowledge base for this specific query. Try asking about Neural Networks, CNNs, or SVMs."

def process_pca(message: str):
    """Uses PCA to reduce dimensionality of the input."""
    coords = ml_models.ml_manager.get_pca_coords(message)
    return f"[PCA Reduction] Your text input was vectorized and reduced to 2D coordinates: x={coords[0]:.2f}, y={coords[1]:.2f}. This represents the semantic position of your query in our simplified feature space."

def process_genai(message: str):
    """Uses the Generative AI model (DistilGPT-2)."""
    return ml_models.ml_manager.generate_text(message)

@app.get("/")
def read_root():
    return {"status": "online", "message": "NeuroChat Backend is running with PyTorch, SVM, PCA, RAG & Transformers (LLM)"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    # Simulate processing delay for realism
    time.sleep(0.3)
    
    model_id = request.model_id
    message = request.message
    
    response_text = ""
    
    # Route to appropriate model
    if model_id == 'svm':
        response_text = process_with_svm(message)
    elif model_id == 'pca':
        response_text = process_pca(message)
    elif model_id == 'rag':
        response_text = process_rag(message)
    elif model_id == 'mlp' or model_id == 'mcp': # Handling 'mcp' as MLP alias
        response_text = process_with_neural_net(message, "MLP (Multi-Layer Perceptron)")
    elif model_id in ['cnn', 'rnn', 'transformer']:
        # For now, we use the generic Neural Net for these deep learning modes
        response_text = process_with_neural_net(message, model_id)
    elif model_id == 'general':
        # Use real LLM for general chat
        response_text = process_genai(message)
    else:
        response_text = process_with_neural_net(message, "General NN")
        
    # Save to Database
    db_log = models.ChatLog(
        message=message,
        response=response_text,
        model_used=model_id,
        timestamp=time.time()
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
        
    return ChatResponse(
        response=response_text,
        model_used=model_id,
        timestamp=db_log.timestamp
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
