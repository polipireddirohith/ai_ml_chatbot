import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import pickle
import os

# --- 1. PyTorch Neural Network (MLP) ---
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# --- 2. RAG System ---
class RAGSystem:
    def __init__(self):
        self.knowledge_base = [
            "Neural Networks are computing systems inspired by the biological neural networks that constitute animal brains.",
            "Convolutional Neural Networks (CNNs) are a class of deep neural networks, most commonly applied to analyzing visual imagery.",
            "Recurrent Neural Networks (RNNs) are a class of neural networks where connections between nodes form a directed graph along a temporal sequence.",
            "Transformers are deep learning models that adopt the mechanism of self-attention, differentially weighting the significance of each part of the input data.",
            "Support Vector Machines (SVMs) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis.",
            "Principal Component Analysis (PCA) is a dimensionality reduction method that is often used to reduce the dimensionality of large data sets.",
            "Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "Reinforcement Learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward."
        ]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kb_vectors = None
        
    def fit(self):
        self.kb_vectors = self.vectorizer.fit_transform(self.knowledge_base)
        
    def retrieve(self, query):
        if self.kb_vectors is None:
            self.fit()
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.kb_vectors).flatten()
        best_idx = np.argmax(similarities)
        
        if similarities[best_idx] < 0.1:
            return None
            
        return self.knowledge_base[best_idx]

# --- 3. Generative AI (LLM) ---
class GenAI:
    def __init__(self):
        self.generator = None
        
    def generate_response(self, prompt):
        if self.generator is None:
            print("Loading DistilGPT-2 model... This may take a moment.")
            # Using distilgpt2 for speed and low memory usage
            self.generator = pipeline('text-generation', model='distilgpt2')
            
        # Generate text
        # We limit max_length to keep it snappy
        try:
            response = self.generator(prompt, max_length=100, num_return_sequences=1, truncation=True)
            generated_text = response[0]['generated_text']
            
            # Simple cleanup: remove the prompt from the response if it repeats it
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            if not generated_text:
                return "I'm thinking..."
                
            return generated_text
        except Exception as e:
            return f"Error generating response: {str(e)}"

# --- 4. ML Model Manager ---
class MLManager:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english', max_features=100)
        self.nn_model = None
        self.svm_model = None
        self.pca = None
        self.rag = RAGSystem()
        self.gen_ai = GenAI()
        self.is_trained = False
        self.intents = ["greeting", "question", "tech_query", "other"]
        
    def train_dummy_models(self):
        """Trains simple models on dummy data for demonstration."""
        print("Training dummy ML models...")
        
        # Dummy Dataset
        corpus = [
            "hello", "hi there", "good morning", "hey",
            "what is this?", "how does it work?", "help me", "question",
            "python code", "neural network", "machine learning", "deep learning",
            "random stuff", "weather is nice", "food", "music"
        ]
        # Labels: 0=Greeting, 1=Question, 2=Tech, 3=Other
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        
        # 1. Vectorize
        X = self.vectorizer.fit_transform(corpus).toarray()
        y = np.array(labels)
        
        # 2. Train SVM
        self.svm_model = SVC(kernel='linear')
        self.svm_model.fit(X, y)
        
        # 3. Train Neural Network (PyTorch)
        input_size = X.shape[1]
        hidden_size = 16
        num_classes = 4
        
        self.nn_model = SimpleNN(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()
        
        # Training loop
        for epoch in range(200):
            outputs = self.nn_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 4. Train PCA
        self.pca = PCA(n_components=2)
        self.pca.fit(X)
        
        # 5. Init RAG
        self.rag.fit()
            
        self.is_trained = True
        print("Models trained successfully.")

    def predict_nn(self, text):
        if not self.is_trained:
            self.train_dummy_models()
            
        # Preprocess
        vec = self.vectorizer.transform([text]).toarray()
        vec_tensor = torch.from_numpy(vec).float()
        
        # Inference
        with torch.no_grad():
            outputs = self.nn_model(vec_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
        intent_idx = predicted.item()
        return self.intents[intent_idx]

    def predict_svm(self, text):
        if not self.is_trained:
            self.train_dummy_models()
            
        vec = self.vectorizer.transform([text]).toarray()
        pred = self.svm_model.predict(vec)
        return self.intents[pred[0]]
    
    def get_pca_coords(self, text):
        if not self.is_trained:
            self.train_dummy_models()
        
        vec = self.vectorizer.transform([text]).toarray()
        coords = self.pca.transform(vec)
        return coords[0]

    def query_rag(self, text):
        if not self.is_trained:
            self.train_dummy_models()
        return self.rag.retrieve(text)
    
    def generate_text(self, text):
        return self.gen_ai.generate_response(text)

# Singleton instance
ml_manager = MLManager()
