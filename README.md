# AI/ML Chatbot Interface

A premium, dark-themed web interface for an AI/ML Chatbot that supports multiple algorithm modes (CNN, RNN, Transformers, etc.).

## Features

-   **Modern UI**: Glassmorphism, gradients, and smooth animations.
-   **Multiple Modes**: Switch between different AI models.
-   **Responsive Design**: Works on desktop and mobile.
-   **Real Backend**: Powered by Python (FastAPI) and PyTorch.
-   **Generative AI**: Integrated with DistilGPT-2 for real text generation.
-   **RAG System**: Retrieval-Augmented Generation for knowledge retrieval.

## Tech Stack

-   **Frontend**: React + Vite
-   **Backend**: FastAPI + Python
-   **ML Libraries**: PyTorch, Scikit-learn, Transformers
-   **Database**: SQLite (via SQLAlchemy)

## Getting Started

### Prerequisites

-   Node.js (v16+)
-   Python (v3.9+)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/polipireddirohith/ai_ml_chatbot.git
    cd ai_ml_chatbot
    ```

2.  **Install Dependencies & Run**:
    *   **Windows**: Double-click `run_backend.bat` (starts both backend and frontend).
    *   **Manual**:
        ```bash
        # Backend
        cd backend
        pip install -r requirements.txt
        uvicorn main:app --reload

        # Frontend
        cd ..
        npm install
        npm run dev
        ```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
