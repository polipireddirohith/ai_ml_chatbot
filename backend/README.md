# Backend Integration

## Setup

1.  **Install Python Dependencies**:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

2.  **Run the Backend Server**:
    ```bash
    cd backend
    python main.py
    ```
    The server will start at `http://localhost:8000`.

## API Endpoints

-   `GET /`: Status check.
-   `POST /chat`: Main chat endpoint.
    -   Payload: `{ "message": "hello", "model_id": "general" }`
    -   Response: `{ "response": "...", "model_used": "general", "timestamp": 1234567890 }`

## Customizing Models

Edit `backend/main.py` to implement your actual AI logic in the `process_*` functions (e.g., `process_cnn`, `process_rnn`).
