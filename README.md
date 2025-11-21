# NeuroChat - AI/ML Chatbot Interface

This is a modern, premium chat interface for an AI/ML Chatbot project.
It is built with **Vite + React** and uses **Vanilla CSS** for styling to achieve a custom, high-performance look.

## Features

- **Premium Dark Theme**: Deep space colors with glassmorphism effects.
- **Dynamic Interface**: Smooth animations for messages and sidebar.
- **Model Selection**: Sidebar menu to switch between different AI/ML modes (CNN, RNN, Transformer, etc.).
- **Responsive Design**: Adapts to different screen sizes (basic implementation).

## Project Structure

- `src/components/ChatInterface.jsx`: The main chat component containing the logic and layout.
- `src/components/ChatInterface.css`: Custom styles for the chat interface.
- `src/index.css`: Global design system (variables, typography, reset).

## Running the Project

1.  Install dependencies (if not already done):
    ```bash
    npm install
    ```
2.  Start the development server:
    ```bash
    npm run dev
    ```
3.  Open [http://localhost:5173](http://localhost:5173) in your browser.

## Next Steps

- Connect the frontend to a backend API (Python/Flask/FastAPI) to handle actual model inference.
- Implement state management for chat history persistence.
- Add more interactive visualizations for the specific AI models (e.g., showing a CNN architecture diagram).
