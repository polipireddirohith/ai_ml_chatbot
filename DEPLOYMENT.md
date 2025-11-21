# Deployment Guide

This project consists of two parts:
1.  **Frontend**: React + Vite (located in root)
2.  **Backend**: FastAPI + Python (located in `backend/`)

To deploy this project online, the recommended (and free) approach is to host them separately.

## 1. Deploy Backend (Render.com)
The backend needs a server to run Python and the ML models.

1.  Push this code to a **GitHub repository**.
2.  Sign up at [Render.com](https://render.com).
3.  Click **New +** -> **Web Service**.
4.  Connect your GitHub repo.
5.  **Settings**:
    *   **Root Directory**: `backend`
    *   **Runtime**: Python 3
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6.  Click **Deploy**.
7.  **Copy the URL** provided by Render (e.g., `https://neurochat-api.onrender.com`).

## 2. Configure Frontend
You need to tell the frontend where the backend is hosted.

1.  Open `src/components/ChatInterface.jsx`.
2.  Find the `fetch` call (around line 100).
3.  Replace `http://localhost:8000/chat` with your **Render URL** + `/chat`.
    *   Example: `https://neurochat-api.onrender.com/chat`
4.  (Optional) For a cleaner setup, use an environment variable:
    *   Create `.env.production` in the root.
    *   Add: `VITE_API_URL=https://neurochat-api.onrender.com`
    *   Update code to use `import.meta.env.VITE_API_URL`.

## 3. Deploy Frontend (Vercel)
1.  Sign up at [Vercel.com](https://vercel.com).
2.  Click **Add New...** -> **Project**.
3.  Import your GitHub repo.
4.  **Settings**:
    *   **Framework Preset**: Vite
    *   **Root Directory**: `./` (default)
5.  Click **Deploy**.

## 4. Docker Deployment (Alternative)
If you prefer using Docker, a `backend/Dockerfile` has been provided.
1.  Build: `docker build -t neurochat-backend ./backend`
2.  Run: `docker run -p 8000:8000 neurochat-backend`
