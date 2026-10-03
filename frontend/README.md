# AI OSINT Analyst — Frontend Dashboard v1

This is a dependency-free first dashboard for the existing FastAPI endpoint:

`POST http://127.0.0.1:8008/analyze?keyword=<company>`

## Features

- Search company keyword
- Loading state
- Error state + retry
- Company overview
- News signals with relevance badges
- GitHub signals with relevance badges
- Claims with confidence bars
- Evidence registry with source URLs/snippets

## Run

From this folder, serve the static files with a local HTTP server:

```powershell
python -m http.server 5173
```

Open:

`http://127.0.0.1:5173`

The frontend calls the FastAPI backend at `http://127.0.0.1:8008`.

## CORS

Because the dashboard is served from port 5173 and the API runs on port 8008, FastAPI must allow the frontend origin.

Add this to `backend/main.py` after `app = FastAPI()`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
