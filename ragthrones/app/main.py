import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from ragthrones.app.api import router as api_router
from ragthrones.app.gradio_ui import build_ui

# NEW for Gradio 4.x
from gradio.routes import mount_gradio_app


app = FastAPI(title="Cosinify")

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# API routes
# ---------------------------------------------------------
app.include_router(api_router, prefix="/api")

# ---------------------------------------------------------
# Gradio UI (Blocks instance)
# ---------------------------------------------------------
gradio_app = build_ui()

# Mount Gradio at root "/"
# Cloud Run will hit "/"
mount_gradio_app(app, gradio_app, path="/")


# ---------------------------------------------------------
# Redirect root → "/" (Gradio home)
# ---------------------------------------------------------
@app.get("/")
def root():
    # Already mounted at "/", but keep redirect for safety
    return RedirectResponse(url="/")


# ---------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "ragthrones.app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        reload=False,
    )