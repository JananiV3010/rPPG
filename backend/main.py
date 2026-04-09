from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from predict import router as predict_router

app = FastAPI(title="rPPG Fatigue Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/health")
def health():
    return {"status": "ok"}
