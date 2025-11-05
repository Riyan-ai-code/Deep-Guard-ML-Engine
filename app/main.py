from typing import Union
from fastapi import FastAPI
from app.routes.video_detection import router as video_detection_router

app = FastAPI(title="DeepFake Detection API")
app.include_router(video_detection_router)