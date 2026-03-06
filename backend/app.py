from __future__ import annotations

import os
from typing import Optional, Dict, Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# Config
# =========================
MODEL_DIR = os.environ.get("MODEL_DIR", "../outputs/final_model")

# HuggingFace IMDB 기본 라벨은 보통 0=NEGATIVE, 1=POSITIVE
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}


# =========================
# App
# =========================
app = FastAPI(title="Sentiment API", version="0.1.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # MVP라 일단 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify")


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.on_event("startup")
def load_model_on_startup() -> None:
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "device": device, "model_dir": MODEL_DIR}


@app.post("/predict", response_model=PredictResponse)
@torch.inference_mode()
def predict(req: PredictRequest) -> PredictResponse:
    if model is None or tokenizer is None:
        # startup 로딩 실패 대비
        load_model_on_startup()

    inputs = tokenizer(
    req.text,
    truncation=True,
    max_length=128,
    return_tensors="pt",
     )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)

    pred_id = int(torch.argmax(probs).item())
    conf = float(probs[pred_id].item())

    return PredictResponse(label=ID2LABEL.get(pred_id, str(pred_id)), confidence=round(conf, 4))