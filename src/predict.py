import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "outputs/final_model"

LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}

def predict(text: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())
    pred_label = LABEL_MAP[pred_id]
    confidence = float(probs[pred_id].item())

    return pred_label, confidence, probs.tolist()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "I love this movie"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    label, conf, probs = predict(text)

    print(f"Text: {text}")
    print(f"Prediction: {label} (confidence={conf:.4f})")
    print(f"Probabilities [NEG, POS]: {[round(p, 4) for p in probs]}")