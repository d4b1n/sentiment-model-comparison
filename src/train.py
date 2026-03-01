import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

def main():
    model_name = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
    output_dir = os.environ.get("OUTPUT_DIR", "outputs/baseline")

    max_train_samples = int(os.environ.get("MAX_TRAIN_SAMPLES", "2000"))
    max_eval_samples = int(os.environ.get("MAX_EVAL_SAMPLES", "1000"))
    epochs = float(os.environ.get("EPOCHS", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))

    dataset = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_ds = dataset["train"].shuffle(seed=42).select(range(max_train_samples)).map(tokenize_fn, batched=True)
    test_ds  = dataset["test"].shuffle(seed=42).select(range(max_eval_samples)).map(tokenize_fn, batched=True)

    train_ds = train_ds.remove_columns(["text"]).with_format("torch")
    test_ds  = test_ds.remove_columns(["text"]).with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    os.makedirs("results", exist_ok=True)
    row = {
        "model": model_name,
        "train_samples": max_train_samples,
        "eval_samples": max_eval_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "accuracy": metrics.get("eval_accuracy"),
        "f1": metrics.get("eval_f1"),
    }

    metrics_path = "results/metrics.csv"
    df = pd.DataFrame([row])
    if os.path.exists(metrics_path):
        old = pd.read_csv(metrics_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(metrics_path, index=False)

    print("Saved:", metrics_path)
    print(row)

if __name__ == "__main__":
    main()
