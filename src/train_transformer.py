import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Hugging Face Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    TextClassificationPipeline
)

# For comparison with baseline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix , f1_score
import joblib

TRANSFORMER_MODEL_NAME = "papluca/xlm-roberta-base-language-detection"


def load_or_download_data():
    print("Loading dataset...")

    try:
        train_df = pd.read_csv('data/train_data.csv')
        test_df = pd.read_csv('data/test_data.csv')
        print("Loaded from saved CSV files")

    except Exception:
        print("Loading from Hugging Face...")
        from datasets import load_dataset
        import os

        dataset = load_dataset("papluca/language-identification")
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])

        os.makedirs('data', exist_ok=True)
        train_df.to_csv('data/train_data.csv', index=False)
        test_df.to_csv('data/test_data.csv', index=False)
        print("Saved to data/ folder for faster loading next time")

    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    print(f"  Columns: {train_df.columns.tolist()}")

    return train_df, test_df

def prepare_splits(train_df, test_df):
    print("=== DATA PREPROCESSING ===")

    X_train = train_df['text'].astype(str)
    y_train = train_df['labels']

    X_test = test_df['text'].astype(str)
    y_test = test_df['labels']

    print(f"Training data: {len(X_train):,} samples")
    print(f"Test data: {len(X_test):,} samples")

    # Data quality checks
    X_train = X_train.fillna('')
    X_test = X_test.fillna('')
    if y_train.isnull().any() or y_test.isnull().any():
        raise ValueError("Missing labels detected in training or test set.")

    return X_train, X_test, y_train, y_test

def evaluate_baseline(X_train, X_test, y_train, y_test):
    print("=== BASELINE MODEL EVALUATION ===")

    model = joblib.load("notebooks/models/baseline_tfidf_lr.pkl")
    tfidf = joblib.load("notebooks/models/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("notebooks/models/label_encoder.pkl")

    # Encode labels (strings -> ints)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    y_train_pred = model.predict(X_train_tfidf)
    y_test_pred = model.predict(X_test_tfidf)

    train_accuracy = accuracy_score(y_train_enc, y_train_pred)
    test_accuracy = accuracy_score(y_test_enc, y_test_pred)

    train_f1 = f1_score(y_train_enc, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test_enc, y_test_pred, average="weighted")

    print("\nPERFORMANCE METRICS:")
    print(f"  Training Accuracy: {train_accuracy:.2%}")
    print(f"  Test Accuracy:     {test_accuracy:.2%}")
    print(f"  Training F1 Score: {train_f1:.2%}")
    print(f"  Test F1 Score:     {test_f1:.2%}")
   

    return test_accuracy, y_test_pred, confusion_matrix(y_test_enc, y_test_pred)

def load_transformer_model(model_name):
    print("=== LOADING PRE-TRAINED XLM-ROBERTA ===")
    print(f"Loading model: {model_name}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    print("\nModel loaded successfully!")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Number of labels: {model.config.num_labels}")
    print(f"   Model parameters: {model.num_parameters():,}")
    print(f"   Max sequence length: {tokenizer.model_max_length}")

    # Check label mapping
    print("\nModel label mapping:")
    if hasattr(model.config, "id2label"):
        for idx, label in model.config.id2label.items():
            print(f"   {idx}: {label}")
    else:
        print("   No label mapping in model config")

    # Tokenization sanity check
    print("\nTokenization test:")
    sample_text = "Hello, how are you today?"
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)

    print(f"   Text: '{sample_text}'")
    print(f"   Tokens: {tokens}")
    print(f"   Token IDs: {token_ids[:10]}... (length: {len(token_ids)})")
    print(f"   Special tokens: [CLS]={tokenizer.cls_token}, [SEP]={tokenizer.sep_token}")

    return tokenizer, model

def evaluate_transformer(X_test, y_test, tokenizer, model, label_encoder, batch_size=32):
    print("=== TRANSFORMER MODEL EVALUATION ===")

    print("Creating inference pipeline...")
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU
        truncation=True,
        padding=True,
        max_length=512
    )

    print("Pipeline created!")
    print(f"  Device: {'GPU' if classifier.device.type == 'cuda' else 'CPU'}")

    # ---- Sanity check on known samples ----
    print("\nSanity check on sample texts:")
    sanity_samples = [
        ("en", "Hello, how are you doing today?"),
        ("fr", "Bonjour, comment allez-vous aujourd'hui?"),
        ("es", "Hola, ¿cómo estás hoy?"),
        ("de", "Hallo, wie geht es dir heute?"),
    ]

    for lang, text in sanity_samples:
        result = classifier(text[:512])[0]
        print(f"  Text: {text[:35]}...")
        print(f"    True: {lang} | Pred: {result['label']} | Conf: {result['score']:.2%}")

    # ---- Batch inference on test set ----
    print(f"\nRunning batch inference on {len(X_test):,} samples...")

    transformer_preds = []
    transformer_confs = []

    for i in range(0, len(X_test), batch_size):
        batch_texts = X_test.iloc[i:i + batch_size].tolist()
        batch_texts = [text[:512] for text in batch_texts]

        results = classifier(batch_texts)

        for res in results:
            transformer_preds.append(res["label"])
            transformer_confs.append(res["score"])

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(X_test))}/{len(X_test)}")

    # ---- Encode labels for metric computation ----
    y_test_enc = label_encoder.transform(y_test)
    y_pred_enc = label_encoder.transform(transformer_preds)

    accuracy = accuracy_score(y_test_enc, y_pred_enc)
    f1 = f1_score(y_test_enc, y_pred_enc, average="weighted")
    avg_conf = float(np.mean(transformer_confs))

    print("\nTRANSFORMER PERFORMANCE:")
    print(f"  Accuracy:          {accuracy:.2%}")
    print(f"  F1 Score:          {f1:.2%}")
    print(f"  Avg. confidence:   {avg_conf:.2%}")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "avg_confidence": avg_conf,
        "predictions": transformer_preds,
        "confidences": transformer_confs,
    }


def main():
    # ---- Load and prepare data ----
    train_df, test_df = load_or_download_data()

    X_train, X_test, y_train, y_test = prepare_splits(
        train_df, test_df
    )

    # ---- Load label encoder (shared by both models) ----
    label_encoder = joblib.load("notebooks/models/label_encoder.pkl")

    # ---- Evaluate baseline model ----
    baseline_accuracy, baseline_preds, baseline_cm = evaluate_baseline(
        X_train, X_test, y_train, y_test
    )

    # ---- Load transformer model ----
    tokenizer, transformer_model = load_transformer_model(
        TRANSFORMER_MODEL_NAME
    )

    # ---- Evaluate transformer ----
    transformer_results = evaluate_transformer(
        X_test,
        y_test,
        tokenizer,
        transformer_model,
        label_encoder
    )

    # ---- Compare results ----
    print("\n=== MODEL COMPARISON ===")
    print(f"Baseline accuracy:    {baseline_accuracy:.2%}")
    print(f"Transformer accuracy: {transformer_results['accuracy']:.2%}")
    print(f"Accuracy difference:  {transformer_results['accuracy'] - baseline_accuracy:+.2%}")


if __name__ == "__main__":
    main()