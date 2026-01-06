import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Hugging Face Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    TextClassificationPipeline,
    AutoConfig
)

# For comparison with baseline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix , f1_score
import joblib

TRANSFORMER_MODEL_NAME = "papluca/xlm-roberta-base-language-detection"

from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


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
    y_true = y_test.tolist()
    y_pred = transformer_preds

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
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
def build_label_mappings(label_encoder):
    label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def create_transformer_model(model_name, num_labels, label_encoder, pretrained=True):
    label2id, id2label = build_label_mappings(label_encoder)

    if pretrained:
        print("Initializing PRETRAINED transformer model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )
    else:
        print("Initializing TRANSFORMER FROM SCRATCH...")
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )
        model = AutoModelForSequenceClassification.from_config(config)

    return model

def tokenize_dataset(texts, labels, tokenizer, label_encoder, max_length=128):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )

    labels_enc = label_encoder.transform(labels)

    dataset = TextClassificationDataset(
        encodings={
            "input_ids": torch.tensor(encodings["input_ids"]),
            "attention_mask": torch.tensor(encodings["attention_mask"]),
        },
        labels=torch.tensor(labels_enc),
    )

    return dataset

from transformers import Trainer, TrainingArguments

def train_transformer_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
    epochs=2
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(
                p.label_ids, np.argmax(p.predictions, axis=1)
            )
        },
    )

    trainer.train()
    return trainer


def main():
    # ---- Load and prepare data ----
    train_df, test_df = load_or_download_data()
    X_train, X_test, y_train, y_test = prepare_splits(train_df, test_df)

    SUBSET_SIZE = 5000

    X_train = X_train.sample(n=SUBSET_SIZE, random_state=42)
    y_train = y_train.loc[X_train.index]

    # ---- Load label encoder ----
    label_encoder = joblib.load("notebooks/models/label_encoder.pkl")
    num_labels = len(label_encoder.classes_)

    # ---- Baseline evaluation ----
    baseline_accuracy, _, _ = evaluate_baseline(
        X_train, X_test, y_train, y_test
    )

    # ---- Shared tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    # ---- Prepare tokenized datasets ----
    train_dataset = tokenize_dataset(
        X_train, y_train, tokenizer, label_encoder, max_length=64
    )
    test_dataset = tokenize_dataset(
        X_test, y_test, tokenizer, label_encoder, max_length=64
    )

    # ================================
    # Experiment 1: Pretrained model
    # ================================
    print("\n=== EXPERIMENT 1: PRETRAINED TRANSFORMER ===")

    pretrained_model = create_transformer_model(
        TRANSFORMER_MODEL_NAME,
        num_labels,
        label_encoder,
        pretrained=True
    )

    train_transformer_model(
        pretrained_model,
        tokenizer,
        train_dataset,
        test_dataset,
        output_dir="models/transformer_finetuned",
        epochs=2
    )

    pretrained_results = evaluate_transformer(
        X_test,
        y_test,
        tokenizer,
        pretrained_model,
        label_encoder
    )

    # ================================
    # Experiment 2: Training from scratch
    # ================================
    print("\n=== EXPERIMENT 2: TRANSFORMER FROM SCRATCH ===")

    scratch_model = create_transformer_model(
        TRANSFORMER_MODEL_NAME,
        num_labels,
        label_encoder,
        pretrained=False
    )

    train_transformer_model(
        scratch_model,
        tokenizer,
        train_dataset,
        test_dataset,
        output_dir="models/transformer_scratch",
        epochs=1
    )

    scratch_results = evaluate_transformer(
        X_test,
        y_test,
        tokenizer,
        scratch_model,
        label_encoder
    )

    # ================================
    # Final comparison
    # ================================
    print("\n=== FINAL MODEL COMPARISON ===")
    print(f"Baseline accuracy:           {baseline_accuracy:.2%}")
    print(f"Pretrained transformer:      {pretrained_results['accuracy']:.2%}")
    print(f"Transformer (from scratch):  {scratch_results['accuracy']:.2%}")


if __name__ == "__main__":
    main()