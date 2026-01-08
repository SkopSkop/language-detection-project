import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path

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


from torch.utils.data import Dataset, DataLoader


# RUN CONFIGURATION

RUN_BASELINE = True

RUN_SANITY_CHECK = True

RUN_PRETRAINED_TRAINING = False
RUN_PRETRAINED_EVALUATION = True

RUN_SCRATCH_TRAINING = False
RUN_SCRATCH_EVALUATION = True

RUN_SHORT_LONG_ANALYSIS = True


SUBSET_SIZE = 5000          
MAX_LENGTH = 64
PRETRAINED_EPOCHS = 2
SCRATCH_EPOCHS = 1

TRANSFORMER_MODEL_NAME = "papluca/xlm-roberta-base-language-detection"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


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

    required_files = [
        MODEL_DIR / "baseline_tfidf_lr.pkl",
        MODEL_DIR / "tfidf_vectorizer.pkl",
        MODEL_DIR / "label_encoder.pkl",
    ]

    if not all(p.exists() for p in required_files):
        print("Baseline artifacts not found. Skipping baseline evaluation.")
        return None, None, None

    model = joblib.load(MODEL_DIR / "baseline_tfidf_lr.pkl")
    tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

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

def evaluate_transformer(X_test, y_test, tokenizer, model,batch_size=32):
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

    print(f"saving trained model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer

def split_by_text_length(X, y, short_max=50, long_min=150):
    lengths = X.str.len()

    short_mask = lengths <= short_max
    long_mask = lengths >= long_min

    X_short = X[short_mask]
    y_short = y[short_mask]

    X_long = X[long_mask]
    y_long = y[long_mask]

    return (X_short, y_short), (X_long, y_long)

def evaluate_on_subset(name, X_subset, y_subset, tokenizer, model):
    print(f"\n--- {name} ---")
    print(f"Samples: {len(X_subset):,}")

    results = evaluate_transformer(
        X_subset,
        y_subset,
        tokenizer,
        model
    )

    return results

def sanity_check_pretrained_model(model_name):
    print("\n=== SANITY CHECK: CLEAN PRETRAINED MODEL ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        truncation=True,
        padding=True,
        max_length=64
    )

    samples = [
        ("en", "Hello, how are you doing today?"),
        ("fr", "Bonjour, comment allez-vous aujourd'hui?"),
        ("es", "Hola, ¿cómo estás hoy?"),
        ("de", "Hallo, wie geht es dir heute?"),
    ]



def main():
    print("DEBUG: main() version = 2024-EXPERIMENT-FIX")
    results = {}
    # 1. Load and prepare data
 
    train_df, test_df = load_or_download_data()
    X_train, X_test, y_train, y_test = prepare_splits(train_df, test_df)

    if SUBSET_SIZE is not None:
        X_train = X_train.sample(n=SUBSET_SIZE, random_state=42)
        y_train = y_train.loc[X_train.index]

    print(f"Training samples used: {len(X_train):,}")

  
    # 2. Load label encoder
    print("\n=== LOADING LABEL ENCODER ===")
    label_encoder = None
    num_labels = None

    if RUN_BASELINE or RUN_PRETRAINED_TRAINING or RUN_SCRATCH_TRAINING:
        label_encoder_path = MODEL_DIR / "label_encoder.pkl"

        if label_encoder_path.exists():
            label_encoder = joblib.load(label_encoder_path)
            num_labels = len(label_encoder.classes_)
        else:
            print("Label encoder not found. Skipping baseline/training.")


    pretrained_model = None
    scratch_model = None



    # 3. Baseline evaluation
   
    if RUN_BASELINE and label_encoder is not None:
     baseline_accuracy, _, _ = evaluate_baseline(
            X_train, X_test, y_train, y_test
        )
    else:
        baseline_accuracy = None


    # 4. Sanity check (clean pretrained model)

    if RUN_SANITY_CHECK:
        sanity_check_pretrained_model(TRANSFORMER_MODEL_NAME)

    # 5. Shared tokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)


    # 6. Tokenized datasets (for training)

    if label_encoder is not None:
        train_dataset = tokenize_dataset(
         X_train, y_train, tokenizer, label_encoder, max_length=MAX_LENGTH
        )
        test_dataset = tokenize_dataset(
            X_test, y_test, tokenizer, label_encoder, max_length=MAX_LENGTH
        )
    else:
        train_dataset = None
        test_dataset = None
    

    pretrained_model = None
    scratch_model = None


    # 7. Pretrained transformer

    if RUN_PRETRAINED_TRAINING and label_encoder is not None:
        print("\n=== EXPERIMENT 1: PRETRAINED TRANSFORMER (TRAINING) ===")

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
            output_dir=MODEL_DIR / "transformer_finetuned",
            epochs=PRETRAINED_EPOCHS
        )

    if RUN_PRETRAINED_EVALUATION:
        print("\n=== EXPERIMENT 1: PRETRAINED TRANSFORMER (EVALUATION) ===")

        model_path = MODEL_DIR / "transformer_finetuned"

        if pretrained_model is None:
            if model_path.exists():
                print("Loading pretrained fine-tuned model from disk...")
                pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    local_files_only=True
                )
            else:
                print("Fine-tuned model not found. Using base pretrained model.")
                pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                    TRANSFORMER_MODEL_NAME
                )
        pretrained_results = evaluate_transformer(
            X_test,
            y_test,
            tokenizer,
            pretrained_model
        )
    else:
        pretrained_results = None

    if pretrained_results is not None:
        results["pretrained"] = pretrained_results

    # 8. Transformer from scratch

    if RUN_SCRATCH_TRAINING and label_encoder is not None:
        print("\n=== EXPERIMENT 2: TRANSFORMER FROM SCRATCH (TRAINING) ===")

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
            output_dir=MODEL_DIR / "transformer_scratch",
            epochs=SCRATCH_EPOCHS
        )

    if RUN_SCRATCH_EVALUATION:
        print("\n=== EXPERIMENT 2: TRANSFORMER FROM SCRATCH (EVALUATION) ===")

        model_path = MODEL_DIR / "transformer_scratch"

        if scratch_model is None:
            if model_path.exists():
                print("Loading scratch-trained model from disk...")
                scratch_model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    local_files_only=True
                )
            else:
                print("Scratch-trained model not found. Using untrained model.")
                config = AutoConfig.from_pretrained(
                    TRANSFORMER_MODEL_NAME,
                   num_labels=num_labels
                )
                scratch_model = AutoModelForSequenceClassification.from_config(config)

        scratch_results = evaluate_transformer(
            X_test,
            y_test,
            tokenizer,
            scratch_model
        )
    else:
        scratch_results = None

    if scratch_results is not None:
        results["scratch"] = scratch_results



    # 9. Short vs long text analysis

    if RUN_SHORT_LONG_ANALYSIS and pretrained_results is not None:
        print("\n=== SHORT vs LONG TEXT ANALYSIS (PRETRAINED) ===")

        (X_short, y_short), (X_long, y_long) = split_by_text_length(
            X_test, y_test
        )

        print("\n[Short texts]")
        short_results = evaluate_transformer(
            X_short,
            y_short,
            tokenizer,
            pretrained_model
        )

        print("\n[Long texts]")
        long_results = evaluate_transformer(
            X_long,
            y_long,
            tokenizer,
            pretrained_model
        )

        results["short_long"] = {
            "short": short_results,
            "long": long_results
        }
    # 10. Final comparison

    print("\n=== FINAL MODEL COMPARISON ===")

    if baseline_accuracy is not None:
        print(f"Baseline accuracy:           {baseline_accuracy:.2%}")

    if pretrained_results is not None:
        print(f"Pretrained transformer:      {pretrained_results['accuracy']:.2%}")

    if scratch_results is not None:
        print(f"Transformer (from scratch):  {scratch_results['accuracy']:.2%}")

    return results


if __name__ == "__main__":
    main()