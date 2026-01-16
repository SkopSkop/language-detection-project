
# Language Identification with Transformers

This project focuses on **automatic language identification** using both classical machine learning techniques and modern **Transformer-based deep learning models**.  
The main objective is to compare a traditional baseline approach with pretrained and non-pretrained Transformer models and to analyze their behavior on different types of text.

---

## Dataset

The project uses the **Papluca Language Identification Dataset**, available on Hugging Face:

https://huggingface.co/datasets/papluca/language-identification

- ~70,000 training samples  
- ~10,000 test samples  
- Over 20 languages  
- Labels are provided as language codes (strings)

The dataset is downloaded automatically on first run and cached locally in the `data/` directory.

---

## Models Used

### Baseline Model
- TF-IDF vectorization
- Logistic Regression classifier

### Deep Learning Models
- XLM-RoBERTa (multilingual Transformer)
  - Pretrained evaluation
  - Fine-tuned model
  - Training-from-scratch experiment (for comparison)

---

## Requirements

Install all dependencies using:

pip install -r requirements.txt

## Running Locally

- Clone the repository
- Install dependencies
- Run the main Script (src/train_transformer.py)

All experiments are controlled via configuration flags at the top of train_transformer.py

RUN_BASELINE = True
RUN_SANITY_CHECK = True

RUN_PRETRAINED_TRAINING = False
RUN_PRETRAINED_EVALUATION = True

RUN_SCRATCH_TRAINING = False
RUN_SCRATCH_EVALUATION = True

RUN_SHORT_LONG_ANALYSIS = True

## Running via Jupyter Notebook

The notebook notebooks/language_identification_model.ipynb contains the complete experimental pipeline and imports the main script

Steps:
- Launch Jupyter Notebook
- Open the notebook
- Run all cells top to bottom

