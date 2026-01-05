# language-detection-project
Language identification using Transformer models with Papluca dataset

language-detection-project/
├── notebooks/ # Jupyter notebooks for exploration
├── src/ # Source code
├── experiments/ # Experiment results and analysis
├── browser_extension/ # Chrome extension files
├── configs/ # Configuration files
├── tests/ # Unit tests
└── docs/ # Documentation

## Baseline Model

A TF-IDF + Logistic Regression model was implemented as a baseline.

### Configuration:
- **TF-IDF Parameters**: max_features=10000, ngram_range=(1,2), min_df=5, max_df=0.7
- **Logistic Regression**: C=1.0, class_weight='balanced', solver='lbfgs'

### Results:
- **Test Accuracy**: 90.69%
- **Test F1 Score**: 90.56%
- **Training Time**: < 5 minutes

### Saved Artifacts:
- `models/baseline_tfidf_lr.pkl` - Trained model
- `models/tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `results/baseline_results.json` - Performance metrics