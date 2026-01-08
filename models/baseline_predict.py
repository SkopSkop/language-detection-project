
def predict_language(text):
    """Predict language using baseline model."""
    import joblib
    import numpy as np

    # Load model and vectorizer
    model = joblib.load('models/baseline_tfidf_lr.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')

    # Transform and predict
    text_vectorized = tfidf.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]

    return {
        'predicted_language': int(prediction),
        'confidence': float(np.max(probability)),
        'all_probabilities': probability.tolist()
    }

# Example usage:
# result = predict_language("Hello, how are you?")
# print(f"Predicted: {result['predicted_language']} with {result['confidence']:.2%} confidence")
