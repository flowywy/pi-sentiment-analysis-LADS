import joblib

# Function untuk load model & vectorizer dari file .pkl
def load_model():
    bundle = joblib.load('mnb_model.pkl')
    return bundle['model'], bundle['vectorizer']

# Function untuk klasifikasi sentimen
def classify_sentiment(model, vectorizer, texts):
    # Transform text ke TF-IDF
    tfidf = vectorizer.transform(texts)
    # Prediksi
    preds = model.predict(tfidf)
    # Konversi output menjadi label "positive" atau "negative"
    return ['positive' if p == 1 else 'negative' for p in preds]
