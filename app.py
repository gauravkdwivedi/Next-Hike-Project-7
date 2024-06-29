from flask import Flask, request, render_template, jsonify
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure the required NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')

app = Flask(__name__)

# Load pretrained model, vectorizer, and feature names
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)

# Define preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sentiment_analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def preprocess_text(text):
    if text is None:
        return ''
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

def compute_sentiment_score(text):
    return sentiment_analyzer.polarity_scores(text)['compound']

def extract_additional_features(text):
    return len(text)

def transform_text(text):
    preprocessed_text = preprocess_text(text)
    sentiment_score = compute_sentiment_score(preprocessed_text)
    tweet_length = extract_additional_features(preprocessed_text)
    return preprocessed_text, sentiment_score, tweet_length

def vectorize_text(text):
    return tfidf_vectorizer.transform([text]).toarray()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('tweet')
    if not text:
        return jsonify(error="Please enter a tweet to analyze."), 400

    preprocessed_text, sentiment_score, tweet_length = transform_text(text)
    vectorized_text = vectorize_text(preprocessed_text)
    features = np.concatenate((vectorized_text, np.array([[sentiment_score, tweet_length]])), axis=1)

    # Debugging statements
    print(f"Original Text: {text}")
    print(f"Preprocessed Text: {preprocessed_text}")
    print(f"Sentiment Score: {sentiment_score}")
    print(f"Tweet Length: {tweet_length}")
    print(f"Vectorized Text Shape: {vectorized_text.shape}")
    print(f"Features Shape: {features.shape}")

    # Convert features to DataFrame with the saved feature names
    features_df = pd.DataFrame(features, columns=feature_names)
    print(f"Features DataFrame Shape: {features_df.shape}")
    print(f"Features DataFrame Columns: {features_df.columns}")
    print(f"Features DataFrame: \n{features_df}")

    prediction = model.predict(features_df)
    print(f"Prediction: {prediction}")

    return jsonify(prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)