from flask import Flask, request, render_template
import re
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

# Load pretrained model and vectorizers
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Define preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sentiment_analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text):
    # Clean text
    if text is None:
        return ''
    else:
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stop words and lemmatize tokens
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

def apply_pca(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    n_components = 50
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(scaled_features)
    return pca_features

# Home route to render HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('tweet')
    preprocessed_text, sentiment_score, tweet_length = transform_text(text)
    vectorized_text = vectorize_text(preprocessed_text)
    features = np.concatenate((vectorized_text, np.array([[sentiment_score, tweet_length]])), axis=1)
    pca_features = apply_pca(features)
    prediction = model.predict(pca_features)
    return render_template('index.html', prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)