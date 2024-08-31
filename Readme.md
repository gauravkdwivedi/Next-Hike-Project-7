# Sentiment Analysis with Logistic Regression

## Project Overview

This project performs sentiment analysis on text data using a logistic regression model. The goal is to classify text into sentiment categories by extracting features such as tokenized text, sentiment scores, and text length.

## Project Structure

- `preprocess.py` - Contains functions for text preprocessing and feature extraction.
- `vectorize.py` - Contains functions for text vectorization using `CountVectorizer` and `TfidfVectorizer`.
- `train_model.py` - Script to train the logistic regression model.
- `predict.py` - Script to make predictions using the trained model.
- `logistic_regression_model.pkl` - Serialized logistic regression model.
- `README.md` - This readme file.

## Requirements

Make sure you have the following packages installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `nltk`
- `textblob`

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn nltk textblob