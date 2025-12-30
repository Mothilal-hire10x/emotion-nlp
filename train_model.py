"""
Model Training Script for Emotion Detection
This script:
1. Reads the emotion dataset
2. Applies NLP preprocessing
3. Trains a Machine Learning model (Naive Bayes)
4. Saves the trained model and TF-IDF vectorizer
"""

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from preprocess import preprocess_text


def load_dataset(filepath):
    """Load the emotion dataset from CSV file."""
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    return df


def train_model(df):
    """
    Train the emotion classification model.
    
    Steps:
    1. Preprocess text data
    2. Split into train/test sets
    3. Apply TF-IDF vectorization
    4. Train Naive Bayes classifier
    5. Evaluate model performance
    
    Returns:
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
    """
    print("\n" + "=" * 50)
    print("Step 1: Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    print("Preprocessing complete!")
    
    # Features and labels
    X = df['processed_text']
    y = df['emotion']
    
    print("\n" + "=" * 50)
    print("Step 2: Splitting dataset into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    print("\n" + "=" * 50)
    print("Step 3: Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
    
    print("\n" + "=" * 50)
    print("Step 4: Training Naive Bayes Classifier...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    print("Model training complete!")
    
    print("\n" + "=" * 50)
    print("Step 5: Evaluating Model Performance...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer


def save_model(model, vectorizer, model_dir='model'):
    """Save the trained model and vectorizer to disk."""
    print("\n" + "=" * 50)
    print("Saving model and vectorizer...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'emotion_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    print("\nModel training and saving complete!")


def main():
    """Main function to run the training pipeline."""
    print("=" * 50)
    print("EMOTION DETECTION MODEL TRAINING")
    print("=" * 50)
    
    # File paths
    dataset_path = 'dataset/emotions.csv'
    model_dir = 'model'
    
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Train model
    model, vectorizer = train_model(df)
    
    # Save model
    save_model(model, vectorizer, model_dir)
    
    print("\n" + "=" * 50)
    print("Training pipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
