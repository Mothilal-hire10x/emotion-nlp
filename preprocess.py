"""
NLP Preprocessing Module for Emotion Detection
This module contains text preprocessing functions used before ML classification.
"""

import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources (run once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Preprocess text for emotion classification.
    
    Steps performed:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove punctuation
    4. Remove numbers
    5. Tokenization
    6. Stopword removal
    7. Lemmatization
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 4. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 5. Tokenization
    tokens = word_tokenize(text)

    # 6. Stopword removal
    tokens = [word for word in tokens if word not in stop_words]

    # 7. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# Test the preprocessing function
if __name__ == "__main__":
    test_sentences = [
        "I am VERY happy today!!!",
        "This is so SAD and depressing...",
        "I hate this 123 situation https://example.com",
    ]
    
    print("Testing Preprocessing Module")
    print("=" * 50)
    
    for sentence in test_sentences:
        processed = preprocess_text(sentence)
        print(f"Original: {sentence}")
        print(f"Processed: {processed}")
        print("-" * 50)
