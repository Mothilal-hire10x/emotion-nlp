# 🎭 Emotion Detection System

A Machine Learning-based web application that detects emotions from text using Natural Language Processing (NLP) techniques.

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Supported Emotions](#supported-emotions)
- [NLP Pipeline](#nlp-pipeline)
- [Screenshots](#screenshots)

## ✨ Features

- 🔍 Real-time emotion detection from text
- 📊 Confidence scores and probability distribution
- 🎨 Modern, responsive web interface
- 🤖 Machine Learning powered classification
- 📝 NLP preprocessing pipeline

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.x | Programming Language |
| Flask | Web Framework |
| scikit-learn | Machine Learning |
| NLTK | Natural Language Processing |
| Pandas | Data Manipulation |
| HTML/CSS | Frontend |

## 📁 Project Structure

```
emotion-nlp/
│
├── dataset/
│   └── emotions.csv          # Training dataset
│
├── model/
│   ├── emotion_model.pkl     # Trained ML model
│   └── tfidf_vectorizer.pkl  # TF-IDF vectorizer
│
├── templates/
│   └── index.html            # Web interface
│
├── static/
│   └── style.css             # CSS styling
│
├── app.py                    # Flask application
├── train_model.py            # Model training script
├── preprocess.py             # NLP preprocessing module
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone/Download the Project

```bash
cd emotion-nlp
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk flask joblib
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### Step 1: Train the Model

First, train the machine learning model using the dataset:

```bash
python train_model.py
```

**Expected Output:**
```
==================================================
EMOTION DETECTION MODEL TRAINING
==================================================
Loading dataset from: dataset/emotions.csv
Dataset loaded successfully! Shape: (36, 2)
...
Model saved to: model\emotion_model.pkl
Vectorizer saved to: model\tfidf_vectorizer.pkl
Training pipeline completed successfully!
```

### Step 2: Run the Web Application

Start the Flask server:

```bash
python app.py
```

**Expected Output:**
```
Model and vectorizer loaded successfully!
==================================================
EMOTION DETECTION WEB APPLICATION
==================================================
Starting Flask server...
Open http://127.0.0.1:5000 in your browser
==================================================
```

### Step 3: Open in Browser

Open your web browser and go to:

```
http://127.0.0.1:5000
```

## 📖 Usage

1. **Enter Text**: Type or paste any text in the input box
2. **Click Analyze**: Press the "Analyze Emotion" button
3. **View Results**: See the detected emotion with:
   - Emotion label and emoji
   - Confidence percentage
   - Probability bars for all emotions

### Example Inputs

| Input Text | Expected Emotion |
|------------|------------------|
| "I am very happy today!" | Joy 😊 |
| "I feel sad and lonely" | Sadness 😢 |
| "This makes me so angry!" | Anger 😠 |
| "I'm scared of the dark" | Fear 😨 |
| "Wow, I didn't expect that!" | Surprise 😲 |
| "Today is a normal day" | Neutral 😐 |

## 🔌 API Endpoints

### Web Interface
- **URL**: `GET /`
- **Description**: Renders the main web interface

### Predict (Form)
- **URL**: `POST /predict`
- **Content-Type**: `application/x-www-form-urlencoded`
- **Parameter**: `text` - The text to analyze

### Predict (API)
- **URL**: `POST /api/predict`
- **Content-Type**: `application/json`
- **Body**:
```json
{
    "text": "I am feeling happy today!"
}
```
- **Response**:
```json
{
    "emotion": "joy",
    "confidence": 85.5,
    "emoji": "😊",
    "all_probabilities": {
        "joy": 85.5,
        "sadness": 5.2,
        "anger": 3.1,
        "fear": 2.8,
        "surprise": 2.0,
        "neutral": 1.4
    }
}
```

## 🎭 Supported Emotions

| Emotion | Emoji | Description |
|---------|-------|-------------|
| Joy | 😊 | Happiness, excitement, cheerfulness |
| Sadness | 😢 | Sorrow, loneliness, depression |
| Anger | 😠 | Frustration, annoyance, rage |
| Fear | 😨 | Anxiety, nervousness, worry |
| Surprise | 😲 | Shock, amazement, unexpectedness |
| Neutral | 😐 | No strong emotion, factual |

## ⚙️ NLP Pipeline

The text preprocessing pipeline includes:

1. **Lowercasing** - Convert text to lowercase
2. **URL Removal** - Remove web links
3. **Punctuation Removal** - Remove special characters
4. **Number Removal** - Remove digits
5. **Tokenization** - Split text into words
6. **Stopword Removal** - Remove common words (the, is, and, etc.)
7. **Lemmatization** - Convert words to base form (running → run)

## 🧪 Testing the Preprocessing

```python
from preprocess import preprocess_text

text = "I am VERY happy today!!!"
result = preprocess_text(text)
print(result)  # Output: "happy today"
```

## 📊 Model Information

- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: Up to 5000 TF-IDF features
- **Train/Test Split**: 80/20

## 🔧 Troubleshooting

### Model not found error
```
Error: Model files not found. Please run train_model.py first.
```
**Solution**: Run `python train_model.py` before starting the app.

### NLTK data not found
```
LookupError: Resource punkt not found
```
**Solution**: The script automatically downloads NLTK data. If it fails, manually run:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Port already in use
```
Address already in use
```
**Solution**: Change the port in `app.py` or kill the existing process.

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Emotion Detection Project - Mothilal

---

**Built with ❤️ using Python, Flask, and Machine Learning**
