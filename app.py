"""
Flask Web Application for Emotion Detection
This is the main application file that:
1. Loads the trained ML model
2. Provides web interface for emotion prediction
3. Handles user input and returns predictions
"""

from flask import Flask, render_template, request, jsonify
import joblib
import os

from preprocess import preprocess_text

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = 'model/emotion_model.pkl'
VECTORIZER_PATH = 'model/tfidf_vectorizer.pkl'

model = None
vectorizer = None


def load_model():
    """Load the trained model and vectorizer."""
    global model, vectorizer
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model and vectorizer loaded successfully!")
        return True
    else:
        print("Warning: Model files not found. Please run train_model.py first.")
        return False


def predict_emotion(text):
    """
    Predict the emotion of the input text.
    
    Args:
        text (str): Input text to classify
        
    Returns:
        dict: Prediction result with emotion and confidence
    """
    if model is None or vectorizer is None:
        return {"error": "Model not loaded. Please train the model first."}
    
    # Preprocess the input text
    processed_text = preprocess_text(text)
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([processed_text])
    
    # Get prediction and probabilities
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Get confidence score
    confidence = max(probabilities) * 100
    
    # Get all emotion probabilities
    emotion_probs = dict(zip(model.classes_, probabilities))
    
    return {
        "emotion": prediction,
        "confidence": round(confidence, 2),
        "all_probabilities": {k: round(v * 100, 2) for k, v in emotion_probs.items()}
    }


# Emotion emoji mapping
EMOTION_EMOJIS = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😲',
    'neutral': '😐'
}


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get text from form or JSON
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
        
        if not text.strip():
            return jsonify({"error": "Please enter some text to analyze."})
        
        # Get prediction
        result = predict_emotion(text)
        
        if "error" in result:
            return jsonify(result)
        
        # Add emoji to result
        result['emoji'] = EMOTION_EMOJIS.get(result['emotion'], '🤔')
        result['original_text'] = text
        
        # If form submission, render template with result
        if not request.is_json:
            return render_template('index.html', result=result)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction (JSON only)."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Please provide text in JSON format: {'text': 'your text'}"}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({"error": "Text cannot be empty."}), 400
        
        result = predict_emotion(text)
        result['emoji'] = EMOTION_EMOJIS.get(result.get('emotion', ''), '🤔')
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    print("\n" + "=" * 50)
    print("EMOTION DETECTION WEB APPLICATION")
    print("=" * 50)
    print("Starting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
