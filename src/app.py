from flask import Flask, render_template, request, jsonify
import joblib
import json
import os
import re

app = Flask(__name__)

def preprocess_text(text):
    """Clean text for prediction"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def load_model():
    """Load trained model"""
    try:
        model = joblib.load('models/spam_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        if not model or not vectorizer:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        # Preprocess
        processed_text = preprocess_text(message)
        
        # Vectorize
        vec = vectorizer.transform([processed_text])
        
        # Predict
        prob = model.predict_proba(vec)[0]
        pred = model.predict(vec)[0]
        
        # Your model uses: spam=1, ham=0
        result = {
            'spam_prob': float(prob[1]),
            'ham_prob': float(prob[0]),
            'is_spam': bool(pred == 1),
            'prediction': 'SPAM' if pred == 1 else 'NOT SPAM'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def get_metrics():
    try:
        with open('models/model_stats.json') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except:
        return jsonify({
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_samples': 0,
            'spam_samples': 0,
            'ham_samples': 0
        })

@app.route('/train-status')
def train_status():
    model_exists = os.path.exists('models/spam_model.pkl')
    data_exists = os.path.exists('data/spam.csv')
    
    return jsonify({
        'model_trained': model_exists,
        'data_available': data_exists
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)