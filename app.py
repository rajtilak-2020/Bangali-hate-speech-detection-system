"""
Flask web application for Bengali Hate Speech Detection
"""

from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
MODEL_PATH = './bangla_hate_speech_model'
MAX_LENGTH = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for model and tokenizer
model = None
tokenizer = None
id2label = None

def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer, id2label
    
    if not os.path.exists(MODEL_PATH):
        return False
    
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    # Get label mapping from model config
    if hasattr(model.config, 'id2label'):
        id2label = model.config.id2label
    else:
        # Default labels if not in config
        id2label = {0: 'Religious', 1: 'Geopolitical', 2: 'Neutral', 3: 'Personal'}
    
    print("Model loaded successfully!")
    return True

def predict(text):
    """Predict the label for given text"""
    if model is None or tokenizer is None:
        return None, None
    
    # Tokenize input
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_id].item()
    
    predicted_label = id2label.get(predicted_id, 'Unknown')
    
    # Get all probabilities
    all_probs = {}
    for idx, prob in enumerate(probabilities[0]):
        label = id2label.get(idx, f'Label_{idx}')
        all_probs[label] = prob.item()
    
    return predicted_label, confidence, all_probs

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'Please provide some text to analyze.'
            }), 400
        
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first using train_model.py'
            }), 500
        
        predicted_label, confidence, all_probs = predict(text)
        
        if predicted_label is None:
            return jsonify({
                'error': 'Prediction failed. Please try again.'
            }), 500
        
        return jsonify({
            'success': True,
            'text': text,
            'predicted_label': predicted_label,
            'confidence': round(confidence * 100, 2),
            'all_probabilities': {k: round(v * 100, 2) for k, v in all_probs.items()}
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Bengali Hate Speech Detection - Web Interface")
    print("=" * 60)
    
    if load_model():
        print("\nStarting Flask server...")
        print("Open your browser and go to: http://127.0.0.1:5000")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first by running: python train_model.py")
        print("=" * 60)

