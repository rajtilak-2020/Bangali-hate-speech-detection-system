# Bengali Hate Speech Detection

A machine learning project for detecting and classifying hate speech in Bengali text using transformer models.

## Features

- **Model Training**: Train a Bengali BERT-based model on hate speech dataset
- **Web Interface**: Simple HTML-based web interface for real-time predictions
- **Classification**: Identifies 4 types of speech:
  - Religious
  - Geopolitical
  - Neutral
  - Personal

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

First, train the model using your dataset:

```bash
python train_model.py
```

This will:
- Load the training, validation, and test datasets
- Train a Bengali BERT model
- Save the trained model to `./bangla_hate_speech_model/`
- Display evaluation metrics

### Step 2: Run the Web Interface

After training, start the Flask web server:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Step 3: Use the Interface

1. Enter Bengali text in the input box
2. Click "Analyze Text" or press `Ctrl+Enter`
3. View the predicted label and confidence scores

## Project Structure

```
.
├── train_model.py          # Model training script
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface
├── Bengali_hate_speech_dataset/
│   └── Bengali_hate_speech_dataset/
│       ├── train.csv      # Training data
│       ├── validate.csv   # Validation data
│       └── test.csv       # Test data
└── bangla_hate_speech_model/  # Saved model (created after training)
```

## Model Details

- **Base Model**: `sagorsarker/bangla-bert-base`
- **Max Length**: 128 tokens
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 5 (with early stopping)

## Notes

- Make sure to train the model before running the web interface
- The model will be saved automatically after training
- GPU is recommended for faster training but not required

