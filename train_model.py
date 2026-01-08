"""
Bengali Hate Speech Detection Model Training Script
This script trains a transformer-based model for Bengali hate speech classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
import os
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
CONFIG = {
    'model_name': 'sagorsarker/bangla-bert-base',  # Bengali BERT model
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'output_dir': './bangla_hate_speech_model',
    'data_dir': 'Bengali_hate_speech_dataset/Bengali_hate_speech_dataset'
}

class BengaliHateSpeechDataset(Dataset):
    """Custom dataset class for Bengali hate speech data"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """Load train, validation, and test datasets"""
    print("Loading datasets...")
    
    train_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train.csv'))
    val_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'validate.csv'))
    test_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'test.csv'))
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Get unique labels
    all_labels = sorted(train_df['label'].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"\nLabels: {all_labels}")
    print(f"Label mapping: {label2id}")
    
    # Convert labels to numeric
    train_df['label_id'] = train_df['label'].map(label2id)
    val_df['label_id'] = val_df['label'].map(label2id)
    test_df['label_id'] = test_df['label'].map(label2id)
    
    # Check for missing values
    train_df = train_df.dropna(subset=['text', 'label'])
    val_df = val_df.dropna(subset=['text', 'label'])
    test_df = test_df.dropna(subset=['text', 'label'])
    
    return train_df, val_df, test_df, label2id, id2label

def prepare_datasets(train_df, val_df, test_df, tokenizer):
    """Prepare datasets for training"""
    print("\nPreparing datasets...")
    
    train_dataset = BengaliHateSpeechDataset(
        train_df['text'].tolist(),
        train_df['label_id'].tolist(),
        tokenizer,
        CONFIG['max_length']
    )
    
    val_dataset = BengaliHateSpeechDataset(
        val_df['text'].tolist(),
        val_df['label_id'].tolist(),
        tokenizer,
        CONFIG['max_length']
    )
    
    test_dataset = BengaliHateSpeechDataset(
        test_df['text'].tolist(),
        test_df['label_id'].tolist(),
        tokenizer,
        CONFIG['max_length']
    )
    
    return train_dataset, val_dataset, test_dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def train_model():
    """Main training function"""
    print("=" * 60)
    print("Bengali Hate Speech Detection - Model Training")
    print("=" * 60)
    
    # Load data
    train_df, val_df, test_df, label2id, id2label = load_data()
    num_labels = len(label2id)
    
    # Load tokenizer and model
    print(f"\nLoading model: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_df, val_df, test_df, tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=0.01,
        logging_dir=f"{CONFIG['output_dir']}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    
    # Save the final model
    print(f"\nSaving model to {CONFIG['output_dir']}...")
    trainer.save_model()
    tokenizer.save_pretrained(CONFIG['output_dir'])
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest Accuracy: {test_results['eval_accuracy']:.4f}")
    
    # Detailed evaluation
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_df['label_id'].tolist()
    
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=[id2label[i] for i in range(num_labels)]
    ))
    
    print("\n" + "=" * 60)
    print("Confusion Matrix")
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {CONFIG['output_dir']}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()

