Copyright 2025 Your Yash Dasari

# File: analyze_errors.py

from unittest import result
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os
import shutil
import subprocess

class ErrorDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if "ERROR" in line or "WARNING" in line:
                    self.data.append(line.strip())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

def analyze_errors(file_path):
    # Load the trained model and tokenizer
    model = BertForSequenceClassification.from_pretrained("./trained_model_updated")
    tokenizer = BertTokenizer.from_pretrained("./trained_model_updated")
    
    # Load error types
    with open("./error_types.txt", "r") as f:
        error_types = [line.strip() for line in f]
    
    dataset = ErrorDataset(file_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            for error, pred in zip(input_ids, predictions):
                error_text = tokenizer.decode(error, skip_special_tokens=True)
                error_type = error_types[pred]
                results.append((error_text, error_type))
    
    return results

def extract_error_details(error_text):
    # Extract timestamp, severity, component, and exception type
    match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (ERROR|WARNING) \[(.*?)\] - (\w+):', error_text)
    if match:
        timestamp, severity, component, exception = match.groups()
        return {
            'timestamp': timestamp,
            'severity': severity,
            'component': component,
            'exception': exception
        }
    return None

# Main function to analyze errors in a log file
def main():
    # Specify the path to the any Build output log file to analyze errors
    file_path = "./logs/test3.txt"
    analyzed_errors = analyze_errors(file_path)
    
    for error_text, predicted_error_type in analyzed_errors:
        print(f"Log Entry: {error_text}")
        print(f"Predicted Error Type: {predicted_error_type}")
        
        details = extract_error_details(error_text)
        if details:
            print(f"Timestamp: {details['timestamp']}")
            print(f"Severity: {details['severity']}")
            print(f"Component: {details['component']}")
            print(f"Exception: {details['exception']}")
        
        print("---")
    
    # After analysis, copy the log file for retraining
    retrain_log_path = "./temp_log_for_retraining.txt"
    shutil.copy2(file_path, retrain_log_path)
    print(f"Log file copied to {retrain_log_path} for retraining.")

    # Automatically trigger the retraining process
    print("Starting automatic retraining process...")
    try:
        subprocess.run(["python", "train_error_classifier.py", "--mode", "retrain"], check=True)
        print("Retraining completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during retraining: {e}")

if __name__ == "__main__":
    main()
