import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import re
import os
import shutil
from datetime import datetime
import argparse

class ErrorDataset(Dataset):
    def __init__(self, file_path, tokenizer, error_types_file, max_length=256):
        self.data = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        #Load Error Types
        with open(error_types_file, 'r') as f:
            self.error_types = [line.strip() for line in f]
        self.error_type_to_id = {error_type: i for i, error_type in enumerate(self.error_types)}

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if "ERROR" in line or "WARNING" in line:
                    error_type = self.extract_error_type(line)
                    if error_type in self.error_type_to_id:
                        self.data.append(line.strip())
                        self.labels.append(self.error_type_to_id[error_type])

    def extract_error_type(self, line):
        # Extract error type from the log line
        match = re.search(r'(ERROR|WARNING) \[.*?\] - (\w+):', line)
        if match:
            return f"{match.group(1)}_{match.group(2)}"
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_dataloader, val_dataloader, device, epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model

def retrain_model(model, tokenizer, new_log_file, error_types_file, device, epochs=3):
    new_dataset = ErrorDataset(new_log_file, tokenizer, error_types_file)
    
    dataloader = DataLoader(new_dataset, batch_size=16, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

def extract_error_type(line):
    match = re.search(r'(ERROR|WARNING) \[.*?\] - (\w+):', line)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None

def main():
    parser = argparse.ArgumentParser(description='Train or retrain error classifier')
    parser.add_argument('--mode', choices=['train', 'retrain'], default='train', help='Mode: train or retrain (default: %(default)s)')
    args = parser.parse_args()

    # Define path to your single log file
    log_file = "./logs/train.txt"
    error_types_file = "./error_types.txt"

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #Initial Training mode
    if args.mode == 'train':
        print("Starting initial training...")
        dataset = ErrorDataset(log_file, tokenizer, error_types_file)

        # Initialize model with the correct number of labels
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dataset.error_types))

        # Split dataset into train and validation
        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
        
        # Create data loaders
        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=16)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Train the model
        trained_model = train_model(model, train_dataloader, val_dataloader, device)

        # Create directory for saving the model if it doesn't exist
        os.makedirs("./trained_model", exist_ok=True)

        # Save the trained model
        trained_model.save_pretrained("./trained_model")
        tokenizer.save_pretrained("./trained_model")
    
        # Copy error_types.txt to trained_model directory
        shutil.copy2(error_types_file, "./trained_model/error_types_updated.txt")

        print(f"Model trained and saved. Number of error types: {len(dataset.error_types)}")

    # Retraining mode
    elif args.mode == 'retrain':

        # Check for new log file and retrain if it exists
        new_log_file = "./temp_log_for_retraining.txt"
        error_types_file = "./error_types.txt"

        if os.path.exists(new_log_file):
            print(f"Retraining model with new log file: {new_log_file}")

            model = BertForSequenceClassification.from_pretrained("./trained_model_updated")

            tokenizer = BertTokenizer.from_pretrained("./trained_model_updated")

            # Load existing error types
            with open(error_types_file, 'r') as f:
                existing_error_types = set(line.strip() for line in f)
            
            # Check for new error types in the new log file
            new_error_types = set()
            with open(new_log_file, 'r') as f:
                for line in f:
                    error_type = extract_error_type(line)  # Implement this function
                    if error_type:
                        new_error_types.add(error_type)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            retrained_model = retrain_model(model, tokenizer, new_log_file, error_types_file, device)

            # Delete the existing folder
            shutil.rmtree("./trained_model_updated", ignore_errors=True)

            # Create directory for saving the model if it doesn't exist
            os.makedirs("./trained_model_updated", exist_ok=True)

            #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #save_dir = f"./retrained_model_{timestamp}"
            #os.makedirs(save_dir, exist_ok=True)

            retrained_model.save_pretrained("./trained_model_updated")

            tokenizer.save_pretrained("./trained_model_updated")
            
            with open(f"{"./trained_model"}/error_types_updated.txt", "w") as f:
                for error_type in retrained_model.config.id2label.values():
                    f.write(f"{error_type}\n")
            
            print(f"Retrained model saved, overwriting the existing model in {"./trained_model_updated"}")
        else:
            print(f"No new log file found for retraining at {new_log_file}")

if __name__ == "__main__":
    main()