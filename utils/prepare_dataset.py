import json
import random
import os

def prepare_splits(input_path, train_path, val_path, split_ratio=0.9):
    print(f"Loading data from {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    print(f"Total samples: {len(data)}")
    
    # Shuffle data with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("Dataset splitting complete.")

if __name__ == "__main__":
    input_file = "data/processed/cleaned_dataset.jsonl"
    train_file = "data/splits/train.jsonl"
    val_file = "data/splits/val.jsonl"
    prepare_splits(input_file, train_file, val_file, split_ratio=0.9)
