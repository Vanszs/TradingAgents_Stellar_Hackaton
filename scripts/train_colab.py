"""
Crypto News Classifier - Training Script for Google Colab

Upload this file to Colab and run all cells.
Requirements: Runtime -> Change runtime type -> T4 GPU
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
# !pip install torch transformers datasets accelerate

# ============================================================
# CELL 2: Upload Data
# ============================================================
# Upload labeled_articles.jsonl ke Colab
# from google.colab import files
# uploaded = files.upload()  # Pilih file labeled_articles.jsonl

# ============================================================
# CELL 3: Setup
# ============================================================
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# CELL 4: Load Data
# ============================================================
LABEL_MAP = {"BIASA": 0, "LUMAYAN": 1, "PENTING": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def load_data(path):
    articles = []
    with open(path, 'r') as f:
        for line in f:
            article = json.loads(line.strip())
            if article.get('label') in LABEL_MAP:
                text = f"TITLE: {article.get('title', '')} CONTENT: {article.get('description', '')}"
                articles.append({
                    'text': text[:512],
                    'label': LABEL_MAP[article['label']]
                })
    return articles

data = load_data('labeled_articles.jsonl')
print(f"Loaded {len(data)} articles")

# Check distribution
from collections import Counter
dist = Counter([d['label'] for d in data])
for label_id, count in sorted(dist.items()):
    print(f"  {ID_TO_LABEL[label_id]}: {count} ({100*count/len(data):.1f}%)")

# ============================================================
# CELL 5: Create Dataset
# ============================================================
class CryptoNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Split data
texts = [d['text'] for d in data]
labels = [d['label'] for d in data]
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
)

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# ============================================================
# CELL 6: Initialize Model
# ============================================================
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    problem_type="single_label_classification"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")

# Create datasets
train_dataset = CryptoNewsDataset(train_texts, train_labels, tokenizer)
val_dataset = CryptoNewsDataset(val_texts, val_labels, tokenizer)
test_dataset = CryptoNewsDataset(test_texts, test_labels, tokenizer)

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ============================================================
# CELL 7: Training Loop
# ============================================================
EPOCHS = 3
LEARNING_RATE = 2e-5

# Class weights
class_counts = [dist[i] for i in range(3)]
class_weights = torch.tensor([sum(class_counts) / (3 * c) for c in class_counts]).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

print(f"Training for {EPOCHS} epochs, {len(train_loader)} batches/epoch")
print(f"Class weights: {class_weights.tolist()}")

best_val_f1 = 0
for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # Validate
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = val_loss / len(val_loader)
    val_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)

    # F1
    from sklearn.metrics import f1_score
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"  -> New best model saved (F1: {val_f1:.4f})")

# ============================================================
# CELL 8: Evaluate on Test Set
# ============================================================
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

print("=== Test Results ===")
print(classification_report(all_labels, all_preds, target_names=['BIASA', 'LUMAYAN', 'PENTING']))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ============================================================
# CELL 9: Save Model
# ============================================================
# Download model ke lokal
from google.colab import files
files.download('best_model.pt')
print("Model downloaded! Copy to tradingagents/news_classifier/pretrained/best_model.pt")
