import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

gpu = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

logging.set_verbosity_error()

df = pd.read_csv("sampled_reviews_by_year.csv")

df['created_date'] = pd.to_datetime(df['created_date'])
df['year'] = df['created_date'].dt.year

data_X = list(df['review'].values)
labels = df['voted_up'].values

tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length")

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

train, val, train_y, val_y = train_test_split(input_ids, labels, test_size=0.2, random_state=2025)
train_mask, val_mask, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2025)

batch_size = 8
train_data = TensorDataset(
    torch.tensor(train), torch.tensor(train_mask), torch.tensor(train_y, dtype=torch.long)
)
train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

val_data = TensorDataset(
    torch.tensor(val), torch.tensor(val_mask), torch.tensor(val_y, dtype=torch.long)
)
val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)


model = MobileBertForSequenceClassification.from_pretrained('mobilebert-uncased', num_labels=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_loader) * epochs)

epoch_results = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        b_input_ids, b_mask, b_labels = [x.to(device) for x in batch]

        model.zero_grad()
        output = model(b_input_ids, attention_mask=b_mask, labels=b_labels)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # 평가 (학습 정확도)
    model.eval()
    train_preds, train_true = [], []
    for batch in train_loader:
        b_input_ids, b_mask, b_labels = [x.to(device) for x in batch]
        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_mask).logits
        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_true.extend(b_labels.cpu().numpy())

    train_acc = np.mean(np.array(train_preds) == np.array(train_true))

    # 검증 정확도
    val_preds, val_true = [], []
    for batch in val_loader:
        b_input_ids, b_mask, b_labels = [x.to(device) for x in batch]
        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_mask).logits
        preds = torch.argmax(logits, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_true.extend(b_labels.cpu().numpy())

    val_acc = np.mean(np.array(val_preds) == np.array(val_true))

    epoch_results.append((avg_loss, train_acc, val_acc))

for idx, (loss, train_acc, val_acc) in enumerate(epoch_results, start=1):
    print(f"Epoch {idx} - Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

model_path = "mobilebert_custom_model.pt"
model.save_pretrained(model_path + '.pt')
print("모델 저장 완료:", model_path)



