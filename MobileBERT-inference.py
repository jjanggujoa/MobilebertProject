import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

GPU = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_path = "remaining_reviews.csv"
df = pd.read_csv(data_path)

df['created_date'] = pd.to_datetime(df['created_date'])
df['year'] = df['created_date'].dt.year

data_X = list(df['review'].values)
labels = df['voted_up'].values
print("리뷰 개수:", len(data_X))

tokenizer = MobileBertTokenizer.from_pretrained("mobilebert-uncased", do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("토큰화 완료")

batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
print("데이터로더 구성 완료")

model = MobileBertForSequenceClassification.from_pretrained("mobilebert-uncased", num_labels=2)
model.load_state_dict(torch.load("mobilebert_custom_model.pt", map_location=device))
model.to(device)
model.eval()

test_pred = []
test_true = []

for batch in tqdm(test_loader, desc="Inferencing Full Dataset"):
    batch_ids, batch_mask, batch_labels = [b.to(device) for b in batch]
    with torch.no_grad():
        output = model(batch_ids, attention_mask=batch_mask)
    logits = output.logits
    pred = torch.argmax(logits, dim=1)
    test_pred.extend(pred.cpu().numpy())
    test_true.extend(batch_labels.cpu().numpy())

test_accuracy = np.mean(np.array(test_pred) == np.array(test_true))
print("전체 데이터에 대한 정확도:", test_accuracy)

df['predicted_sentiment'] = test_pred  # 0 = 부정, 1 = 긍정

sentiment_yearly = df.groupby(['year', 'predicted_sentiment']).size().unstack().fillna(0)
sentiment_yearly.columns = ['Negative', 'Positive']

# 백분율 계산
percent_df = sentiment_yearly.div(sentiment_yearly.sum(axis=1), axis=0) * 100

# 시각화
ax = sentiment_yearly.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])

# 백분율 텍스트 추가
for i, (index, row) in enumerate(sentiment_yearly.iterrows()):
    neg = row['Negative']
    pos = row['Positive']
    total = neg + pos
    if total == 0:
        continue
    neg_pct = percent_df.loc[index, 'Negative']
    pos_pct = percent_df.loc[index, 'Positive']

    ax.text(i, neg / 2, f"{neg_pct:.1f}%", ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    ax.text(i, neg + pos / 2, f"{pos_pct:.1f}%", ha='center', va='center', color='white', fontsize=9, fontweight='bold')

plt.title('Yearly Sentiment of PUBG Reviews (Predicted)')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()



