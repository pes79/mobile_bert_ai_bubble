import pandas as pd
import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os

# 1. 데이터 로드
df = pd.read_csv('../data/finance_data.csv')
sentences = df['Sentence'].tolist()
labels = df['Label'].tolist()

# 2. 훈련/테스트 데이터 분리
train_texts, val_texts, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# 3. 토크나이저 준비
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_texts, train_labels)
val_dataset = NewsDataset(val_texts, val_labels)

# 4. 모델 불러오기
model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=3)

# 5. 학습 설정 (에러 수정 부분)
training_args = TrainingArguments(
    output_dir='../results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    logging_dir='./logs',
    eval_strategy="epoch",  # evaluation_strategy -> eval_strategy로 수정
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 6. 트레이너 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 7. 학습 시작
print("🚀 AI가 공부를 시작합니다. 잠시만 기다려주세요...")
trainer.train()

# 8. 모델 저장
model.save_pretrained("./mobilebert_finance_model")
tokenizer.save_pretrained("./mobilebert_finance_model")
print("✅ 학습 완료! 이제 똑똑해진 AI 모델이 './mobilebert_finance_model' 폴더에 저장되었습니다.")