import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. 데이터 로드 및 전처리
df = pd.read_csv('../data/AI_TOTAL_DATASET.csv')
# 모델이 인식할 수 있도록 'Label' 컬럼명을 'label'로 변경
df = df[['Sentence', 'Label']].rename(columns={'Label': 'label'}).dropna()

# 2. 학습/테스트 데이터 분리 (8:2)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. 토크나이저 및 모델 준비
model_name = "google/mobilebert-uncased"
tokenizer = MobileBertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # truncation과 padding을 통해 모든 문장 길이를 통일
    return tokenizer(examples["Sentence"], padding="max_length", truncation=True, max_length=128)

# 4. 데이터셋 변환 (Rename column을 통해 정답지를 확실히 명시)
train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

# 5. 모델 로드 (0:거품, 1:성장, 2:중립)
model = MobileBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 6. 성능 평가 함수
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# 7. 학습 설정
training_args = TrainingArguments(
    output_dir="../results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    # logging_dir 경고 해결을 위해 별도 설정 없이 자동 관리하게 함
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# 8. 트레이너 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("🚀 라벨 동기화 완료! MobileBERT 학습을 다시 시작합니다.")
trainer.train()

# 9. 모델 최종 저장
model.save_pretrained("./my_ai_bubble_model")
tokenizer.save_pretrained("./my_ai_bubble_model")
print("✅ 학습 완료! './my_ai_bubble_model' 폴더를 확인하세요.")