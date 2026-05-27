import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification

# 1. 저장된 모델과 토크나이저 불러오기
model_path = "../models/mobilebert_finance_model"
tokenizer = MobileBertTokenizer.from_pretrained(model_path)
model = MobileBertForSequenceClassification.from_pretrained(model_path)


def predict_sentiment(text):
    # 텍스트를 AI가 이해할 수 있는 숫자로 변환
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)

    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)

    # 결과 해석 (가장 높은 확률의 클래스 선택)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()

    mapping = {0: "부정(Negative) 📉", 1: "긍정(Positive) 📈", 2: "중립(Neutral) 😐"}
    return mapping[prediction]


# 2. 테스트 시작
print("🔍 AI 감성 분석 테스트 (종료하려면 'exit' 입력)")
while True:
    user_input = input("뉴스 제목을 입력하세요: ")
    if user_input.lower() == 'exit':
        break

    result = predict_sentiment(user_input)
    print(f"🤖 AI의 판단: {result}\n")