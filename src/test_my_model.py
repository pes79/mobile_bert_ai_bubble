import torch
import torch.nn.functional as F
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification

model_path = "../models/my_ai_bubble_model"
tokenizer = MobileBertTokenizer.from_pretrained(model_path)
model = MobileBertForSequenceClassification.from_pretrained(model_path)


def predict_bubble_with_prob(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        # 확률값으로 변환 (Softmax)
        probs = F.softmax(outputs.logits, dim=-1)

    # 각 라벨별 확률 추출
    bubble_p = probs[0][0].item() * 100
    growth_p = probs[0][1].item() * 100
    neutral_p = probs[0][2].item() * 100

    prediction = torch.argmax(probs, dim=-1).item()
    labels = {0: "⚠️ 거품 위기", 1: "📈 시장 성장", 2: "😐 중립"}

    print(f"뉴스: {text}")
    print(f"결과: {labels[prediction]} (위험: {bubble_p:.1f}% | 성장: {growth_p:.1f}% | 중립: {neutral_p:.1f}%)")
    print("-" * 50)


test_news = [
    "Nvidia's revenue soars as AI demand remains insatiable",
    "Concerns grow over massive AI infrastructure spending without clear returns",
    "AI stocks face massive sell-off as bubble fears mount",  # 더 강한 문장 테스트
    "Is the AI revolution just a repeat of the dot-com crash?"  # 역사적 비유 테스트
]

for news in test_news:
    predict_bubble_with_prob(news)