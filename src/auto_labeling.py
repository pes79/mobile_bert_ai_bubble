import google.generativeai as genai
import pandas as pd
import time
import os
from gnews import GNews

API_KEY = "api"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')


def get_label_final(sentence):
    sentence_lower = sentence.lower()
    pos = ['surge', 'gain', 'growth', 'beat', 'profit', 'breakthrough', 'success', 'up']
    neg = ['warning', 'bubble', 'risk', 'drop', 'crash', 'overvalued', 'fear', 'down']

    if any(w in sentence_lower for w in pos): return 1
    if any(w in sentence_lower for w in neg): return 0

    try:
        clean_text = "".join(i for i in sentence if ord(i) < 128)
        prompt = f"Analyze tone for AI market bubble/growth: '{clean_text}'. 1:Growth, 0:Bubble Risk, 2:Neutral. Answer only digit."
        response = model.generate_content(prompt)
        res = response.text.strip()
        digit = "".join(filter(str.isdigit, res))
        return int(digit[0]) if digit else 2
    except:
        return 2


def run_massive_project():
    # max_results를 100으로 늘려 더 많이 가져오도록 설정
    google_news = GNews(language='en', country='US', period='30d', max_results=100)

    # 확장된 키워드 리스트
    # auto_labeling.py의 keywords 리스트를 이걸로 교체!
    keywords = [
        "AI stock market bubble burst", "Nvidia overvalued warning", "AI ROI disappointment",
        "Generative AI hype vs reality", "Tech sector debt crisis AI", "AI investment fraud",
        "Artificial intelligence massive layoffs", "AI chip demand falling", "AI profit margin squeeze",
        "Why AI is a scam", "History of dotcom bubble vs AI", "AI infrastructure waste",
        "Slowing AI adoption in enterprise", "AI companies losing money", "Is AI a trillion dollar mistake"
    ]

    all_news = []
    print(f"📡 수집 시작 (키워드 확장 버전)...")

    for q in keywords:
        search_results = google_news.get_news(q)
        for entry in search_results:
            all_news.append({"Sentence": entry['title'], "Keyword": q})
        print(f"✔️ {q} 완료")

    df = pd.DataFrame(all_news).drop_duplicates(subset=['Sentence'])
    print(f"📊 중복 제거 후 총 {len(df)}건 확보")

    print(f"🚀 라벨링 시작...")
    labels = []
    for i, row in df.iterrows():
        label = get_label_final(row['Sentence'])
        labels.append(label)

        if (i + 1) % 20 == 0:
            print(f"⏳ [{i + 1}/{len(df)}] 진행 중 (결과: {label})")

        time.sleep(1.2)

    df['Label'] = labels
    # 기존 파일과 헷갈리지 않게 파일명을 PART2로 변경했습니다.
    output_name = 'AI_BUBBLE_DATA_PART2.csv'
    df.to_csv(output_name, index=False, encoding='utf-8-sig')

    print(f"✅ 완료: {output_name}")
    print(df['Label'].value_counts())


if __name__ == "__main__":
    run_massive_project()