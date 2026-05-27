# 📉 MobileBERT를 활용한 AI 산업 거품론(AI Bubble) 여론 분석 프로젝트

---

![AI Bubble Visualization](https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&q=80&w=1000)

---

<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" /> <img src="https://img.shields.io/badge/MobileBERT-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> <img src="https://img.shields.io/badge/Gemini_Pro-8E75FB?style=for-the-badge&logo=google-gemini&logoColor=white" />

---

## ## 1. 문제 인식 (Problem Identification)
**인공지능(AI)은 현대 자본시장에서 가장 영향력 있는 테마이며, 특히 2023년 ChatGPT 등장 이후 주식 시장의 모든 자금이 AI로 쏠리고 있습니다.** 하지만 최근 빅테크 기업들의 막대한 투자 규모 대비 실질적인 수익화가 더디다는 지적이 나오면서 **"AI는 제2의 닷컴버블인가?"**라는 의구심이 증폭되고 있습니다. 

본 프로젝트에서는 단순한 주가 지표를 넘어, 뉴스 문맥 속에 숨겨진 **시장의 심리적 임계점**을 파악하고자 합니다. 투자자들의 기술적 낙관론이 언제 불안감으로 변하는지 그 실체를 확인하기 위해 MobileBERT를 활용한 정교한 여론 분석 모델을 구축하였습니다.

## ## 2. 문제의 심각성 (Severity & Impact)
**기술적 거품론은 단순히 시장의 우려를 넘어 투자 의사결정에 심각한 혼선을 야기할 수 있습니다.**
- **데이터 과부하:** 매일 쏟아지는 수만 건의 AI 관련 정보 중 실제 리스크 신호를 수작업으로 선별하는 것은 불가능합니다.
- **디지털 피로감:** 긍정과 부정의 정보가 뒤섞인 환경에서 투자자는 객관성을 잃고 '포모(FOMO)'나 과도한 공포에 노출되기 쉽습니다.
- **보안 및 사기 위협:** 거품론을 틈탄 가짜 뉴스나 피싱성 금융 정보는 자본시장의 신뢰도를 저해하는 심각한 요인이 됩니다.

## ## 3. 데이터 (Data)
### ### 2.1 데이터 수집
- **수집 방법:** 직접 수집
    - **어디서 얼마나:** 구글 뉴스(GNews) 및 주요 경제 플랫폼에서 약 **50,000건** 원천 데이터 확보
    - **데이터 항목:** 날짜, 뉴스 제목, 관련 핵심 키워드
    - **기간:** 2023년 1분기 ~ 2026년 4월 (약 3년 4개월)
    - **수집 방식:** GNews API 및 Pandas를 활용한 시계열 자동화 크롤링 파이프라인 구축

### ### 2.2 학습 데이터 구축 (Labeling)
- **정제 규모:** 중복 제거 및 필터링을 거친 **최종 2,374건**의 정밀 학습 데이터셋
- **라벨링 체계:** 0(거품 위기), 1(시장 성장), 2(중립/관망)
- **도구:** **Gemini 1.5 Flash API**를 활용한 지능형 오토 라벨링 및 수동 검수(Human-in-the-loop) 병행

## ## 4. MobileBERT 모델 학습 (Fine-tuning)
- **설정:** Train(학습)과 Validation(검증) 데이터를 **8:2** 비율로 설정
- **모델:** `google/mobilebert-uncased` (경량화된 고성능 언어 모델 사용)
- **성능 지표:** - **검증 정확도(Accuracy): 80.1%** 달성
    - **F1-Score: 0.80** 기록
- **분석 결과:** 데이터 양이 증가함에 따라 정확도가 꾸준히 향상되었으며, 특히 약 3,000건의 데이터만으로도 안정적인 리스크 탐지 성능을 확보함.

## ## 5. 문제 해결 결과 (Topic Modeling & Inference)
**모델을 활용한 실제 뉴스 판독 결과 및 토픽 모델링을 통해 거품론의 실체를 분석하였습니다.**

| 토픽 번호 | 핵심 키워드 | 분석 해석 |
| :--- | :--- | :--- |
| **Topic 1** | bubble, crash, dot-com | 과거 경제 위기(닷컴 버블)와의 비교를 통한 역사적 공포감 형성 |
| **Topic 2** | spending, ROI, revenue | 막대한 인프라 투자 대비 불확실한 수익화 구조에 대한 의문 |
| **Topic 3** | overvalued, warning, sell | 기술 가치 고평가에 따른 전문가들의 경고 및 매도 압력 증대 |

- **실시간 추론 사례:** - *"AI stocks face massive sell-off as bubble fears mount"* → **⚠️ 거품 위기 (위험 확률 96.9%)**
    - *"Nvidia's revenue soars as AI demand remains insatiable"* → **📈 시장 성장 (성장 확률 83.8%)**

## ## 6. 느낀점 및 개선방향
텍스트 데이터를 통해 시장의 **'심리적 거품'**을 정량화하고 수치로 변환(96.9% 등)해보는 귀중한 경험이었습니다. MobileBERT가 적은 데이터셋으로도 매우 날카로운 리스크 감지 능력을 보여준 점이 인상적이었으며, 향후 다국어 모델 도입을 통해 글로벌 시장과 국내 시장의 반응 차이를 정밀하게 비교 분석해보고 싶습니다.