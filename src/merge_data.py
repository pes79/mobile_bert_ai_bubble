import pandas as pd

# 1. 모든 파일 합치기 (파일명 확인 필수!)
files = ['AI_BUBBLE_DATA_FINAL.csv', 'AI_BUBBLE_DATA_PART2.csv']
# 만약 파일명이 PART2로 겹쳐서 생성되었다면, 폴더 내 실제 파일명들을 모두 리스트에 넣으세요.

df_list = []
for f in files:
    try:
        df_list.append(pd.read_csv(f))
    except:
        print(f"파일 {f}를 찾을 수 없어 건너뜁니다.")

df_total = pd.concat(df_list, ignore_index=True)
df_total = df_total.drop_duplicates(subset=['Sentence'])

# 2. 최종 저장
df_total.to_csv('AI_TOTAL_DATASET.csv', index=False, encoding='utf-8-sig')

# 3. 최종 리포트 출력
print("="*30)
print(f"✅ 최종 통합 데이터셋 완성!")
print(f"📊 전체 데이터 수: {len(df_total)} 건")
print("-"*30)
print("[라벨별 분포]")
print(df_total['Label'].value_counts().sort_index())
print("="*30)