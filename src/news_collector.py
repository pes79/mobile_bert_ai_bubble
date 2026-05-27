import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import os


def news_crawler():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # 분석 대상 확대
    tickers = {
        "NVIDIA": "NVDA", "Microsoft": "MSFT", "Meta": "META",
        "Google": "GOOGL", "Apple": "AAPL", "Tesla": "TSLA", "Amazon": "AMZN"
    }
    all_results = []

    for name, symbol in tickers.items():
        print(f"🚀 {name} 뉴스 수집 시작...")
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        driver.get(url)
        time.sleep(3)

        # 페이지를 아래로 5번 스크롤해서 뉴스 양을 늘립니다
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)

        titles = driver.find_elements(By.CSS_SELECTOR, "h3.clamp")
        for t in titles:
            if t.text and len(t.text) > 10:
                all_results.append({"Sentence": t.text, "Keyword": name, "Label": 2})

    driver.quit()

    df = pd.DataFrame(all_results).drop_duplicates()  # 중복 제거
    df.to_csv('collected_raw_data.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 대량 수집 완료! 총 {len(df)}건 확보.")


if __name__ == "__main__":
    news_crawler()