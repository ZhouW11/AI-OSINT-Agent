import os
import requests
from dotenv import load_dotenv

load_dotenv()

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")


def search_news(company):
    url = "https://gnews.io/api/v4/search"

    params = {
        "q": company,
        "lang": "en",
        "max": 5,
        "apikey": GNEWS_API_KEY
    }

    response = requests.get(url, params=params, timeout=10)

    response.raise_for_status()

    data = response.json()

    return data.get("articles", [])


if __name__ == "__main__":
    company = input("请输入你想查询的公司名称：").strip()

    if not company:
        print("公司名称不能为空")
    else:
        articles = search_news(company)

        print(f"\n正在显示 {company} 的最新新闻：\n")

        for article in articles:
            print("标题:", article["title"])
            print("来源:", article["source"]["name"])
            print("时间:", article["publishedAt"])
            print("链接:", article["url"])
            print("-" * 50)