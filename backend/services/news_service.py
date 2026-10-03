from news.client import search_news
from models.news import NewsArticle, NewsSource


def search_news_articles(keyword: str) -> list[NewsArticle]:

    raw_data = search_news(keyword)

    articles = []

    for item in raw_data:

        source = NewsSource(
            id=item["source"].get("id"),
            name=item["source"]["name"],
            url=item["source"]["url"],
            country=item["source"].get("country"),
        )

        source_id = f"news:{item['id']}"

        article = NewsArticle(
            source_id=source_id,

            id=item["id"],

            title=item["title"],
            description=item["description"],
            content=item["content"],

            url=item["url"],
            image=item.get("image"),

            publishedAt=item["publishedAt"],
            lang=item["lang"],

            source=source,
        )

        articles.append(article)

    return articles