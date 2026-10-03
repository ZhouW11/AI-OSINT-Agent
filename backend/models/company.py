from pydantic import BaseModel

from models.github import GitHubRepository
from models.news import NewsArticle


class CompanyIntelligence(BaseModel):
    github: list[GitHubRepository]
    news: list[NewsArticle]