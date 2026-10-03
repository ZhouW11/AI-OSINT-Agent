from pydantic import BaseModel, Field, ConfigDict


class NewsSource(BaseModel):
    id: str | None = None
    name: str
    url: str
    country: str | None = None


class NewsArticle(BaseModel):

    model_config = ConfigDict(
        populate_by_name=True
    )

    source_id: str

    id: str

    title: str
    description: str
    content: str

    url: str

    image: str | None = None

    published_at: str = Field(alias="publishedAt")
    language: str = Field(alias="lang")

    source: NewsSource