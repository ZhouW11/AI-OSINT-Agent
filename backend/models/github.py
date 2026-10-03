from pydantic import BaseModel
from typing import Optional


class GitHubRepository(BaseModel):

    name: str
    description: Optional[str] = None

    stars: int
    forks: int

    language: Optional[str] = None

    updated: str

    url: str