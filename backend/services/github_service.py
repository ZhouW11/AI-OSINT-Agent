import os

import requests
from dotenv import load_dotenv

from models.github import GitHubRepository


load_dotenv()


def search_github(keyword: str) -> list[GitHubRepository]:

    url = "https://api.github.com/search/repositories"

    params = {
        "q": keyword,
        "sort": "stars",
        "order": "desc",
        "per_page": 5,
    }

    headers = {
        "Accept": "application/vnd.github+json",
    }

    github_token = os.getenv("GITHUB_TOKEN")

    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    repositories = []

    for item in data["items"]:
        repository = GitHubRepository(
            name=item["full_name"],
            description=item.get("description"),
            stars=item["stargazers_count"],
            forks=item["forks_count"],
            language=item.get("language"),
            updated=item["updated_at"],
            url=item["html_url"],
        )

        repositories.append(repository)

    return repositories