from pydantic import BaseModel, ConfigDict
from typing import Literal


class Evidence(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    source_id: str

    source_type: Literal["github", "news"]

    title: str

    url: str

    published_at: str | None = None