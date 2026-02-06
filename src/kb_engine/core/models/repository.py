"""Repository configuration models."""

from pydantic import BaseModel, Field


class RepositoryConfig(BaseModel):
    """Configuration for a Git repository to index."""

    name: str
    local_path: str
    remote_url: str | None = None
    branch: str = "main"
    include_patterns: list[str] = Field(default_factory=lambda: ["**/*.md"])
    exclude_patterns: list[str] = Field(default_factory=list)
    base_url_template: str | None = None  # e.g. "{remote}/blob/{branch}/{path}"
