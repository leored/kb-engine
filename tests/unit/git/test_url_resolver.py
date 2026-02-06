"""Tests for URL resolver."""

import pytest

from kb_engine.core.models.repository import RepositoryConfig
from kb_engine.git.url_resolver import URLResolver


@pytest.mark.unit
class TestURLResolver:
    """Tests for URLResolver."""

    def test_resolve_local(self) -> None:
        config = RepositoryConfig(
            name="test-repo",
            local_path="/tmp/test-repo",
        )
        resolver = URLResolver(config)
        url = resolver.resolve("docs/entity.md", "atributos")
        assert url.startswith("file://")
        assert url.endswith("/test-repo/docs/entity.md#atributos")

    def test_resolve_local_no_anchor(self) -> None:
        config = RepositoryConfig(
            name="test-repo",
            local_path="/tmp/test-repo",
        )
        resolver = URLResolver(config)
        url = resolver.resolve("docs/entity.md")
        assert url.startswith("file://")
        assert url.endswith("/test-repo/docs/entity.md")
        assert "#" not in url

    def test_resolve_remote_https(self) -> None:
        config = RepositoryConfig(
            name="test-repo",
            local_path="/tmp/test-repo",
            remote_url="https://github.com/org/repo.git",
            branch="main",
        )
        resolver = URLResolver(config)
        url = resolver.resolve("docs/entity.md", "atributos")
        assert url == "https://github.com/org/repo/blob/main/docs/entity.md#atributos"

    def test_resolve_remote_ssh(self) -> None:
        config = RepositoryConfig(
            name="test-repo",
            local_path="/tmp/test-repo",
            remote_url="git@github.com:org/repo.git",
            branch="develop",
        )
        resolver = URLResolver(config)
        url = resolver.resolve("README.md")
        assert url == "https://github.com/org/repo/blob/develop/README.md"

    def test_resolve_with_template(self) -> None:
        config = RepositoryConfig(
            name="test-repo",
            local_path="/tmp/test-repo",
            remote_url="https://github.com/org/repo",
            branch="main",
            base_url_template="{remote}/blob/{branch}/{path}",
        )
        resolver = URLResolver(config)
        url = resolver.resolve("docs/entity.md", "sec")
        assert url == "https://github.com/org/repo/blob/main/docs/entity.md#sec"

    def test_normalize_ssh_url(self) -> None:
        result = URLResolver._normalize_remote_url("git@github.com:org/repo.git")
        assert result == "https://github.com/org/repo"

    def test_normalize_https_url(self) -> None:
        result = URLResolver._normalize_remote_url("https://github.com/org/repo.git")
        assert result == "https://github.com/org/repo"

    def test_normalize_clean_url(self) -> None:
        result = URLResolver._normalize_remote_url("https://github.com/org/repo")
        assert result == "https://github.com/org/repo"
