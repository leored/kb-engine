"""URL resolver for document references."""

import re
from pathlib import Path

from kb_engine.core.models.repository import RepositoryConfig


class URLResolver:
    """Resolves (relative_path, anchor) to full URLs.

    Supports two modes:
    - Local: file:///absolute/path/to/doc.md#section
    - Remote: https://github.com/org/repo/blob/main/doc.md#section
    """

    def __init__(self, config: RepositoryConfig) -> None:
        self._config = config
        self._repo_path = Path(config.local_path).resolve()

    def resolve(self, relative_path: str, anchor: str | None = None) -> str:
        """Resolve a relative path and optional anchor to a full URL.

        If a remote_url or base_url_template is configured, produces
        a remote URL. Otherwise, produces a local file:// URL.
        """
        if self._config.base_url_template:
            return self._resolve_template(relative_path, anchor)
        elif self._config.remote_url:
            return self._resolve_remote(relative_path, anchor)
        else:
            return self._resolve_local(relative_path, anchor)

    def _resolve_local(self, relative_path: str, anchor: str | None) -> str:
        """Resolve to a local file:// URL."""
        absolute_path = self._repo_path / relative_path
        url = f"file://{absolute_path}"
        if anchor:
            url += f"#{anchor}"
        return url

    def _resolve_remote(self, relative_path: str, anchor: str | None) -> str:
        """Resolve to a remote URL based on the git remote."""
        remote = self._config.remote_url or ""
        base = self._normalize_remote_url(remote)
        branch = self._config.branch
        url = f"{base}/blob/{branch}/{relative_path}"
        if anchor:
            url += f"#{anchor}"
        return url

    def _resolve_template(self, relative_path: str, anchor: str | None) -> str:
        """Resolve using a custom URL template."""
        template = self._config.base_url_template or ""
        remote = self._normalize_remote_url(self._config.remote_url or "")
        url = template.replace("{remote}", remote)
        url = url.replace("{branch}", self._config.branch)
        url = url.replace("{path}", relative_path)
        if anchor:
            url += f"#{anchor}"
        return url

    @staticmethod
    def _normalize_remote_url(url: str) -> str:
        """Normalize a git remote URL to an HTTPS base URL.

        Handles:
        - git@github.com:org/repo.git -> https://github.com/org/repo
        - https://github.com/org/repo.git -> https://github.com/org/repo
        """
        # Strip .git suffix
        url = re.sub(r"\.git$", "", url)
        # Convert SSH to HTTPS
        ssh_match = re.match(r"git@([^:]+):(.+)", url)
        if ssh_match:
            host, path = ssh_match.groups()
            return f"https://{host}/{path}"
        return url
