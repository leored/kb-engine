"""Git repository scanner using subprocess."""

import subprocess
from pathlib import Path

import structlog

from kb_engine.core.models.repository import RepositoryConfig

logger = structlog.get_logger(__name__)


class GitRepoScanner:
    """Scans a Git repository for indexable files.

    Uses subprocess + git CLI directly (no gitpython dependency).
    """

    def __init__(self, config: RepositoryConfig) -> None:
        self._config = config
        self._repo_path = Path(config.local_path).resolve()

    @property
    def repo_path(self) -> Path:
        return self._repo_path

    def _run_git(self, *args: str) -> str:
        """Run a git command and return stdout."""
        result = subprocess.run(
            ["git", *args],
            cwd=self._repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def is_git_repo(self) -> bool:
        """Check if the path is a valid git repository."""
        try:
            self._run_git("rev-parse", "--git-dir")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_current_commit(self) -> str:
        """Get the current HEAD commit hash."""
        return self._run_git("rev-parse", "HEAD")

    def get_remote_url(self) -> str | None:
        """Get the remote origin URL, if available."""
        try:
            url = self._run_git("remote", "get-url", "origin")
            return url if url else None
        except subprocess.CalledProcessError:
            return None

    def get_current_branch(self) -> str:
        """Get the current branch name."""
        try:
            return self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        except subprocess.CalledProcessError:
            return "main"

    def scan_files(self) -> list[str]:
        """Scan the repo for files matching include/exclude patterns.

        Returns relative paths from the repo root.
        """
        # Use git ls-files to get tracked files
        try:
            output = self._run_git("ls-files")
        except subprocess.CalledProcessError:
            logger.warning("git ls-files failed, falling back to filesystem scan")
            return self._scan_filesystem()

        all_files = output.splitlines() if output else []
        return self._filter_files(all_files)

    def _scan_filesystem(self) -> list[str]:
        """Fallback: scan filesystem directly."""
        all_files = []
        for path in self._repo_path.rglob("*"):
            if path.is_file():
                all_files.append(str(path.relative_to(self._repo_path)))
        return self._filter_files(all_files)

    def _filter_files(self, files: list[str]) -> list[str]:
        """Filter files by include/exclude patterns."""
        result = []
        for filepath in files:
            # Check include patterns
            included = any(
                self._match_pattern(filepath, pattern)
                for pattern in self._config.include_patterns
            )
            if not included:
                continue

            # Check exclude patterns
            excluded = any(
                self._match_pattern(filepath, pattern)
                for pattern in self._config.exclude_patterns
            )
            if excluded:
                continue

            result.append(filepath)

        return sorted(result)

    @staticmethod
    def _match_pattern(filepath: str, pattern: str) -> bool:
        """Match a filepath against a glob pattern.

        Handles ** patterns correctly for both root and nested files.
        For example, "**/*.md" matches both "README.md" and "docs/entity.md".
        """
        path = Path(filepath)
        # PurePath.match doesn't match root files against **/*.ext patterns
        # So we also check with the plain extension pattern
        if path.match(pattern):
            return True
        # If pattern starts with **/, also try without the **/ prefix
        if pattern.startswith("**/"):
            return path.match(pattern[3:])
        return False

    def get_changed_files(self, since_commit: str) -> list[str]:
        """Get files changed since a given commit.

        Returns relative paths of changed files that match patterns.
        """
        try:
            output = self._run_git("diff", "--name-only", since_commit, "HEAD")
            changed = output.splitlines() if output else []
        except subprocess.CalledProcessError:
            logger.warning(
                "git diff failed, returning all files",
                since_commit=since_commit,
            )
            return self.scan_files()

        return self._filter_files(changed)

    def get_deleted_files(self, since_commit: str) -> list[str]:
        """Get files deleted since a given commit."""
        try:
            output = self._run_git(
                "diff", "--name-only", "--diff-filter=D", since_commit, "HEAD"
            )
            deleted = output.splitlines() if output else []
        except subprocess.CalledProcessError:
            return []

        return self._filter_files(deleted)

    def read_file(self, relative_path: str) -> str:
        """Read a file from the repository."""
        file_path = self._repo_path / relative_path
        return file_path.read_text(encoding="utf-8")
