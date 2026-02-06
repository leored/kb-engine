"""Tests for Git repository scanner."""

import subprocess
from pathlib import Path

import pytest

from kb_engine.core.models.repository import RepositoryConfig
from kb_engine.git.scanner import GitRepoScanner


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary Git repository with some files."""
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    # Init repo
    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_path, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_path, capture_output=True, check=True,
    )

    # Create markdown files
    (repo_path / "docs").mkdir()
    (repo_path / "docs" / "entity.md").write_text("# Entity\n\nContent here.\n")
    (repo_path / "docs" / "process.md").write_text("# Process\n\nSteps here.\n")
    (repo_path / "README.md").write_text("# Test Repo\n\nA test repository.\n")
    (repo_path / "src").mkdir()
    (repo_path / "src" / "main.py").write_text("print('hello')\n")

    # Commit
    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path, capture_output=True, check=True,
    )

    return repo_path


@pytest.mark.unit
class TestGitRepoScanner:
    """Tests for GitRepoScanner."""

    def test_is_git_repo(self, git_repo: Path) -> None:
        config = RepositoryConfig(name="test", local_path=str(git_repo))
        scanner = GitRepoScanner(config)
        assert scanner.is_git_repo() is True

    def test_is_not_git_repo(self, tmp_path: Path) -> None:
        config = RepositoryConfig(name="test", local_path=str(tmp_path))
        scanner = GitRepoScanner(config)
        assert scanner.is_git_repo() is False

    def test_get_current_commit(self, git_repo: Path) -> None:
        config = RepositoryConfig(name="test", local_path=str(git_repo))
        scanner = GitRepoScanner(config)
        commit = scanner.get_current_commit()
        assert len(commit) == 40  # Full SHA

    def test_scan_files_default_pattern(self, git_repo: Path) -> None:
        config = RepositoryConfig(name="test", local_path=str(git_repo))
        scanner = GitRepoScanner(config)
        files = scanner.scan_files()
        assert "README.md" in files
        assert "docs/entity.md" in files
        assert "docs/process.md" in files
        assert "src/main.py" not in files  # Not .md

    def test_scan_files_custom_pattern(self, git_repo: Path) -> None:
        config = RepositoryConfig(
            name="test",
            local_path=str(git_repo),
            include_patterns=["**/*.py"],
        )
        scanner = GitRepoScanner(config)
        files = scanner.scan_files()
        assert "src/main.py" in files
        assert "README.md" not in files

    def test_scan_files_with_exclude(self, git_repo: Path) -> None:
        config = RepositoryConfig(
            name="test",
            local_path=str(git_repo),
            include_patterns=["**/*.md"],
            exclude_patterns=["README.md"],
        )
        scanner = GitRepoScanner(config)
        files = scanner.scan_files()
        assert "README.md" not in files
        assert "docs/entity.md" in files

    def test_get_changed_files(self, git_repo: Path) -> None:
        config = RepositoryConfig(name="test", local_path=str(git_repo))
        scanner = GitRepoScanner(config)

        initial_commit = scanner.get_current_commit()

        # Modify a file and create a new commit
        (git_repo / "docs" / "entity.md").write_text("# Entity\n\nUpdated content.\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Update entity"],
            cwd=git_repo, capture_output=True, check=True,
        )

        changed = scanner.get_changed_files(initial_commit)
        assert "docs/entity.md" in changed
        assert "docs/process.md" not in changed

    def test_read_file(self, git_repo: Path) -> None:
        config = RepositoryConfig(name="test", local_path=str(git_repo))
        scanner = GitRepoScanner(config)
        content = scanner.read_file("docs/entity.md")
        assert "# Entity" in content

    def test_get_current_branch(self, git_repo: Path) -> None:
        config = RepositoryConfig(name="test", local_path=str(git_repo))
        scanner = GitRepoScanner(config)
        branch = scanner.get_current_branch()
        # Depending on git version, may be "main" or "master"
        assert isinstance(branch, str)
        assert len(branch) > 0
