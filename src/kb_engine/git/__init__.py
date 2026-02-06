"""Git integration module for KB-Engine."""

from kb_engine.git.scanner import GitRepoScanner
from kb_engine.git.url_resolver import URLResolver

__all__ = ["GitRepoScanner", "URLResolver"]
