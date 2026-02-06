"""Markdown parsing utilities."""

import re
import unicodedata
from typing import Any

import frontmatter


def extract_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter from markdown content.

    Returns a tuple of (metadata dict, content without frontmatter).
    """
    try:
        post = frontmatter.loads(content)
        return dict(post.metadata), post.content
    except Exception:
        return {}, content


def parse_markdown_sections(
    content: str,
) -> list[tuple[list[str], str]]:
    """Parse markdown content into sections with heading paths.

    Returns a list of (heading_path, section_content) tuples.
    """
    sections: list[tuple[list[str], str]] = []
    current_path: list[str] = []
    current_content: list[str] = []
    current_levels: list[int] = []

    lines = content.split("\n")

    for line in lines:
        if line.startswith("#"):
            # Save previous section
            section_text = "\n".join(current_content).strip()
            if section_text:
                sections.append((list(current_path), section_text))
            current_content = []

            # Parse heading
            level = len(line) - len(line.lstrip("#"))
            heading_text = line.lstrip("#").strip()

            # Update path
            while current_levels and current_levels[-1] >= level:
                current_levels.pop()
                if current_path:
                    current_path.pop()

            current_path.append(heading_text)
            current_levels.append(level)
        else:
            current_content.append(line)

    # Don't forget last section
    section_text = "\n".join(current_content).strip()
    if section_text:
        sections.append((list(current_path), section_text))

    return sections


def heading_to_anchor(heading: str) -> str:
    """Convert a heading to a GitHub-compatible anchor.

    Algorithm matches GitHub's anchor generation:
    1. Convert to lowercase
    2. Remove anything that is not a letter, number, space, or hyphen
    3. Replace spaces with hyphens
    4. Strip leading/trailing hyphens

    Examples:
        "Atributos" -> "atributos"
        "Ciclo de Vida" -> "ciclo-de-vida"
        "Entity: User" -> "entity-user"
        "## Estados (v2)" -> "estados-v2"
    """
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", heading)
    # Lowercase
    text = text.lower()
    # Remove anything not alphanumeric, space, or hyphen
    text = re.sub(r"[^\w\s-]", "", text)
    # Replace whitespace with hyphens
    text = re.sub(r"[\s]+", "-", text)
    # Strip leading/trailing hyphens
    text = text.strip("-")
    return text


def heading_path_to_anchor(heading_path: list[str]) -> str | None:
    """Convert a heading path to an anchor using the most specific heading.

    Uses the last element of the heading_path (most specific section).
    Returns None if the heading_path is empty.
    """
    if not heading_path:
        return None
    return heading_to_anchor(heading_path[-1])


def extract_snippet(content: str, max_length: int = 200) -> str:
    """Extract a snippet for preview from content.

    Truncates at sentence or word boundary, adds ellipsis if truncated.
    """
    # Strip markdown formatting for cleaner snippets
    text = content.strip()
    # Remove heading markers
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # Remove link syntax, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_length:
        return text

    # Try to break at sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind(". ")
    if last_period > max_length // 2:
        return truncated[: last_period + 1]

    # Break at word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        return truncated[:last_space] + "..."

    return truncated + "..."
