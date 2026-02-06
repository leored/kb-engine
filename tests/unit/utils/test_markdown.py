"""Tests for markdown utility functions."""

import pytest

from kb_engine.utils.markdown import (
    extract_frontmatter,
    extract_snippet,
    heading_path_to_anchor,
    heading_to_anchor,
    parse_markdown_sections,
)


@pytest.mark.unit
class TestHeadingToAnchor:
    """Tests for heading_to_anchor function."""

    def test_simple_heading(self) -> None:
        assert heading_to_anchor("Atributos") == "atributos"

    def test_heading_with_spaces(self) -> None:
        assert heading_to_anchor("Ciclo de Vida") == "ciclo-de-vida"

    def test_heading_with_colon(self) -> None:
        assert heading_to_anchor("Entity: User") == "entity-user"

    def test_heading_with_parentheses(self) -> None:
        assert heading_to_anchor("Estados (v2)") == "estados-v2"

    def test_heading_with_accents(self) -> None:
        assert heading_to_anchor("Descripción") == "descripcion"

    def test_heading_with_special_chars(self) -> None:
        assert heading_to_anchor("¿Qué es?") == "que-es"

    def test_empty_heading(self) -> None:
        assert heading_to_anchor("") == ""

    def test_heading_all_special(self) -> None:
        assert heading_to_anchor("---") == ""


@pytest.mark.unit
class TestHeadingPathToAnchor:
    """Tests for heading_path_to_anchor function."""

    def test_with_path(self) -> None:
        assert heading_path_to_anchor(["Usuario", "Atributos"]) == "atributos"

    def test_single_element(self) -> None:
        assert heading_path_to_anchor(["Descripción"]) == "descripcion"

    def test_empty_path(self) -> None:
        assert heading_path_to_anchor([]) is None


@pytest.mark.unit
class TestExtractSnippet:
    """Tests for extract_snippet function."""

    def test_short_content(self) -> None:
        text = "This is short content."
        assert extract_snippet(text) == "This is short content."

    def test_long_content_truncated(self) -> None:
        text = "A" * 300
        snippet = extract_snippet(text, max_length=200)
        assert len(snippet) <= 203  # 200 + "..."

    def test_strips_markdown(self) -> None:
        text = "**Bold text** and *italic* with [link](http://example.com)"
        snippet = extract_snippet(text)
        assert "**" not in snippet
        assert "*" not in snippet
        assert "](http" not in snippet
        assert "Bold text" in snippet
        assert "link" in snippet

    def test_sentence_boundary_truncation(self) -> None:
        text = "First sentence. Second sentence. " + "A" * 200
        snippet = extract_snippet(text, max_length=50)
        # Should truncate at sentence boundary if possible
        assert snippet.endswith(".") or snippet.endswith("...")


@pytest.mark.unit
class TestExtractFrontmatter:
    """Tests for extract_frontmatter function."""

    def test_with_frontmatter(self) -> None:
        content = """---
title: Test
tags:
  - entity
---

# Content here
"""
        metadata, body = extract_frontmatter(content)
        assert metadata["title"] == "Test"
        assert metadata["tags"] == ["entity"]
        assert "# Content here" in body

    def test_without_frontmatter(self) -> None:
        content = "# Just a heading\n\nSome content."
        metadata, body = extract_frontmatter(content)
        assert metadata == {}
        assert "# Just a heading" in body


@pytest.mark.unit
class TestParseMarkdownSections:
    """Tests for parse_markdown_sections function."""

    def test_basic_sections(self) -> None:
        content = """# Title

Intro text.

## Section A

Section A content.

## Section B

Section B content.
"""
        sections = parse_markdown_sections(content)
        assert len(sections) >= 3
        assert sections[0][0] == ["Title"]
        assert "Intro text" in sections[0][1]

    def test_nested_sections(self) -> None:
        content = """# Doc

## Parent

### Child

Child content.
"""
        sections = parse_markdown_sections(content)
        # The child section should have full path
        child_sections = [s for s in sections if len(s[0]) == 3]
        assert len(child_sections) == 1
        assert child_sections[0][0] == ["Doc", "Parent", "Child"]
