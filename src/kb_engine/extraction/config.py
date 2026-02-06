"""Extraction configuration."""

from pydantic import BaseModel, Field


class ExtractionConfig(BaseModel):
    """Configuration for the entity extraction process.

    These values follow the recommendations in ADR-0003 for
    entity and relationship extraction.
    """

    # Extraction behavior
    use_llm: bool = Field(
        default=False, description="Use LLM for extraction (vs. pattern-only)"
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence for extracted entities"
    )

    # Extractor selection
    enable_frontmatter_extraction: bool = Field(
        default=True, description="Extract entities from frontmatter"
    )
    enable_pattern_extraction: bool = Field(
        default=True, description="Extract entities using patterns"
    )
    enable_llm_extraction: bool = Field(
        default=False, description="Extract entities using LLM"
    )

    # LLM settings
    llm_model: str = Field(
        default="gpt-4-turbo-preview", description="LLM model to use for extraction"
    )
    llm_temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, description="LLM temperature for extraction"
    )

    # Deduplication
    deduplicate_entities: bool = Field(
        default=True, description="Deduplicate extracted entities"
    )
    similarity_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Threshold for entity deduplication"
    )

    class Config:
        frozen = True
