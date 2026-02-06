"""Tests for entity extraction functionality."""

import pytest

from kb_engine.core.models.document import Chunk, ChunkType, Document
from kb_engine.core.models.graph import NodeType
from kb_engine.extraction import ExtractionConfig, ExtractionPipelineFactory
from kb_engine.extraction.extractors import FrontmatterExtractor, PatternExtractor
from kb_engine.extraction.models import ExtractedNode


@pytest.mark.unit
class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ExtractionConfig()

        assert config.use_llm is False
        assert config.confidence_threshold == 0.7
        assert config.enable_frontmatter_extraction is True
        assert config.enable_pattern_extraction is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ExtractionConfig(
            use_llm=False,
            confidence_threshold=0.9,
        )

        assert config.use_llm is False
        assert config.confidence_threshold == 0.9


@pytest.mark.unit
class TestFrontmatterExtractor:
    """Tests for FrontmatterExtractor."""

    def test_extractor_properties(self) -> None:
        """Test extractor properties."""
        extractor = FrontmatterExtractor()

        assert extractor.name == "frontmatter"
        assert extractor.priority == 10

    def test_can_extract_with_metadata(self) -> None:
        """Test can_extract with document metadata."""
        extractor = FrontmatterExtractor()
        doc = Document(
            title="Test",
            content="Content",
            metadata={"key": "value"},
        )
        chunk = Chunk(document_id=doc.id, content="Content")

        assert extractor.can_extract(chunk, doc) is True

    def test_cannot_extract_without_metadata(self) -> None:
        """Test can_extract without document metadata."""
        extractor = FrontmatterExtractor()
        doc = Document(title="Test", content="Content", metadata={})
        chunk = Chunk(document_id=doc.id, content="Content")

        assert extractor.can_extract(chunk, doc) is False

    @pytest.mark.asyncio
    async def test_extract_from_frontmatter(self) -> None:
        """Test extraction from frontmatter."""
        extractor = FrontmatterExtractor()
        doc = Document(
            title="Test Document",
            content="Content",
            domain="test-domain",
            tags=["tag1", "tag2"],
            metadata={"tags": ["tag1", "tag2"]},
        )
        chunk = Chunk(document_id=doc.id, content="Content")

        result = await extractor.extract(chunk, doc)

        assert len(result.nodes) > 0
        # Should have document node and concept nodes for tags/domain
        node_types = [n.node_type for n in result.nodes]
        assert NodeType.DOCUMENT in node_types
        assert NodeType.CONCEPT in node_types


@pytest.mark.unit
class TestPatternExtractor:
    """Tests for PatternExtractor."""

    def test_extractor_properties(self) -> None:
        """Test extractor properties."""
        extractor = PatternExtractor()

        assert extractor.name == "pattern"
        assert extractor.priority == 20

    def test_can_extract_with_content(self) -> None:
        """Test can_extract with sufficient content."""
        extractor = PatternExtractor()
        doc = Document(title="Test", content="Content")
        chunk = Chunk(document_id=doc.id, content="This is sufficient content for extraction.")

        assert extractor.can_extract(chunk, doc) is True

    def test_cannot_extract_short_content(self) -> None:
        """Test can_extract with very short content."""
        extractor = PatternExtractor()
        doc = Document(title="Test", content="")
        chunk = Chunk(document_id=doc.id, content="Short")

        assert extractor.can_extract(chunk, doc) is False

    @pytest.mark.asyncio
    async def test_extract_actor(self) -> None:
        """Test extraction of actor entities."""
        extractor = PatternExtractor()
        doc = Document(title="Test", content="Content")
        chunk = Chunk(
            document_id=doc.id,
            content="The actor: Administrator manages the system.",
        )

        result = await extractor.extract(chunk, doc)

        actor_nodes = [n for n in result.nodes if n.node_type == NodeType.ACTOR]
        assert len(actor_nodes) > 0

    @pytest.mark.asyncio
    async def test_extract_relationship(self) -> None:
        """Test extraction of relationships."""
        extractor = PatternExtractor()
        doc = Document(title="Test", content="Content")
        chunk = Chunk(
            document_id=doc.id,
            content="The OrderService depends on PaymentService for processing.",
        )

        result = await extractor.extract(chunk, doc)

        assert len(result.edges) > 0


@pytest.mark.unit
class TestExtractionPipelineFactory:
    """Tests for ExtractionPipelineFactory."""

    def test_create_pipeline_default(self) -> None:
        """Test creating pipeline with default config."""
        factory = ExtractionPipelineFactory()
        pipeline = factory.create_pipeline()

        assert pipeline is not None

    def test_create_pipeline_no_llm(self) -> None:
        """Test creating pipeline without LLM."""
        config = ExtractionConfig(use_llm=False)
        factory = ExtractionPipelineFactory(config)
        pipeline = factory.create_pipeline()

        # Should still have frontmatter and pattern extractors
        assert pipeline is not None


@pytest.mark.unit
class TestExtractedNode:
    """Tests for ExtractedNode model."""

    def test_create_extracted_node(self) -> None:
        """Test creating an extracted node."""
        node = ExtractedNode(
            name="TestEntity",
            node_type=NodeType.ENTITY,
            description="A test entity",
            confidence=0.9,
            extraction_method="test",
        )

        assert node.name == "TestEntity"
        assert node.node_type == NodeType.ENTITY
        assert node.confidence == 0.9
        assert node.extraction_method == "test"
