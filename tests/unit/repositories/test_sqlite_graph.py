"""Tests for SQLite graph repository."""

import pytest
from uuid import uuid4

from kb_engine.core.models.graph import Edge, EdgeType, Node, NodeType
from kb_engine.repositories.graph.sqlite import SQLiteGraphRepository


@pytest.fixture
async def graph_repo(tmp_path) -> SQLiteGraphRepository:
    """Create a SQLite graph repository for testing."""
    db_path = str(tmp_path / "test.db")
    repo = SQLiteGraphRepository(db_path=db_path)
    await repo.initialize()
    yield repo
    await repo.close()


@pytest.mark.unit
class TestSQLiteGraphRepository:
    """Tests for SQLiteGraphRepository."""

    @pytest.mark.asyncio
    async def test_create_and_get_node(self, graph_repo: SQLiteGraphRepository) -> None:
        node = Node(
            name="User",
            node_type=NodeType.ENTITY,
            description="A user entity",
        )
        created = await graph_repo.create_node(node)
        assert created.name == "User"

        fetched = await graph_repo.get_node(node.id)
        assert fetched is not None
        assert fetched.name == "User"
        assert fetched.node_type == NodeType.ENTITY

    @pytest.mark.asyncio
    async def test_find_nodes_by_type(self, graph_repo: SQLiteGraphRepository) -> None:
        await graph_repo.create_node(
            Node(name="User", node_type=NodeType.ENTITY)
        )
        await graph_repo.create_node(
            Node(name="Login", node_type=NodeType.USE_CASE)
        )
        await graph_repo.create_node(
            Node(name="Admin", node_type=NodeType.ENTITY)
        )

        entities = await graph_repo.find_nodes(node_type="entity")
        assert len(entities) == 2

        use_cases = await graph_repo.find_nodes(node_type="use_case")
        assert len(use_cases) == 1

    @pytest.mark.asyncio
    async def test_find_nodes_by_name(self, graph_repo: SQLiteGraphRepository) -> None:
        await graph_repo.create_node(
            Node(name="User", node_type=NodeType.ENTITY)
        )
        await graph_repo.create_node(
            Node(name="Admin", node_type=NodeType.ENTITY)
        )

        results = await graph_repo.find_nodes(name_pattern="Us")
        assert len(results) == 1
        assert results[0].name == "User"

    @pytest.mark.asyncio
    async def test_create_and_get_edges(self, graph_repo: SQLiteGraphRepository) -> None:
        node1 = Node(name="User", node_type=NodeType.ENTITY)
        node2 = Node(name="Login", node_type=NodeType.USE_CASE)
        await graph_repo.create_node(node1)
        await graph_repo.create_node(node2)

        edge = Edge(
            source_id=node1.id,
            target_id=node2.id,
            edge_type=EdgeType.PERFORMS,
        )
        await graph_repo.create_edge(edge)

        edges = await graph_repo.get_edges(node1.id, direction="out")
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.PERFORMS

    @pytest.mark.asyncio
    async def test_traverse(self, graph_repo: SQLiteGraphRepository) -> None:
        # Create a simple graph: A -> B -> C
        a = Node(name="A", node_type=NodeType.ENTITY)
        b = Node(name="B", node_type=NodeType.ENTITY)
        c = Node(name="C", node_type=NodeType.ENTITY)
        await graph_repo.create_node(a)
        await graph_repo.create_node(b)
        await graph_repo.create_node(c)

        await graph_repo.create_edge(
            Edge(source_id=a.id, target_id=b.id, edge_type=EdgeType.RELATED_TO)
        )
        await graph_repo.create_edge(
            Edge(source_id=b.id, target_id=c.id, edge_type=EdgeType.RELATED_TO)
        )

        # Traverse with max_hops=1 should find B
        results_1 = await graph_repo.traverse(a.id, max_hops=1)
        target_names_1 = {t.name for _, _, t in results_1}
        assert "B" in target_names_1

        # Traverse with max_hops=2 should find B and C
        results_2 = await graph_repo.traverse(a.id, max_hops=2)
        target_names_2 = {t.name for _, _, t in results_2}
        assert "B" in target_names_2
        assert "C" in target_names_2

    @pytest.mark.asyncio
    async def test_delete_by_document(self, graph_repo: SQLiteGraphRepository) -> None:
        doc_id = uuid4()
        node = Node(
            name="User",
            node_type=NodeType.ENTITY,
            source_document_id=doc_id,
        )
        await graph_repo.create_node(node)

        deleted = await graph_repo.delete_by_document(doc_id)
        assert deleted >= 1

        fetched = await graph_repo.get_node(node.id)
        assert fetched is None
