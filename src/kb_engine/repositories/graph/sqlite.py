"""SQLite implementation of the graph repository."""

import json
from uuid import UUID

import aiosqlite
import structlog

from kb_engine.core.models.graph import Edge, EdgeType, Node, NodeType

logger = structlog.get_logger(__name__)

CREATE_GRAPH_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,
    external_id TEXT,
    name TEXT NOT NULL,
    node_type TEXT NOT NULL,
    description TEXT,
    source_document_id TEXT,
    source_chunk_id TEXT,
    properties TEXT DEFAULT '{}',
    confidence REAL DEFAULT 1.0,
    extraction_method TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    name TEXT,
    properties TEXT DEFAULT '{}',
    weight REAL DEFAULT 1.0,
    source_document_id TEXT,
    source_chunk_id TEXT,
    confidence REAL DEFAULT 1.0,
    extraction_method TEXT,
    created_at TEXT,
    FOREIGN KEY (source_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_name ON graph_nodes(name);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_document ON graph_nodes(source_document_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_document ON graph_edges(source_document_id);
"""


class SQLiteGraphRepository:
    """SQLite implementation for knowledge graph storage.

    Stores nodes and edges in the same SQLite database as traceability.
    Uses recursive CTEs for graph traversal.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize graph tables in the database."""
        from pathlib import Path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(CREATE_GRAPH_TABLES_SQL)
        await self._db.commit()
        logger.info("SQLite graph repository initialized", db_path=self._db_path)

    async def _ensure_connected(self) -> aiosqlite.Connection:
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        return self._db

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # --- Node operations ---

    async def create_node(self, node: Node) -> Node:
        db = await self._ensure_connected()
        await db.execute(
            """INSERT OR REPLACE INTO graph_nodes
            (id, external_id, name, node_type, description,
             source_document_id, source_chunk_id, properties,
             confidence, extraction_method, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(node.id),
                node.external_id,
                node.name,
                node.node_type.value,
                node.description,
                str(node.source_document_id) if node.source_document_id else None,
                str(node.source_chunk_id) if node.source_chunk_id else None,
                json.dumps(node.properties),
                node.confidence,
                node.extraction_method,
                node.created_at.isoformat(),
                node.updated_at.isoformat(),
            ),
        )
        await db.commit()
        return node

    async def get_node(self, node_id: UUID) -> Node | None:
        db = await self._ensure_connected()
        cursor = await db.execute(
            "SELECT * FROM graph_nodes WHERE id = ?", (str(node_id),)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    async def find_nodes(
        self,
        node_type: str | None = None,
        name_pattern: str | None = None,
        limit: int = 100,
    ) -> list[Node]:
        db = await self._ensure_connected()
        query = "SELECT * FROM graph_nodes"
        params: list = []
        conditions: list[str] = []

        if node_type:
            conditions.append("node_type = ?")
            params.append(node_type)
        if name_pattern:
            conditions.append("name LIKE ?")
            params.append(f"%{name_pattern}%")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " LIMIT ?"
        params.append(limit)

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_node(row) for row in rows]

    # --- Edge operations ---

    async def create_edge(self, edge: Edge) -> Edge:
        db = await self._ensure_connected()
        await db.execute(
            """INSERT OR REPLACE INTO graph_edges
            (id, source_id, target_id, edge_type, name, properties,
             weight, source_document_id, source_chunk_id,
             confidence, extraction_method, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(edge.id),
                str(edge.source_id),
                str(edge.target_id),
                edge.edge_type.value,
                edge.name,
                json.dumps(edge.properties),
                edge.weight,
                str(edge.source_document_id) if edge.source_document_id else None,
                str(edge.source_chunk_id) if edge.source_chunk_id else None,
                edge.confidence,
                edge.extraction_method,
                edge.created_at.isoformat(),
            ),
        )
        await db.commit()
        return edge

    async def get_edges(
        self,
        node_id: UUID,
        direction: str = "both",
        edge_type: str | None = None,
    ) -> list[Edge]:
        db = await self._ensure_connected()
        node_str = str(node_id)

        if direction == "out":
            query = "SELECT * FROM graph_edges WHERE source_id = ?"
            params: list = [node_str]
        elif direction == "in":
            query = "SELECT * FROM graph_edges WHERE target_id = ?"
            params = [node_str]
        else:
            query = "SELECT * FROM graph_edges WHERE source_id = ? OR target_id = ?"
            params = [node_str, node_str]

        if edge_type:
            query += " AND edge_type = ?"
            params.append(edge_type)

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_edge(row) for row in rows]

    async def traverse(
        self,
        start_node_id: UUID,
        max_hops: int = 2,
        edge_types: list[str] | None = None,
    ) -> list[tuple[Node, Edge, Node]]:
        """Traverse the graph using recursive CTEs."""
        db = await self._ensure_connected()

        edge_filter = ""
        params: list = [str(start_node_id), max_hops]
        if edge_types:
            placeholders = ",".join("?" * len(edge_types))
            edge_filter = f"AND e.edge_type IN ({placeholders})"
            params = [str(start_node_id)] + edge_types + [max_hops] + edge_types

        query = f"""
        WITH RECURSIVE path AS (
            SELECT e.id as edge_id, e.source_id, e.target_id, 1 as depth
            FROM graph_edges e
            WHERE e.source_id = ? {edge_filter}
            UNION ALL
            SELECT e.id as edge_id, e.source_id, e.target_id, p.depth + 1
            FROM graph_edges e
            JOIN path p ON e.source_id = p.target_id
            WHERE p.depth < ? {edge_filter}
        )
        SELECT DISTINCT
            p.edge_id,
            sn.id as source_node_id, sn.name as source_name,
            sn.node_type as source_type, sn.description as source_desc,
            sn.properties as source_props, sn.confidence as source_confidence,
            sn.extraction_method as source_extraction,
            sn.source_document_id as source_doc_id,
            sn.source_chunk_id as source_chunk_id,
            sn.created_at as source_created, sn.updated_at as source_updated,
            sn.external_id as source_external_id,
            tn.id as target_node_id, tn.name as target_name,
            tn.node_type as target_type, tn.description as target_desc,
            tn.properties as target_props, tn.confidence as target_confidence,
            tn.extraction_method as target_extraction,
            tn.source_document_id as target_doc_id,
            tn.source_chunk_id as target_chunk_id,
            tn.created_at as target_created, tn.updated_at as target_updated,
            tn.external_id as target_external_id,
            e.edge_type, e.name as edge_name, e.properties as edge_props,
            e.weight, e.confidence as edge_confidence,
            e.extraction_method as edge_extraction,
            e.source_document_id as edge_doc_id,
            e.source_chunk_id as edge_chunk_id,
            e.created_at as edge_created
        FROM path p
        JOIN graph_edges e ON e.id = p.edge_id
        JOIN graph_nodes sn ON sn.id = p.source_id
        JOIN graph_nodes tn ON tn.id = p.target_id
        """

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        results: list[tuple[Node, Edge, Node]] = []
        for row in rows:
            source = Node(
                id=UUID(row["source_node_id"]),
                external_id=row["source_external_id"],
                name=row["source_name"],
                node_type=NodeType(row["source_type"]),
                description=row["source_desc"],
                source_document_id=UUID(row["source_doc_id"]) if row["source_doc_id"] else None,
                source_chunk_id=UUID(row["source_chunk_id"]) if row["source_chunk_id"] else None,
                properties=json.loads(row["source_props"]) if row["source_props"] else {},
                confidence=row["source_confidence"],
                extraction_method=row["source_extraction"],
            )
            target = Node(
                id=UUID(row["target_node_id"]),
                external_id=row["target_external_id"],
                name=row["target_name"],
                node_type=NodeType(row["target_type"]),
                description=row["target_desc"],
                source_document_id=UUID(row["target_doc_id"]) if row["target_doc_id"] else None,
                source_chunk_id=UUID(row["target_chunk_id"]) if row["target_chunk_id"] else None,
                properties=json.loads(row["target_props"]) if row["target_props"] else {},
                confidence=row["target_confidence"],
                extraction_method=row["target_extraction"],
            )
            edge = Edge(
                id=UUID(row["edge_id"]),
                source_id=UUID(row["source_node_id"]),
                target_id=UUID(row["target_node_id"]),
                edge_type=EdgeType(row["edge_type"]),
                name=row["edge_name"],
                properties=json.loads(row["edge_props"]) if row["edge_props"] else {},
                weight=row["weight"],
                source_document_id=UUID(row["edge_doc_id"]) if row["edge_doc_id"] else None,
                source_chunk_id=UUID(row["edge_chunk_id"]) if row["edge_chunk_id"] else None,
                confidence=row["edge_confidence"],
                extraction_method=row["edge_extraction"],
            )
            results.append((source, edge, target))

        return results

    async def delete_by_document(self, document_id: UUID) -> int:
        db = await self._ensure_connected()
        doc_str = str(document_id)
        cursor = await db.execute(
            "DELETE FROM graph_edges WHERE source_document_id = ?", (doc_str,)
        )
        edge_count = cursor.rowcount
        cursor = await db.execute(
            "DELETE FROM graph_nodes WHERE source_document_id = ?", (doc_str,)
        )
        node_count = cursor.rowcount
        await db.commit()
        return edge_count + node_count

    async def find_similar_nodes(
        self,
        node_id: UUID,
        limit: int = 10,
    ) -> list[tuple[Node, float]]:
        """Find similar nodes by shared edges."""
        db = await self._ensure_connected()
        node_str = str(node_id)
        cursor = await db.execute(
            """SELECT n.*, COUNT(DISTINCT e2.id) as shared_edges
            FROM graph_nodes n
            JOIN graph_edges e2 ON (e2.source_id = n.id OR e2.target_id = n.id)
            WHERE e2.target_id IN (
                SELECT target_id FROM graph_edges WHERE source_id = ?
            ) OR e2.source_id IN (
                SELECT source_id FROM graph_edges WHERE target_id = ?
            )
            AND n.id != ?
            GROUP BY n.id
            ORDER BY shared_edges DESC
            LIMIT ?""",
            (node_str, node_str, node_str, limit),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            node = self._row_to_node(row)
            similarity = float(row["shared_edges"]) / 10.0  # Normalize
            results.append((node, min(similarity, 1.0)))
        return results

    # --- Row mapping ---

    @staticmethod
    def _row_to_node(row: aiosqlite.Row) -> Node:
        from datetime import datetime

        return Node(
            id=UUID(row["id"]),
            external_id=row["external_id"],
            name=row["name"],
            node_type=NodeType(row["node_type"]),
            description=row["description"],
            source_document_id=UUID(row["source_document_id"]) if row["source_document_id"] else None,
            source_chunk_id=UUID(row["source_chunk_id"]) if row["source_chunk_id"] else None,
            properties=json.loads(row["properties"]) if row["properties"] else {},
            confidence=row["confidence"],
            extraction_method=row["extraction_method"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.utcnow(),
        )

    @staticmethod
    def _row_to_edge(row: aiosqlite.Row) -> Edge:
        from datetime import datetime

        return Edge(
            id=UUID(row["id"]),
            source_id=UUID(row["source_id"]),
            target_id=UUID(row["target_id"]),
            edge_type=EdgeType(row["edge_type"]),
            name=row["name"],
            properties=json.loads(row["properties"]) if row["properties"] else {},
            weight=row["weight"],
            source_document_id=UUID(row["source_document_id"]) if row["source_document_id"] else None,
            source_chunk_id=UUID(row["source_chunk_id"]) if row["source_chunk_id"] else None,
            confidence=row["confidence"],
            extraction_method=row["extraction_method"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
        )
