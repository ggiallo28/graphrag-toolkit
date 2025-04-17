import duckdb
import logging
import json
from typing import List, Optional, Any, AsyncGenerator
from dataclasses import dataclass

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.async_utils import asyncio_run
from llama_index.core.vector_stores.types import (
    VectorStoreQueryResult,
    VectorStoreQueryMode,
    MetadataFilters,
)
from llama_index.core.indices.utils import embed_nodes

from graphrag_toolkit.lexical_graph.config import GraphRAGConfig, EmbeddingType
from graphrag_toolkit.lexical_graph.storage.vector import (
    VectorIndex,
    VectorIndexFactoryMethod,
    to_embedded_query,
)
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY

logger = logging.getLogger(__name__)

DUCKDB_DATABSE = "duckdb://"


@dataclass
class DuckDBVectorIndexFactory(VectorIndexFactoryMethod):
    def try_create(
        self, index_names: List[str], vector_index_info: str, **kwargs
    ) -> Optional[List[VectorIndex]]:
        if vector_index_info.startswith(DUCKDB_DATABSE):
            db_path = vector_index_info[len(DUCKDB_DATABSE) :]
            logger.debug(
                f"Opening DuckDB vector indexes [index_names: {index_names}, db_path: {db_path}]"
            )
            return [
                DuckDBIndex.for_index(index_name, db_path, **kwargs)
                for index_name in index_names
            ]
        return None


class DuckDBIndex(VectorIndex):
    @staticmethod
    def for_index(
        index_name: str,
        db_path: str,
        embed_model: Optional[EmbeddingType] = None,
        dimensions: Optional[int] = None,
        vector_results: bool = True,
        fts_results: bool = False,
    ):
        embed_model = embed_model or GraphRAGConfig.embed_model
        dimensions = dimensions or GraphRAGConfig.embed_dimensions
        return DuckDBIndex(
            index_name=index_name,
            db_path=db_path,
            dimensions=dimensions,
            embed_model=embed_model,
            vector_results=vector_results,
            fts_results=fts_results,
        )

    class Config:
        arbitrary_types_allowed = True

    index_name: str
    db_path: str
    dimensions: int
    embed_model: EmbeddingType
    vector_results: bool = True
    fts_results: bool = False

    _conn: duckdb.DuckDBPyConnection = PrivateAttr(default=None)

    def __getstate__(self):
        self._conn = None
        return super().__getstate__()

    def __del__(self):
        if self._conn:
            self._conn.close()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
            self._conn.execute("INSTALL fts; LOAD fts;")
            self._conn.execute("INSTALL vss; LOAD vss;")
        return self._conn

    def _clean_id(self, s):
        return "".join(c for c in s if c.isalnum())

    def add_embeddings(self, nodes: List[BaseNode]) -> List[BaseNode]:
        if not self.writeable:
            raise IndexError(f"Index {self.index_name()} is read-only")

        id_to_embed_map = embed_nodes(nodes, self.embed_model)
        data = [
            (node.node_id, node.text, id_to_embed_map[node.node_id]) for node in nodes
        ]

        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.index_name} (
                id VARCHAR PRIMARY KEY,
                text TEXT,
                embedding FLOAT[{self.dimensions}]
            );
        """
        )
        self.conn.execute(
            f"DELETE FROM {self.index_name} WHERE id IN ({','.join(['?']*len(data))})",
            [d[0] for d in data],
        )
        self.conn.executemany(f"INSERT INTO {self.index_name} VALUES (?, ?, ?)", data)

        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.index_name}_hnsw ON {self.index_name} USING HNSW (embedding);"
        )

        if self.fts_results:
            self.conn.execute(
                f"PRAGMA create_fts_index('{self.index_name}', 'id', 'text', overwrite=1);"
            )

        return nodes

    def top_k(self, query_bundle: QueryBundle, top_k: int = 5) -> List[dict]:
        query_bundle = to_embedded_query(query_bundle, self.embed_model)
        results = []

        if self.vector_results:
            vector_query = f"""
                SELECT id, text, array_cosine_similarity(embedding, {query_bundle.embedding}) AS score
                FROM {self.index_name}
                ORDER BY score DESC
                LIMIT {top_k}
            """
            vector_results = self.conn.execute(vector_query).fetchall()
            results.extend(
                [{"id": r[0], "text": r[1], "score": r[2]} for r in vector_results]
            )

        if self.fts_results:
            fts_query = f"""
                SELECT id, text, fts_main_{self.index_name}.match_bm25(id, '{query_bundle.query_str}') AS score
                FROM {self.index_name}
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT {top_k}
            """
            fts_results = self.conn.execute(fts_query).fetchall()
            results.extend(
                [{"id": r[0], "text": r[1], "score": r[2]} for r in fts_results]
            )

        if self.vector_results and self.fts_results:
            combined = {}
            for r in results:
                if r["id"] in combined:
                    combined[r["id"]]["score"] += r["score"]
                else:
                    combined[r["id"]] = r
            results = list(combined.values())
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]

        return results

    def get_embeddings(self, ids: List[str] = []) -> List[dict]:
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        query = f"SELECT id, text, embedding FROM {self.index_name} WHERE id IN ({placeholders})"
        rows = self.conn.execute(query, ids).fetchall()
        return [{"id": r[0], "text": r[1], "embedding": r[2]} for r in rows]

    async def paginated_search(
        self, query: str, page_size: int = 10000, max_pages: Optional[int] = None
    ) -> AsyncGenerator[List[dict], None]:
        # Embed the natural language query
        embedded_query = embed_nodes([BaseNode(text=query)], self.embed_model)[0]

        offset = 0
        page = 0

        while True:
            paged_query = f"""
                SELECT id, text, embedding, array_cosine_similarity(embedding, {embedded_query}) AS score
                FROM {self.index_name}
                ORDER BY score DESC
                LIMIT {page_size} OFFSET {offset}
            """
            rows = self.conn.execute(paged_query).fetchall()

            if not rows:
                break

            results = [
                {"id": r[0], "text": r[1], "embedding": r[2], "score": r[3]}
                for r in rows
            ]
            yield results

            offset += page_size
            page += 1

            if max_pages and page >= max_pages:
                break

    async def get_all_embeddings(
        self, query: str, max_results: Optional[int] = None
    ) -> List[dict]:
        all_results = []

        async for page in self.paginated_search(query):
            all_results.extend(page)
            if max_results and len(all_results) >= max_results:
                return all_results[:max_results]

        return all_results
