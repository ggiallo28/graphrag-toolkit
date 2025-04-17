import kuzu
import json
import logging
import time
import uuid
from typing import Optional, Any

from graphrag_toolkit.lexical_graph.storage.graph import (
    GraphStoreFactoryMethod,
    GraphStore,
    NodeId,
    get_log_formatting,
)

from llama_index.core.bridge.pydantic import PrivateAttr

logger = logging.getLogger(__name__)

KUZU_DATABASE = "kuzu-db://"


def format_id_for_kuzu(id_name: str):
    parts = id_name.split(".")
    if len(parts) == 1:
        return NodeId(parts[0], "`~id`", False)
    else:
        return NodeId(parts[1], f"id({parts[0]})", False)


class KuzuGraphStoreFactory(GraphStoreFactoryMethod):
    def try_create(self, graph_info: str, **kwargs) -> GraphStore:
        if graph_info.startswith(KUZU_DATABASE):
            db_path = graph_info[len(KUZU_DATABASE) :]
            logger.debug(f"Opening Kùzu database [path: {db_path}]")
            return KuzuClient(
                db_path=db_path, log_formatting=get_log_formatting(kwargs)
            )
        return None


class KuzuClient(GraphStore):
    db_path: str
    _db: Optional[Any] = PrivateAttr(default=None)
    _conn: Optional[Any] = PrivateAttr(default=None)

    def __getstate__(self):
        self._db = None
        self._conn = None
        return super().__getstate__()

    @property
    def connection(self):
        if self._db is None or self._conn is None:
            self._db = kuzu.Database(self.db_path)
            self._conn = kuzu.Connection(self._db)
        return self._conn

    def node_id(self, id_name: str) -> NodeId:
        return format_id_for_kuzu(id_name)

    def execute_query(self, cypher, parameters={}, correlation_id=None):
        query_id = uuid.uuid4().hex[:5]

        request_log_entry_parameters = self.log_formatting.format_log_entry(
            self._logging_prefix(query_id, correlation_id), cypher, parameters
        )

        logger.debug(
            f"[{request_log_entry_parameters.query_ref}] Query: [query: {request_log_entry_parameters.query}, parameters: {request_log_entry_parameters.parameters}]"
        )

        start = time.time()

        try:
            result = self.connection.execute(
                request_log_entry_parameters.format_query_with_query_ref(cypher)
            )
            results = (
                result.to_dict()
            )  # Kùzu supports conversion to dict from result object
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

        end = time.time()

        if logger.isEnabledFor(logging.DEBUG):
            response_log_entry_parameters = self.log_formatting.format_log_entry(
                self._logging_prefix(query_id, correlation_id),
                cypher,
                parameters,
                results,
            )
            logger.debug(
                f"[{response_log_entry_parameters.query_ref}] {int((end - start) * 1000)}ms Results: [{response_log_entry_parameters.results}]"
            )

        return results
