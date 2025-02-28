# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, List
from graphrag_toolkit.storage.constants import ALL_EMBEDDING_INDEXES
from graphrag_toolkit.storage.vector_index import VectorIndex
from graphrag_toolkit.storage.vector_index_factory import VectorIndexFactory
from llama_index.core.bridge.pydantic import BaseModel, Field

class VectorStore(BaseModel):
    indexes:Optional[Dict[str, VectorIndex]] = Field(description='Vector indexes', default_factory=dict)

    def get_index(self, index_name):
        if index_name not in ALL_EMBEDDING_INDEXES:
            raise ValueError(f'Invalid index name ({index_name}): must be one of {ALL_EMBEDDING_INDEXES}')
        if index_name not in self.indexes:
            return VectorIndexFactory.for_dummy_vector_index(index_name)
        return self.indexes[index_name]

    def all_indexes(self) -> List[VectorIndex]:
        return list(self.indexes.values())

