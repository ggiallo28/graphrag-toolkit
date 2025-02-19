# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union, Any
from pipe import Pipe
from dataclasses import dataclass, field

from graphrag_toolkit.storage import GraphStoreFactory, GraphStoreType
from graphrag_toolkit.storage import VectorStoreFactory, VectorStoreType
from graphrag_toolkit.storage.graph_store import DummyGraphStore
from graphrag_toolkit.indexing.extract import BatchConfig
from graphrag_toolkit.indexing import NodeHandler
from graphrag_toolkit.indexing import sink
from graphrag_toolkit.indexing.constants import PROPOSITIONS_KEY, DEFAULT_ENTITY_CLASSIFICATIONS
from graphrag_toolkit.indexing.extract import ScopedValueProvider, FixedScopedValueProvider, DEFAULT_SCOPE
from graphrag_toolkit.indexing.extract import GraphScopedValueStore
from graphrag_toolkit.indexing.extract import LLMPropositionExtractor, BatchLLMPropositionExtractor
from graphrag_toolkit.indexing.extract import TopicExtractor, BatchTopicExtractor
from graphrag_toolkit.indexing.extract import ExtractionPipeline
from graphrag_toolkit.indexing.extract import InferClassifications, OnExistingClassifications, InferClassificationsConfig
from graphrag_toolkit.indexing.build import BuildPipeline
from graphrag_toolkit.indexing.build import VectorIndexing
from graphrag_toolkit.indexing.build import GraphConstruction
from graphrag_toolkit.indexing.build import Checkpoint
from graphrag_toolkit.indexing.build import Filter
from graphrag_toolkit.indexing.build.null_builder import NullBuilder

from llama_index.core.node_parser import SentenceSplitter, NodeParser
from llama_index.core.schema import BaseNode, NodeRelationship

DEFAULT_INDEX_NAME = 'default-index'
DEFAULT_EXTRACTION_DIR = 'output'

class ExtractionConfig():
    def __init__(self, 
                 enable_proposition_extraction:bool=True,
                 preferred_entity_classifications:List[str]=DEFAULT_ENTITY_CLASSIFICATIONS,
                 infer_entity_classifications:Union[InferClassificationsConfig, bool]=False):
        
        self.enable_proposition_extraction = enable_proposition_extraction
        self.preferred_entity_classifications = preferred_entity_classifications
        self.infer_entity_classifications = infer_entity_classifications

class IndexingConfig():
    def __init__(self,
                 chunking:Optional[List[NodeParser]]=[],
                 extraction:Optional[ExtractionConfig]=None,
                 batch_config:Optional[BatchConfig]=None):
        
        if chunking is not None and len(chunking) == 0:
            chunking.append(SentenceSplitter(chunk_size=256, chunk_overlap=20))
        
        self.chunking = chunking
        self.extraction = extraction or ExtractionConfig()
        self.batch_config = batch_config

def get_topic_scope(node:BaseNode):
    source = node.relationships.get(NodeRelationship.SOURCE, None)
    if not source:
        return DEFAULT_SCOPE
    else:
        return source.node_id

class LexicalGraphIndex():
    """
    Extracts graph elements from source documents and builds a graph and vector index.

    Args:
        graph_store (Optional[GraphStoreType], optional):
            GraphStore instance or GraphStore connection string. If None, defaults to a DummyGraphStore.
        vector_store (Optional[VectorStoreType], optional):
            VectorStore instance or VectorStore connection string. If None, defaults to a VectorStore with DummyVectorIndexes.
        index_name (str, optional):
            Unique name of the index. Defaults to DEFAULT_INDEX_NAME.
        extraction_dir (List[TransformComponent], optional):
            Directory to which intermediate artefacts (e.g. checkpoints) will be written. Defaults to DEFAULT_EXTRACTION_DIR.
        indexing_config (Optional[IndexingConfig], optional):
            If None, defaults to using default IndexingConfig values.

    Examples:
        ```python
        from graphrag_toolkit import TopicGraphIndex

        index = TopicGraphIndex(
            'neptune-db://gr-1123456789.cluster-abcdefghijk.us-east-1.neptune.amazonaws.com', 
            'aoss://https://abcdefghijk.us-east-1.aoss.amazonaws.com'
        )
        index.extract_and_build(documents, show_progress=True)
        ```
    """

    def __init__(
            self,
            graph_store:Optional[GraphStoreType]=None,
            vector_store:Optional[VectorStoreType]=None,
            index_name:Optional[str]=None,
            extraction_dir:Optional[str]=None,
            indexing_config:Optional[IndexingConfig]=None,
        ):

        self.graph_store = GraphStoreFactory.for_graph_store(graph_store)
        self.vector_store = VectorStoreFactory.for_vector_store(vector_store)
        self.index_name = index_name or DEFAULT_INDEX_NAME
        self.extraction_dir = extraction_dir or DEFAULT_EXTRACTION_DIR
        self.indexing_config = indexing_config or IndexingConfig()

        (pre_processors, components) = self._configure_extraction_pipeline(self.indexing_config)

        self.extraction_pre_processors = pre_processors
        self.extraction_components = components
        self.allow_batch_inference = self.indexing_config.batch_config is not None


    def _configure_extraction_pipeline(self, config:IndexingConfig):
        
        pre_processors = []
        components = []

        if config.chunking:
            for c in config.chunking:
                components.append(c)
        
        if config.extraction.enable_proposition_extraction:
            if config.batch_config:
                components.append(BatchLLMPropositionExtractor(batch_config=config.batch_config))
            else:
                components.append(LLMPropositionExtractor())

        entity_classification_provider = None
        topic_provider = None

        classification_label = f'{self.index_name}_EntityClassification'
        classification_scope = DEFAULT_SCOPE
        
        if isinstance(self.graph_store, DummyGraphStore):
            entity_classification_provider = FixedScopedValueProvider(scoped_values={DEFAULT_SCOPE: config.extraction.preferred_entity_classifications})
            topic_provider = FixedScopedValueProvider(scoped_values={DEFAULT_SCOPE: []})
        else:
            initial_scope_values = [] if config.extraction.infer_entity_classifications else config.extraction.preferred_entity_classifications
            entity_classification_provider = ScopedValueProvider(
                label=classification_label,
                scoped_value_store=GraphScopedValueStore(graph_store=self.graph_store),
                initial_scoped_values = { classification_scope: initial_scope_values }
            )           
            topic_provider = ScopedValueProvider(
                label=f'{self.index_name}_StatementTopic',
                scoped_value_store=GraphScopedValueStore(graph_store=self.graph_store),
                scope_func=get_topic_scope
            )

        if config.extraction.infer_entity_classifications:
            infer_config = config.extraction.infer_entity_classifications if isinstance(config.extraction.infer_entity_classifications, InferClassificationsConfig) else InferClassificationsConfig()
            pre_processors.append(InferClassifications(
                classification_label=classification_label,
                classification_scope=classification_scope,
                classification_store=GraphScopedValueStore(graph_store=self.graph_store),
                splitter=SentenceSplitter(chunk_size=256, chunk_overlap=20) if config.chunking else None,
                default_classifications=config.extraction.preferred_entity_classifications,
                num_samples=infer_config.num_samples,
                num_iterations=infer_config.num_iterations,
                merge_action=infer_config.on_existing_classifications
            ))

        topic_extractor = None

        if config.batch_config:
            topic_extractor = BatchTopicExtractor(
                batch_config=config.batch_config,
                source_metadata_field=PROPOSITIONS_KEY if config.extraction.enable_proposition_extraction else None,
                entity_classification_provider=entity_classification_provider,
                topic_provider=topic_provider
            )
        else:
            topic_extractor = TopicExtractor(
                source_metadata_field=PROPOSITIONS_KEY if config.extraction.enable_proposition_extraction else None,
                entity_classification_provider=entity_classification_provider,
                topic_provider=topic_provider
            )

        components.append(topic_extractor)

        return (pre_processors, components)
        
    def extract(
            self,
            nodes:List[BaseNode]=[],
            handler:Optional[NodeHandler]=None,
            checkpoint:Optional[Checkpoint]=None,
            show_progress:Optional[bool]=False,
            **kwargs:Any) -> None:
        """
        Run a series of transformations on a set of nodes to extract graph elements.

        Args:
            nodes (List[BaseNode], optional): Set of nodes from which graph elements are to be extracted.
            handler (Optional[NodeHandler], optional): Handles nodes emitted at the end of the series of transformations. 
            checkpoint (Optional[Checkpoint], optional): Nodes that are successfully processed by all transformations are 
                checkpointed to the extraction directory, so that they are not reprocessed on subsequent invocations.
            show_progress (bool, optional): Shows execution progress bar(s). Defaults to False.

        Examples:
        ```python
        from graphrag_toolkit.indexing.load import FileBasedChunks

        index.extract(documents, handler=FileBasedChunks('./output/file-based-chunks/'), show_progress=True)
        ```
        """

        extraction_pipeline = ExtractionPipeline.create(
            components=self.extraction_components,
            pre_processors=self.extraction_pre_processors,
            show_progress=show_progress,
            checkpoint=checkpoint,
            num_workers=1 if self.allow_batch_inference else None,
            **kwargs
        )

        build_pipeline = BuildPipeline.create(
            components=[
                NullBuilder()
            ],
            show_progress=show_progress,
            checkpoint=checkpoint,
            num_workers=1,
            batch_size=5,
            **kwargs
        )

        if handler:
            nodes | extraction_pipeline | Pipe(handler.accept) | build_pipeline | sink
        else:
            nodes | extraction_pipeline | build_pipeline | sink

    def build(
            self,
            nodes:List[BaseNode]=[],
            handler:Optional[NodeHandler]=None,
            filter:Optional[Filter]=None,
            checkpoint:Optional[Checkpoint]=None,
            show_progress:Optional[bool]=False,
            **kwargs:Any) -> None:
        """
        Build a graph and vector index from previously extracted nodes.

        Args:
            nodes (List[BaseNode], optional): Set of previously extracted nodes.
            handler (Optional[NodeHandler], optional): Handles nodes emitted at the end of the build process.
            filter (Optional[Filter], optional): Filters out topics and statements
            checkpoint (Optional[Checkpoint], optional): Nodes that are successfully processed by all stages of the build process
                are checkpointed to the extraction directory, so that they are not reprocessed on subsequent invocations.
            show_progress (bool, optional): Shows execution progress bar(s). Defaults to False.

        Examples:
        ```python
        from graphrag_toolkit.indexing.load import FileBasedChunks

        index.build(FileBasedChunks('./output/file-based-chunks/'), show_progress=True)
        ```
        """

        build_pipeline = BuildPipeline.create(
            components=[
                GraphConstruction.for_graph_store(self.graph_store),
                VectorIndexing.for_vector_store(self.vector_store)
            ],
            show_progress=show_progress,
            checkpoint=checkpoint,
            filter=filter,
            **kwargs
        )

        sink_fn = sink if not handler else Pipe(handler.accept)
        nodes | build_pipeline | sink_fn
        
    def extract_and_build(
            self, 
            nodes:List[BaseNode]=[], 
            handler:Optional[NodeHandler]=None,
            filter:Optional[Filter]=None,
            checkpoint:Optional[Checkpoint]=None,
            show_progress:Optional[bool]=False,
            **kwargs:Any
        ) -> None:
        """
        Extract graph elements from a set of nodes and then build a graph and vector index.

        Args:
            nodes (List[BaseNode], optional): Set of nodes from which graph elements are to be extracted.
            handler (Optional[NodeHandler], optional): Handles nodes emitted at the end of the build process.
            filter (Optional[Filter], optional): Filters out topics and statements
            checkpoint (Optional[Checkpoint], optional): Nodes that are successfully processed by all stages of the build process
                are checkpointed to the extraction directory, so that they are not reprocessed on subsequent invocations.
            show_progress (bool, optional): Shows execution progress bar(s). Defaults to False.

        Examples:
        ```python
        index.extract_and_build(documents, show_progress=True)
        ```
        """

        extraction_pipeline = ExtractionPipeline.create(
            components=self.extraction_components,
            pre_processors=self.extraction_pre_processors,
            show_progress=show_progress,
            checkpoint=checkpoint,
            num_workers=1 if self.allow_batch_inference else None,
            **kwargs
        )
        
        build_pipeline = BuildPipeline.create(
            components=[
                GraphConstruction.for_graph_store(self.graph_store),
                VectorIndexing.for_vector_store(self.vector_store)
            ],
            show_progress=show_progress,
            checkpoint=checkpoint,
            filter=filter,
            **kwargs
        )

        sink_fn = sink if not handler else Pipe(handler.accept)
        nodes | extraction_pipeline | build_pipeline | sink_fn