# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Dict

from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.schema import NodeRelationship

from graphrag_toolkit.indexing.build.build_filter import BuildFilter
from graphrag_toolkit.indexing.build.node_builder import NodeBuilder
from graphrag_toolkit.indexing.model import TopicCollection, Topic, Statement
from graphrag_toolkit.indexing.constants import TOPICS_KEY
from graphrag_toolkit.storage.constants import INDEX_KEY

class TopicNodeBuilder(NodeBuilder):
    
    @classmethod
    def name(cls) -> str:
        return 'TopicNodeBuilder'
    
    @classmethod
    def metadata_keys(cls) -> List[str]:
        return [TOPICS_KEY]
    
    def _add_chunk_id(self, node:TextNode, chunk_id:str):
        
        topic = Topic.model_validate(node.metadata['topic'])

        existing_chunk_ids = dict.fromkeys(topic.chunkIds)
        existing_chunk_ids[chunk_id] = None

        topic.chunkIds = list(existing_chunk_ids.keys())

        node.metadata['topic'] = topic.model_dump(exclude_none=True)
        
        return node
    
    def _add_statements(self, node:TextNode, statements:List[Statement], filter:BuildFilter):
        
        existing_statements = dict.fromkeys(node.metadata['statements'])
                
        for statement in statements:
            if filter.ignore_statement(statement.value):
                continue
            existing_statements[statement.value] = None
            
        node.metadata['statements'] = list(existing_statements.keys())

        return node


    def build_nodes(self, nodes:List[BaseNode], filter:BuildFilter):

        topic_nodes:Dict[str, TextNode] = {}

        for node in nodes:

            chunk_id = node.node_id
            
            data = node.metadata.get(TOPICS_KEY, [])
            
            if not data:
                continue

            topics = TopicCollection.model_validate(data)

            source_info = node.relationships[NodeRelationship.SOURCE]
            source_id = source_info.node_id

            for topic in topics.topics:

                if filter.ignore_topic(topic.value):
                    continue
                
                topic_id =  self.id_generator.create_node_id('topic', source_id, topic.value) # topic identity defined by source, not chunk, so that we can connect same topic to multiple chunks in scope of single source

                if topic_id not in topic_nodes:
                    
                    metadata = {
                        'source': {
                            'sourceId': source_id
                        },
                        'topic': Topic(topicId=topic_id, value=topic.value).model_dump(exclude_none=True),
                        'statements': [],
                        INDEX_KEY: {
                            'index': 'topic',
                            'key': self._clean_id(topic_id)
                        }
                    }

                    if source_info.metadata:
                        metadata['source']['metadata'] = source_info.metadata

                    topic_node = TextNode(
                        id_ = topic_id,
                        text = topic.value,
                        metadata = metadata,
                        excluded_embed_metadata_keys = [INDEX_KEY, 'topic'],
                        excluded_llm_metadata_keys = [INDEX_KEY, 'topic']
                    )

                    topic_nodes[topic_id] = topic_node

                topic_node = topic_nodes[topic_id]
                
                topic_node = self._add_chunk_id(topic_node, chunk_id)
                topic_node = self._add_statements(topic_node, topic.statements, filter)
            
                topic_nodes[topic_id] = topic_node

        return list(topic_nodes.values())
