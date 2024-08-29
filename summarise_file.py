#pip install llama-index-llms-openai
#pip install llama-index-embeddings-openai
#pip install llama-index-graph-stores-nebula
#pip install llama-index-llms-azure-openai


import browserbase
import llama_index
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.tools import QueryEngineTool

from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

import openai

openai.api_key = ""

chunk_data = SentenceSplitter(chunk_size =1024)


datasets = SimpleDirectoryReader(input_files = ["/home/saul/Desktop/agentic_rag/basics/lora.pdf"]).load_data()

nodes = chunk_data.get_nodes_from_documents(datasets)

node_metadata = nodes[0].get_content(metadata_mode = True)
print(str(node_metadata))

print(len(nodes))
#print(datasets)


# Create Model

Settings.llm = OpenAI(model = "gpt-3.5-turbo")
Settings.embedding = OpenAIEmbedding(model = "text-embedding-ada-002")

#Create Indexes

summary_index = SummaryIndex(nodes = nodes)
#vector_index = VectorStoreIndex(nodes = nodes)

# Creating Query Engines

summary_query_engine = summary_index.as_query_engine(
	response_model = "tree_summary",
	use_async = True
)

#vector_query_engine = vector_index.as_query_engine()

# Create Query Tool

summary_tool = QueryEngineTool(
	query_engine=summary_query_engine,
	metadata = node_metadata
	)
	
#vector_query_engine = vector_index.as_query_engine ()

# Create Router Query Engine

query_engine = RouterQueryEngine(
	selector = LLMSingleSelector.from_defaults(),
	query_engine_tools = [summary_tool],
	verbose = True
)

#response = query_engine.query("What is Equal contribution?")

#print(str(response))












