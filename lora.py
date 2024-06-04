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



