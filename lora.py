#pip install llama-index-llms-openai
#pip install llama-index-embeddings-openai
#pip install llama-index-graph-stores-nebula
#pip install llama-index-llms-azure-openai

import browserbase
import llama_index
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
chunk_data = SentenceSplitter(chunk_size =1024)


datasets = SimpleDirectoryReader(input_files = ["/home/saul/Desktop/agentic_rag/basics/lora.pdf"]).load_data()

nodes = chunk_data.get_nodes_from_documents(datasets)

node_metadata = nodes[0].get_content(metadata_mode = True)
print(str(node_metadata))

len(nodes)
#print(datasets)




