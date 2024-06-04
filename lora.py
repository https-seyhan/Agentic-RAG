#pip install llama-index-llms-openai
#pip install llama-index-embeddings-openai
#pip install llama-index-graph-stores-nebula
#pip install llama-index-llms-azure-openai

import browserbase
import llama_index
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

datasets = SimpleDirectoryReader(input_files = ["/home/saul/Desktop/agentic_rag/basics/lora.pdf"]).load_data()

print(datasets)




