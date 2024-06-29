# Set up LLM

import os
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

#from dotenv import load_dotenv
# This is needed for jupyter notebook to do asynchronous rendering
nest_asyncio.apply()
# Ensure you replace this before executing this cell
#load_dotenv("<path to your environment file where OpenAI API key is stored as OPENAI_API_KEY=sk-#######>.env")

# Setup OpenAI Model and Embeddings used for indexing the documents
Settings.llm = OpenAI(model='gpt-4-0125-preview', temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.chunk_size = 1024

PERSIST_INDEX_DIR = '/home/saul/Desktop/agentic_rag/'

