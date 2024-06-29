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


def get_index(index_name, doc_file_path):
  index = None
  if not os.path.exists(f"{PERSIST_INDEX_DIR}{index_name}/"):
    # Load the documents
    documents = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Store the index to disk
    index.storage_context.persist(f"{PERSIST_INDEX_DIR}{index_name}/")
  else: # Load index from disk
    storage_context = StorageContext.from_defaults(persist_dir=f"{PERSIST_INDEX_DIR}{index_name}/")
    index = load_index_from_storage(storage_context)

  return index


# Setup Uber and Lyft Vector Indices
uber_index = get_index("uber_10k",f"/{data_dir}/RAG/data/10k/uber_10k_2023.pdf")
lyft_index = get_index("lyft_10k",f"/{data_dir}/RAG/data/10k/lyft_10k_2023.pdf")
expedia_index = get_index("expedia_10k", f"/{data_dir}/RAG/data/10k/expedia_10k_2023.pdf")
booking_index = get_index("booking_10k", f"/{data_dir}/RAG/data/10k/booking_10k_2023.pdf")
