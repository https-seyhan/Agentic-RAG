from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Connect to Elasticsearch
es = Elasticsearch()

# Index sample documents (run only once to set up the index)
documents = [
    {"text": "Document 1 text content"},
    {"text": "Document 2 text content"},
    # Add more documents here
]

for i, doc in enumerate(documents):
    es.index(index='documents', id=i, document=doc)

# Define document retrieval function
def retrieve_documents(query, index='documents'):
    response = es.search(
        index=index,
        body={
            "query": {
                "match": {
                    "text": query
                }
            }
        }
    )
    return [hit['_source']['text'] for hit in response['hits']['hits']]

# Load the GPT-J-6B model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# Define the RAG function
def generate_response(query):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query)
    
    # Concatenate retrieved documents and the query
    context = " ".join(retrieved_docs) + " " + query
    
    # Tokenize the context
    inputs = tokenizer.encode(context, return_tensors='pt')
    
    # Generate a response
    outputs = model.generate(inputs, max_length=512, do_sample=True, top_p=0.95, top_k=60)
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the RAG system
query = "Explain the concept of retrieval-augmented generation."
response = generate_response(query)
print(response)
