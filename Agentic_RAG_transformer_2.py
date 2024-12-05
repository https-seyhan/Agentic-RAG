import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
import faiss
from datasets import load_dataset

# Load datasets
dataset = load_dataset('wikipedia', '20200501.en', split='train[:1%]')  # Using a small subset

# Preprocess data
def preprocess_data(data):
    # Tokenize the text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # BERT Model
    encoded_input = tokenizer(data['text'], padding=True, truncation=True, return_tensors='pt')
    return encoded_input

# Create embeddings
def create_embeddings(encoded_input):
    model = BertModel.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        embeddings = model(**encoded_input).last_hidden_state[:, 0, :].numpy()  # Use the CLS token representation
    return embeddings

# Index the embeddings with FAISS
def index_embeddings(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Search the index
def search_index(index, query_embedding, k=5):
    D, I = index.search(query_embedding, k)
    return I

# Generate response
def generate_response(query, retrieved_docs):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2') #GPT2 model
    
    # Concatenate query and retrieved documents
    input_text = query + ' '.join(retrieved_docs)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate response
    output_ids = model.generate(input_ids, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Main RAG pipeline
def rag_pipeline(query, dataset, index, k=5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    query_encoded = tokenizer(query, return_tensors='pt')
    query_embedding = create_embeddings(query_encoded)
    
    # Retrieve documents
    top_k_indices = search_index(index, query_embedding.numpy(), k=k)
    retrieved_docs = [dataset[i]['text'] for i in top_k_indices[0]]
    
    # Generate response
    response = generate_response(query, retrieved_docs)
    return response

# Preprocess and index the dataset
encoded_input = preprocess_data(dataset)
embeddings = create_embeddings(encoded_input)
index = index_embeddings(embeddings)

# Example query
query = "What is the capital of France?"
response = rag_pipeline(query, dataset, index)
print(response)
