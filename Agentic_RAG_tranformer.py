import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
# Step 1: Set up the retrieval system

# Load pre-trained model and tokenizer
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to encode text into embeddings
def encode_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Create FAISS index
index = faiss.IndexFlatL2(768)  # 768 is the dimensionality of the embeddings

# Add documents to the index (example documents)
documents = ["This is a document about AI.", "Another document related to machine learning."]
embeddings = encode_text(documents).numpy()
index.add(embeddings)

# Step 2: Create the generative model

# Load generative model and tokenizer
gen_model_name = "gpt-3.5-turbo"  # Replace with the model you want to use
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)

# Step 3: Implement the RAG mechanism

def retrieve_and_generate(query, num_docs=2):
    # Step 1: Retrieve relevant documents
    query_embedding = encode_text([query]).numpy()
    distances, indices = index.search(query_embedding, num_docs)
    
    retrieved_docs = [documents[idx] for idx in indices[0]]

    # Step 2: Generate a response using the retrieved documents
    context = " ".join(retrieved_docs) + " " + query
    inputs = gen_tokenizer.encode(context, return_tensors='pt')
    outputs = gen_model.generate(inputs, max_length=100, num_return_sequences=1)
    generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# Example usage
query = "Tell me about AI advancements."
response = retrieve_and_generate(query)
print(response)


# Step 4: Create the agentic framework
class RAGAgent:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def act_on_query(self, query):
        response = self.retriever(query)
        self.take_action(response)
    
    def take_action(self, response):
        # Example action based on response content
        if "AI advancements" in response:
            print("Initiating research update...")
        else:
            print("Logging response:", response)

# Instantiate the agent
agent = RAGAgent(retrieve_and_generate, print)

# Example usage
agent.act_on_query("Tell me about AI advancements.")

