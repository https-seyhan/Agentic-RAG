import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import TextGenerator, Agent

from langchain.tools.retriever import create_retriever_tool


retriever_tool = create_retriever_tool(
    retriever,
    "retrieve",
    "Search and return information on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]

print('TOOLS : ', tools)

# Initialize DPR for question and context encoding
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Initialize GPT-2 model for text generation
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Sample documents to be used in retrieval
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy."
]

# Function to encode documents
def encode_documents(documents, tokenizer, encoder):
    inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
    embeddings = encoder(**inputs).pooler_output
    return embeddings

# Encode the documents
doc_embeddings = encode_documents(documents, context_tokenizer, context_encoder)

# Initialize LangChain components
#text_splitter = TextSplitter()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
text_retriever = TextRetriever(embeddings=doc_embeddings, documents=documents)
text_generator = TextGenerator(model=gpt_model, tokenizer=gpt_tokenizer)

# Agent function to handle user queries
class RAGAgent(Agent):
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def answer(self, query):
        # Step 1: Encode the query
        query_inputs = question_tokenizer(query, return_tensors="pt")
        query_embedding = question_encoder(**query_inputs).pooler_output

        # Step 2: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query_embedding)

        # Step 3: Generate response using GPT-2
        context = " ".join(retrieved_docs)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        response = self.generator.generate(prompt, max_length=150)
        return response

# Initialize the agent
agent = RAGAgent(retriever=text_retriever, generator=text_generator)

# Example usage
query = "What is the capital of France?"
response = agent.answer(query)
print(response)
