import torch
from transformers import GPTNeoForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceLLM
from langchain.agents import AgentExecutor, Tool

# Initialize Sentence Transformer model for retrieval
retriever_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') #sBERT

# Sample documents
documents = [
    {"id": 1, "text": "Paris is the capital and most populous city of France."},
    {"id": 2, "text": "Berlin is the capital of Germany."},
    {"id": 3, "text": "Madrid is the capital of Spain."}
]

# Function to retrieve relevant documents
def retrieve_documents(query, top_k=3):
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    doc_embeddings = retriever_model.encode([doc["text"] for doc in documents], convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    retrieved_docs = [documents[idx] for idx in top_results[1]]
    return retrieved_docs

# Initialize GPT-Neo model and tokenizer for generation
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Function to generate response using GPT-Neo
def generate_response(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# LangChain tool for document retrieval
class RetrievalTool(Tool):
    def run(self, input_text):
        retrieved_docs = retrieve_documents(input_text)
        context = " ".join([doc["text"] for doc in retrieved_docs])
        return context

retrieval_tool = RetrievalTool(name="retrieval", description="Document Retrieval Tool")

# LangChain LLM chain for generating responses
llm = HuggingFaceLLM(model_name=model_name, tokenizer=tokenizer)
prompt_template = PromptTemplate(template="Context: {context}\n\nQuestion: {query}\nAnswer:", input_variables=["context", "query"])
llm_chain = LLMChain(prompt_template=prompt_template, llm=llm)

# Define an agent executor
class RAGAgentExecutor(AgentExecutor):
    def __init__(self, llm_chain, retrieval_tool):
        self.llm_chain = llm_chain
        self.retrieval_tool = retrieval_tool

    def run(self, query):
        context = self.retrieval_tool.run(query)
        prompt = self.llm_chain.prompt_template.format(context=context, query=query)
        response = generate_response(prompt)
        return response

# Initialize the agent
agent = RAGAgentExecutor(llm_chain=llm_chain, retrieval_tool=retrieval_tool)

# Function to handle user queries
def agent_query(query):
    response = agent.run(query)
    return response

# Example usage
query = "What is the capital of France?"
response = agent_query(query)
print(response)
