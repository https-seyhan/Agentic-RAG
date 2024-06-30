import random

class Agent:
    def __init__(self, strategy):
        self.strategy = strategy

    def retrieve(self, knowledge_base, query):
        return self.strategy(knowledge_base, query)

def keyword_strategy(knowledge_base, query):
    # Simple keyword matching strategy
    return [doc for doc in knowledge_base if query.lower() in doc.lower()]

def random_strategy(knowledge_base, query):
    # Randomly select documents
    return random.sample(knowledge_base, min(len(knowledge_base), 3))

class KnowledgeBase:
    def __init__(self, documents):
        self.documents = documents

    def get_documents(self):
        return self.documents

class RAGSystem:
    def __init__(self, agents, knowledge_base):
        self.agents = agents
        self.knowledge_base = knowledge_base

    def generate_response(self, query):
        all_retrieved_docs = []
        for agent in self.agents:
            retrieved_docs = agent.retrieve(self.knowledge_base.get_documents(), query)
            all_retrieved_docs.extend(retrieved_docs)
        
        # Combine retrieved documents to generate a response
        response = self.generate_from_docs(all_retrieved_docs)
        return response

    def generate_from_docs(self, documents):
        # Simple response generation by combining documents
        return " ".join(documents)

if __name__ == "__main__":
    # Sample knowledge base documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The only thing we have to fear is fear itself."
    ]

    knowledge_base = KnowledgeBase(documents)

    # Define agents with different strategies
    agents = [Agent(keyword_strategy), Agent(random_strategy)]

    # Create RAG system
    rag_system = RAGSystem(agents, knowledge_base)

    # Sample query
    query = "fear"

    # Generate response
    response = rag_system.generate_response(query)
    print(f"Query: {query}\nResponse: {response}")
