# pip install transformers sentence-transformers faiss-cpu

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import faiss

class RAGQA:
    def __init__(self, retriever_model_name='sentence-transformers/all-mpnet-base-v2', generator_model_name='facebook/bart-large-cnn'):
        # Load the retriever model
        self.retriever = SentenceTransformer(retriever_model_name)
        
        # Load the generator model
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)

        # Placeholder for the FAISS index and document store
        self.index = None
        self.documents = []

    def add_documents(self, docs):
        self.documents.extend(docs)
        embeddings = self.retriever.encode(docs, convert_to_tensor=True)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.cpu().numpy())

    def retrieve(self, query, top_k=5):
        query_embedding = self.retriever.encode(query, convert_to_tensor=True)
        distances, indices = self.index.search(query_embedding.cpu().numpy(), top_k)
        return [self.documents[idx] for idx in indices[0]]

    def generate_answer(self, question, context):
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.generator.generate(inputs['input_ids'], max_length=256, early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def answer(self, question, top_k=5):
        retrieved_docs = self.retrieve(question, top_k)
        combined_context = " ".join(retrieved_docs)
        answer = self.generate_answer(question, combined_context)
        return answer

# Example usage
if __name__ == "__main__":
    # Initialize the RAG QA system
    rag_qa = RAGQA()

    # Add documents to the document store
    documents = [
        "Python is a high-level, interpreted programming language.",
        "It was created by Guido van Rossum and first released in 1991.",
        "Python has a design philosophy that emphasizes code readability.",
        "It provides constructs that enable clear programming on both small and large scales."
    ]
    rag_qa.add_documents(documents)

    # Ask a question
    question = "Who created Python?"
    answer = rag_qa.answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


#SentenceTransformer is used to encode documents and queries into embeddings.
#FAISS is used to perform efficient similarity searches over document embeddings.
#A sequence-to-sequence model like BART is used for the generative part to generate answers based on the retrieved documents.
#The add_documents method adds documents to the system, 
#while the answer method retrieves relevant documents and generates an answer to the input question.