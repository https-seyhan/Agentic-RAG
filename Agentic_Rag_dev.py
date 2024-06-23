
# Document Corpus
document_corpus = [
    "The Eiffel Tower is located in Paris, France.",
    "The Great Wall of China is a historic fortification.",
    "The Amazon Rainforest is known for its biodiversity.",
    "Mount Everest is the highest mountain in the world.",
    "The Nile River is the longest river in the world."
]

# Retrieval Mechanism
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai


def retrieve_documents(query, corpus, top_n=2):
    vectorizer = TfidfVectorizer()
    corpus_vectors = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, corpus_vectors).flatten()
    relevant_indices = similarities.argsort()[-top_n:][::-1]
    relevant_docs = [corpus[idx] for idx in relevant_indices]
    
    return relevant_docs

# Text Generation

openai.api_key = ''

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Agentic Behavior

def agentic_rag(query, corpus):
    # Decide when to retrieve based on query length or keywords
    if len(query.split()) > 3 or "where" in query.lower():
        retrieved_docs = retrieve_documents(query, corpus)
        context = " ".join(retrieved_docs)
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    else:
        prompt = query
    
    return generate_text(prompt)

query = "Where is the Eiffel Tower located?"
response = agentic_rag(query, document_corpus)
print(response)
