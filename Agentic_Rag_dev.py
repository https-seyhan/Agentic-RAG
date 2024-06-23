
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

def retrieve_documents(query, corpus, top_n=2):
    vectorizer = TfidfVectorizer()
    corpus_vectors = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, corpus_vectors).flatten()
    relevant_indices = similarities.argsort()[-top_n:][::-1]
    relevant_docs = [corpus[idx] for idx in relevant_indices]
    
    return relevant_docs


