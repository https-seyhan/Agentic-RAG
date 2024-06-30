
# Document Corpus
document_corpus = [
"total loc: 240742845",
"Validation loc:  2520910",
"Test loc:  2897766",
"Encoding loc:  23581583",
"1% training set loc:  4242264",
"Full training set loc:  211742586"
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
    if len(query.split()) > 3 or "what" in query.lower():
        retrieved_docs = retrieve_documents(query, corpus)
        context = " ".join(retrieved_docs)
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    else:
        prompt = query
    
    return generate_text(prompt)

query = "What is encoding loc"
response = agentic_rag(query, document_corpus)
print(response)
