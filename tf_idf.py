from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = ["Generative Artificial Intelligence (GenAI) systems are being increasingly deployed across all parts ofindustry and research settings", "Prompting is the process of providing a prompt to a GenAI, which then generates aresponse"]

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (terms)
terms = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to an array
tfidf_array = tfidf_matrix.toarray()

# Display the TF-IDF representation
for doc_idx, doc in enumerate(tfidf_array):
    print(f"Document {doc_idx+1}:")
    for term_idx, tfidf_value in enumerate(doc):
        print(f"  Term: {terms[term_idx]}, TF-IDF: {tfidf_value:.4f}")
