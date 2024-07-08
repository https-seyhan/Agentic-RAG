from haystack import Finder
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline

#pip install farm-haystack[all]
#pip install transformers

r#Ceating a Question Answering (QA) system using Retrieval-Augmented Generation (RAG) in Python involves several components, including a retriever model to find relevant documents 
#and a generator model to produce answers based on those documents. Agentic libraries and frameworks can help in orchestrating these components.

# Step 1: Initialize the Document Store
document_store = ElasticsearchDocumentStore(
    host="localhost",
    username="",
    password="",
    index="document"
)

# Step 2: Write documents to the Document Store
documents = [
    {"content": "Python is a programming language that lets you work quickly and integrate systems more effectively."},
    {"content": "Python is powerful... and fast; plays well with others; runs everywhere; is friendly & easy to learn; is Open."},
    {"content": "The Python Package Index (PyPI) is a repository of software for the Python programming language."},
]

document_store.write_documents(documents)

# Step 3: Initialize the Retriever
retriever = BM25Retriever(document_store=document_store)

# Step 4: Initialize the Reader
reader = TransformersReader(
    model_name_or_path="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

# Step 5: Create a Pipeline
pipe = ExtractiveQAPipeline(reader, retriever)

# Step 6: Ask Questions
questions = [
    "What is Python?",
    "What is PyPI?"
]

for question in questions:
    prediction = pipe.run(
        query=question,
        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )
    print(f"Question: {question}")
    for answer in prediction['answers']:
        print(f"Answer: {answer.answer} (Score: {answer.score})")
