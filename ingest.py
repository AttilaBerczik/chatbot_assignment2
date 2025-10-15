import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Specify the URL to crawl
url = "https://en.wikipedia.org/wiki/Norway"

# Load the document from the web
loader = WebBaseLoader(url)
documents = loader.load()

# Split the documents into smaller, more manageable chunks
text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store from the documents
db = FAISS.from_documents(texts, embeddings)

# Save the vector store to a local file
db.save_local("faiss_index")

print("Vector store created and saved to faiss_index.")
