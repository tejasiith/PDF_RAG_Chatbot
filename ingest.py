from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
INDEX_DIR = "faiss_index"

def ingest():
    documents = []

    # Load all PDFs in data folder
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            documents.extend(loader.load())

    if not documents:
        print("⚠️ No documents found in data/. Make sure your PDF is there.")
        return

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save FAISS index locally
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    vectorstore.save_local(INDEX_DIR)

    print("✅ FAISS index created successfully!")

if __name__ == "__main__":
    ingest()