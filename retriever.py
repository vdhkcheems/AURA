import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
import faiss

def load_and_split_docs(data_dir="data"):
    all_chunks = []
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, filename))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
    return all_chunks

def build_faiss_index(chunks, index_filename="faiss_index.index"):
    # Using SentenceTransformer for embeddings
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectordb = FAISS.from_documents(chunks, embedding_model)
    
    vectordb.save_local(index_filename)
    return vectordb

def load_faiss_index(index_filename="faiss_index.index"):
    # Load the saved FAISS index from the file
    if os.path.exists(index_filename):
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.load_local(index_filename, embedding_model)
        return vectordb
    else:
        raise FileNotFoundError(f"FAISS index file '{index_filename}' not found.")