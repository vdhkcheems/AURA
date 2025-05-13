import os
import streamlit as st
from query_rag import classify_query_type, extract_definition_target, extract_calculation_expression, classify_query, rag_query
from retriever import load_and_split_docs, build_faiss_index, load_faiss_index
from langchain.embeddings import SentenceTransformerEmbeddings
import faiss
from langchain.vectorstores import FAISS
from tools import get_definition

index_filename = "faiss_index.index"

def load_or_create_vector_db():
    # Check if the FAISS index file exists
    if os.path.exists(index_filename):
        try:
            embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = FAISS.load_local(index_filename, embedding_model, allow_dangerous_deserialization=True)
            return vectordb
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
            return None
    else:
        # If the index doesn't exist, create it
        st.write("FAISS index not found. Building the index now...")
        try:
            chunks = load_and_split_docs()
            vectordb = build_faiss_index(chunks, index_filename)
            st.write("FAISS index created successfully.")
            return vectordb
        except Exception as e:
            st.error(f"Error while creating FAISS index: {e}")
            return None


st.title("Agentic AI - for research papers and definitions or calculations")

description = """
Welcome to the Agnetic AI by Antriksh Arya. Ask a question related to the provided papers.
- Attention Is All You Need
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Improving Language Understanding by Generative Pre-Training

Or you can ask for the definition of any term, which will be answered through a dictionary api like
- Define calculator
- What do you mean by computer?

You can also provide simple calulations or word problems like
- Calculate 4+4
- If A has 4 mangoes and B takes from him 2 gut C gives him 1, how many mangoes does A have.
"""
st.write(description)


query = st.text_input("Ask a question:")


if query:
    
    vectordb = load_or_create_vector_db()
    
    if vectordb:
        # Classify the query and get the category and target
        category, target, program_expr = classify_query(query, doc_titles=['Attention is all you need', 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', 'Improving Language Understanding by Generative Pre-Training'])
        
        if category == "definition":
            # Handle definition queries
            definition = get_definition(target)
            response = f"""Definition route, Using the dictionary API.
            {definition}"""
            st.write(response)
        
        elif category == "calculation":
            # Handle calculation queries
            try:
                result = eval(program_expr)
                safe_expr = program_expr.replace("*", "\\*")
                response = f"Calculation route.\nThe result of {safe_expr} is {result}"
                st.write(response)
            except Exception as e:
                st.error(f"Error calculating the expression: {e}")
        
        else:
            # If it's a regular RAG query
            response = rag_query(vectordb, query)
            result = f"""RAG route, reffering the provided papers.
            Answer: {response}"""
            st.write(result)
