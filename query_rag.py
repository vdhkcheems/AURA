import google.generativeai as genai
from retriever import load_and_split_docs, build_faiss_index, load_faiss_index
from dotenv import load_dotenv
import os
from tools import get_definition
import json
import streamlit as st

load_dotenv()
api_key = st.secrets["GEMINI_API_KEY"]
os.environ["GEMINI_API_KEY"] = api_key  

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

index_filename = "faiss_index.index"

doc_titles = ['Attention is all you need', 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', 'Improving Language Understanding by Generative Pre-Training']

import json

def classify_query_type(query: str, doc_titles: list[str]) -> str:
    """
    Classify a query as 'definition', 'calculation', or 'rag', considering document context.
    """
    title_context = ", ".join(doc_titles)
    routing_prompt = f"""
You are a smart routing assistant.
Given a user question and the titles of available research papers, classify the question into one of these categories:
1. definition — if the user wants the meaning of a term **not related to the research papers**.
2. calculation — if the user wants to calculate something (math-related).
3. rag — if the question relates to the research papers or needs in-depth understanding.

Available papers:
{title_context}

Respond ONLY in this JSON format:
{{
  "category": "..."
}}

Question: "{query}"
"""

    response = model.generate_content(routing_prompt)
    cleaned_response = response.text.strip('`').strip()
    if cleaned_response.startswith("json"):
        cleaned_response = cleaned_response[4:].strip()

    try:
        parsed = json.loads(cleaned_response)
        return parsed.get("category", "rag").strip()
    except Exception:
        return "rag"


# --- Function 2: extract_definition_target ---
def extract_definition_target(query: str) -> str:
    """
    Extract the target term to be defined from a definition-type query.
    """
    prompt = f"""
You are a definition extractor.
From the following question, extract the exact term the user wants to define.
Respond ONLY in this JSON format:
{{
  "target": "..."
}}

Question: "{query}"
"""
    response = model.generate_content(prompt)
    cleaned_response = response.text.strip('`').strip()
    if cleaned_response.startswith("json"):
        cleaned_response = cleaned_response[4:].strip()

    try:
        parsed = json.loads(cleaned_response)
        return parsed.get("target", "").strip()
    except Exception:
        return ""


# --- Function 3: extract_calculation_expression ---
def extract_calculation_expression(query: str) -> str:
    """
    Convert a natural language math query into a Python-evaluable expression.
    """
    prompt = f"""
You are a code converter.
Convert the following math problem into a single-line Python-evaluable expression.
Use '**' for powers (e.g., 2^5+3^2 => 2**5+3**2), and only return valid Python.
Respond ONLY in this JSON format:
{{
  "expression": "..."
}}

Question: "{query}"
"""
    response = model.generate_content(prompt)
    cleaned_response = response.text.strip('`').strip()
    if cleaned_response.startswith("json"):
        cleaned_response = cleaned_response[4:].strip()

    try:
        parsed = json.loads(cleaned_response)
        return parsed.get("expression", "").strip()
    except Exception:
        return ""


# Optional: wrapper function if needed for legacy compatibility
def classify_query(query: str, doc_titles: list[str]):
    category = classify_query_type(query, doc_titles)
    if category == "definition":
        return category, extract_definition_target(query), None
    elif category == "calculation":
        return category, query, extract_calculation_expression(query)
    else:
        return category, None, None



def rag_query(vectordb, query):
    # Retrieving top 3 chunks
    docs = vectordb.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""You are an academic assistant helping explain AI research papers.
Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    # Generate the answer using Gemini
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    try:
        vectordb = load_faiss_index(index_filename)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        chunks = load_and_split_docs()
        vectordb = build_faiss_index(chunks)

    print("Welcome to the Research Paper Q&A Assistant!")
    print("Ask a question (or type 'exit' to quit):")

    while True:
        query = input(">> ").strip()
        if query.lower() == "exit":
            break

        category, target, program_expr = classify_query(query, doc_titles)
        print(f"[Agent] Category: {category} | Target: {target}")

        if category == "definition":
            print("[Agent] Routing to Dictionary Tool")
            print(get_definition(target))

        elif category == "calculation":
            print("[Agent] Routing to Calculator")
            try:
                result = eval(program_expr)
                print(f"{program_expr} :", result)
            except Exception as e:
                print("Could not compute the result:", e)


        else:
            print("[Agent] Routing to RAG Pipeline")
            answer = rag_query(vectordb, query)
            print("Answer:", answer)

        print("-" * 50)