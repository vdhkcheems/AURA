import google.generativeai as genai
from query_rag import retrieve_top_k
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()
api_key = st.secrets["GEMINI_API_KEY"]
os.environ["GEMINI_API_KEY"] = api_key

# Papers list (can be dynamically generated later)
papers = ["Attention Is All You Need"]
papers_str = ", ".join(papers)

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# --- 🧠 Query Classifier ---
def classify_query(query, model):
    system_prompt = f"""
You are a classifier. Determine if the user's query requires retrieval from the following research papers.
[papers]
{papers_str}

Only respond with "RAG" if grounding is needed, or "CHAT" if it's general and not related to the papers in any way.
"""
    response = model.generate_content([system_prompt, query])
    return response.text.strip().upper()


# --- 🧠 Answer Generator ---
def build_heading(meta):
    parts = [meta.get("section"), meta.get("subsection"), meta.get("subsubsection")]
    return " > ".join([p for p in parts if p])

def generate_answer(query, top_chunks):
    # Combine top-k chunks into a single context string
    context = "\n\n".join([
        f"Paper Title: {chunk.get('paper_title')}\n"
        f"Heading: {build_heading(chunk)}\n"
        f"Authors: {', '.join(chunk.get('authors', []))}\n"
        f"Organization: {chunk.get('organization', 'N/A')}\n"
        f"Text: {chunk['text'][:2000]}{'...' if len(chunk['text']) > 2000 else ''}"
        for chunk in top_chunks
    ])

    prompt = f"""
You are a research paper Q&A system. Based **only** on the following context from relevant papers, answer the user's question clearly and in detail. Do not use any external knowledge.

If the query is unrelated to the context, respond with:
**"The context does not provide information related to this question."**

At the end of your response, always include a summary of **which papers** (title + authors) and **which sections** (heading) you referred to. This information is provided in each context block.

[Context]
{context}

[User Query]
{query}
"""
    response = model.generate_content(prompt)
    return response.text

# --- 🧪 Example ---
if __name__ == "__main__":
    query = input("Ask a question: ")

    # Classify
    query_type = classify_query(query, model)
    if query_type == "CHAT":
        print("🔵 General Chat Query — no retrieval needed.")
        # You could call Gemini again with just the query if needed
    else:
        print("🟢 RAG Query — retrieving relevant chunks...")
        top_chunks = retrieve_top_k(query, k=5)
        answer = generate_answer(query, top_chunks)
        print("\n🔍 Answer:\n", answer)
