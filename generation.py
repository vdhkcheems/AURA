import google.generativeai as genai
from query_rag import retrieve_top_k
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()
# Load the API key from the environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
api_key = st.secrets["GEMINI_API_KEY"]
os.environ["GEMINI_API_KEY"] = api_key
papers=['attention is all you need']
papers = ", ".join(papers)

def classify_query(query, model):
    system_prompt =f"""
You are a classifier. Determine if the user's query requires retrieval from the following research papers.
[papers]
{papers}
Only respond with "RAG" if grounding is needed, or "CHAT" if it's general not reated to the papers in any way.
"""
    response = model.generate_content(
        [system_prompt, query]
    )
    return response.text.strip().upper()


# Make sure it's actually loaded
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in the environment.")

# Configure Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Function to generate answer using Gemini
def generate_answer(query, top_chunks):
    # Combine top-k chunks into a single context
    context = "\n\n".join([
        f"Paper Title: {chunk.get('paper_title')}\n"
        f"Heading: {chunk.get('heading')}\n"
        f"Authors: {chunk.get('authors')}\n"
        f"Organization: {chunk.get('organization')}\n"
        f"Text: {chunk['text']}..."
        for chunk in top_chunks
    ])

    # Send the query and context to Gemini Pro
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

    # Request answer from Gemini
    response = model.generate_content(prompt)
    return response.text

# Example usage
if __name__ == "__main__":
    query = input("Ask a question: ")
    top_chunks = retrieve_top_k(query, k=5)
    
    # Get the answer from Gemini
    answer = generate_answer(query, top_chunks)
    
    print("\n🔍 Answer: \n", answer)
