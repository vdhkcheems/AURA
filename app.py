import streamlit as st
from query_rag import retrieve_top_k
from generation import generate_answer, classify_query, model

# Setup Streamlit config
st.set_page_config(page_title="🧠 AURA - Research Paper Q&A", layout="wide", page_icon="📄")

# Inject custom CSS
st.markdown("""
<style>
.chat-section {
    padding: 1rem 0;
}

.chat-bubble {
    background-color: #2b313e;
    color: white;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    max-width: 80%;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
}

.chat-bubble.user {
    background-color: #4a90e2;
    margin-left: auto;
    text-align: right;
}

.chat-bubble b {
    color: #ffd700;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.container():
    st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <h1 style='color: #f1f1f1;'>📄 AURA - Artificial Understanding of Research Articles</h1>
            <p style='font-size: 1.1rem; color: #d1d1d1; max-width: 800px; margin: auto;'>
                Welcome to <strong>AURA</strong> – your smart agentic AI for demystifying complex research papers!<br>
                I’m powered by <strong>Google Gemini</strong> and use a <strong>RAG (Retrieval-Augmented Generation)</strong> architecture.
            </p>
            <p style='font-size: 1rem; color: #b0b0b0; max-width: 700px; margin: auto;'>
                🧠 Ask me about any of the research papers I have access to, and I’ll fetch the most relevant parts to answer you precisely.<br>
                🐶 Ask me anything else (even about dogs!), and I’ll respond like a regular Gemini model.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Use normal Streamlit markdown for the rest
st.markdown("---")
st.markdown("### 📚 Available Paper(s)\n- *Attention Is All You Need*")

# Display chat history
st.markdown("<div class='chat-section'>", unsafe_allow_html=True)
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"<div class='chat-bubble user'>🧑‍💻 <b>You:</b><br>{user_msg}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble'>🤖 <b>AURA:</b><br>{bot_msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input box
user_input = st.text_input("💬 Ask a question about the research papers:", key="input" + str(st.session_state.chat_history[-1][0] if st.session_state.chat_history else ''))

# Handle input
if user_input.strip():
    query_type = classify_query(user_input, model)

    history_text = "\n\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in st.session_state.chat_history]
    )

    if query_type == "RAG":
        top_chunks = retrieve_top_k(user_input, k=5)
        context_text = "\n\n".join([
            f"Paper Title: {c.get('paper_title')}\n"
            f"Heading: {c.get('heading')}\n"
            f"Authors: {c.get('authors')}\n"
            f"Organization: {c.get('organization')}\n"
            f"Text: {c['text']}..."
            for c in top_chunks
        ])
        prompt = f"""
        You are AURA - Artificial Undertsanding of Research Articles, an Agentic AI and a smart research paper Q&A assistant.

        Begin your response with 'RAG Route:' and answer based **only** on the [Context] and [Previous conversation]. If the question is not clearly answered in the context, say:
        "The context does not provide enough information to answer this question."

        Always end your response with a list of which papers and which sections (heading) were used. If information repeated in multiple times then mention only once in the top relevant one.

        [Previous conversation]
        {history_text if history_text else "None"}

        [Context]
        {context_text}

        [User query]
        {user_input}
        """
    else:
        prompt = f"""
        You are AURA - Artificial Undertsanding of Research Articles, an Agentic AI and a smart research paper Q&A assistant.

        Begin your response with 'Normal Route:' and continue the conversation naturally using only the [Previous conversation] and [User query].

        [Previous conversation]
        {history_text if history_text else "None"}

        [User query]
        {user_input}
        """

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"⚠️ Sorry, an error occurred: {str(e)}"

    st.session_state.chat_history.append((user_input, answer))
    st.rerun()

# Clear chat
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
