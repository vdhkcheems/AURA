import streamlit as st
from query_rag import retrieve_top_k
from generation import generate_answer, classify_query, model

papers=['attention_is_all_you_need']
papers = ", ".join(papers)

st.set_page_config(page_title="Research Paper Q&A", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📄 Research Paper Q&A Chatbot")

# Chat display
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"**🧑‍💻 You:** {user_msg}")
    st.markdown(f"**🤖 Gemini:** {bot_msg}")

# Input box – auto-submits on Enter
user_input = st.text_input("Ask a question about the research papers:", key="input" + str(st.session_state.chat_history[-1][0] if st.session_state.chat_history else ''))

if user_input.strip():  # Triggers when user presses Enter
    # Step 1: Classify the query
    query_type = classify_query(user_input, model)

    # Step 2: Maintain conversation history
    history_text = "\n\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in st.session_state.chat_history]
    )

    # Step 3: Build prompt based on query type
    if query_type == "RAG":
        top_chunks = retrieve_top_k(user_input, k=5)
        context_text = "".join(
            [f"Title: {c['title']}\nHeading: {c['heading']}\nText: {c['text']}\n\n" for c in top_chunks]
        )
        prompt = f"""
        You are a research paper Q&A assistant. Answer based ONLY on the following [Context] and [Previous conversation]. also begin your answer with 'RAG Route'. At the end of your response tell which papers you took it from and which sections you referred, it has been provided to you under title and heading sections.
        Give answer clearly and in detail.

        [Previous conversation]
        {history_text if history_text else "None"}

        [Context]
        {context_text}

        [User query]
        {user_input}
        """
    else:
        prompt = f"""
        You are a helpful research assistant. Continue the conversation naturally using only the [Previous conversation] and [User query]. also begin your answer with 'Normal route'

        [Previous conversation]
        {history_text if history_text else "None"}

        [User query]
        {user_input}
        """

    # Step 4: Generate response safely
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"⚠️ Sorry, an error occurred: {str(e)}"

    # Step 5: Update chat history and clear input
    st.session_state.chat_history.append((user_input, answer))
    st.rerun()

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
