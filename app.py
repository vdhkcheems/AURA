import streamlit as st
from generation import classify_query, model
from utils import (
    generate_rag_response, 
    generate_chat_response, 
    get_available_papers,
    format_math_for_display,
    convert_backticks_to_latex
)

# Setup Streamlit config
st.set_page_config(page_title="🧠 AURA - Research Paper Q&A", layout="wide", page_icon="📄")

# Inject MathJax to support inline LaTeX rendering
st.markdown("""
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$']],
  },
  svg: { fontCache: 'global' }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
""", unsafe_allow_html=True)


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
    max-width: 85%;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    word-wrap: break-word;
}

.chat-bubble.user {
    background-color: #4a90e2;
    margin-left: auto;
    text-align: right;
}

.chat-bubble b {
    color: #ffd700;
}

.math-section {
    background-color: #1a1a2e;
    border-left: 4px solid #4a90e2;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
}

.stats-info {
    font-size: 0.8rem;
    color: #888;
    text-align: right;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header
with st.container():
    st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <h1 style='color: #f1f1f1;'>📄 AURA - Artificial Understanding of Research Articles</h1>
            <p style='font-size: 1.1rem; color: #d1d1d1; max-width: 800px; margin: auto;'>
                Welcome to <strong>AURA</strong> – your smart agentic AI for demystifying complex research papers!<br>
                I'm powered by <strong>Google Gemini</strong> and use a <strong>RAG (Retrieval-Augmented Generation)</strong> architecture.
            </p>
            <p style='font-size: 1rem; color: #b0b0b0; max-width: 700px; margin: auto;'>
                🧠 Ask me about any of the research papers I have access to, and I'll fetch the most relevant parts to answer you precisely.<br>
                💬 Ask me anything else, and I'll respond like a regular conversational AI.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Available papers section
st.markdown("---")
available_papers = get_available_papers()
papers_list = "\n".join([f"- *{paper}*" for paper in available_papers])
st.markdown(f"### 📚 Available Paper(s)\n{papers_list}")

# Display chat history
st.markdown("<div class='chat-section'>", unsafe_allow_html=True)
for chat_item in st.session_state.chat_history:
    user_msg = chat_item['user_message']
    bot_response = chat_item['bot_response']
    math_equations = chat_item.get('math_equations', [])
    chunks_used = chat_item.get('chunks_used', 0)
    
    # User message
    st.markdown(f"<div class='chat-bubble user'>🧑‍💻 <b>You:</b><br>{user_msg}</div>", unsafe_allow_html=True)
    
    # Bot response
    processed_bot_response = convert_backticks_to_latex(bot_response)

# Bot response with MathJax-friendly inline LaTeX
    st.markdown(
        f"<div class='chat-bubble'>🤖 <b>AURA:</b><br>{processed_bot_response}</div>", 
        unsafe_allow_html=True
)
    
    # Stats info
    if chunks_used > 0:
        st.markdown(f"<div class='stats-info'>📊 Retrieved {chunks_used} relevant chunks</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    
    # Math equations if any
    # if math_equations:
    #     st.markdown("<div class='math-section'>", unsafe_allow_html=True)
    #     for eq in math_equations:
    #         st.latex(eq)
    #     st.markdown("</div>", unsafe_allow_html=True)


st.markdown("</div>", unsafe_allow_html=True)

# Input section
st.markdown("---")
last_key = (
    st.session_state.chat_history[-1]['user_message']
    if st.session_state.chat_history else "initial_key"
)

user_input = st.text_input(
    "💬 Ask a question about the research papers:", 
    key=last_key,
    placeholder="e.g., What is the attention mechanism in transformers?"
)

# Handle input
if user_input and user_input.strip():
    with st.spinner("🤔 Processing your question..."):
        # Classify query type
        query_type = classify_query(user_input, model)
        
        # Build conversation history
        history_text = "\n\n".join([
            f"User: {item['user_message']}\nAssistant: {item['bot_response']}" 
            for item in st.session_state.chat_history
        ])
        
        # Generate response based on query type
        if query_type == "RAG":
            response_data = generate_rag_response(user_input, history_text, model)
        else:
            response_data = generate_chat_response(user_input, history_text, model)
        
        # Add to chat history
        chat_item = {
            'user_message': user_input,
            'bot_response': response_data['text'],
            'math_equations': response_data['math_equations'],
            'chunks_used': response_data['chunks_used'],
            'query_type': query_type
        }
        
        st.session_state.chat_history.append(chat_item)

        # Clear input field for next run
        st.session_state.temp_input = ""
        st.rerun()

# Sidebar with controls
with st.sidebar:
    st.markdown("### 🛠️ Controls")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    
    total_messages = len(st.session_state.chat_history)
    rag_queries = sum(1 for item in st.session_state.chat_history if item.get('query_type') == 'RAG')
    chat_queries = total_messages - rag_queries
    
    st.metric("Total Messages", total_messages)
    st.metric("RAG Queries", rag_queries)
    st.metric("Chat Queries", chat_queries)
    
    if total_messages > 0:
        st.markdown("---")
        st.markdown("### 🔍 Recent Activity")
        recent_activity = st.session_state.chat_history[-3:]  # Last 3 messages
        for i, item in enumerate(reversed(recent_activity), 1):
            query_type_icon = "🧠" if item.get('query_type') == 'RAG' else "💬"
            st.markdown(f"{query_type_icon} **Query {i}:** {item['user_message'][:50]}...")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
    "🚀 AURA v1.0 | Powered by Google Gemini & BGE Embeddings"
    "</div>", 
    unsafe_allow_html=True
)