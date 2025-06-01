import re
import google.generativeai as genai
from query_rag import retrieve_top_k, build_heading

def extract_math_equations(text):
    """Extract LaTeX math equations from text enclosed in triple backticks"""
    pattern = r'```\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches if match.strip()]

def get_available_papers():
    """Get list of available papers dynamically"""
    # For now, we only have one paper, but this can be extended
    return ["Attention Is All You Need"]

def build_context_text(chunks):
    """Build context text from retrieved chunks"""
    context_parts = []
    for chunk in chunks:
        context_part = (
            f"Paper Title: {chunk.get('paper_title', 'N/A')}\n"
            f"Heading: {chunk.get('heading', 'N/A')}\n"
            f"Authors: {', '.join(chunk.get('authors', []))}\n"
            f"Organization: {chunk.get('organization', 'N/A')}\n"
            f"Year: {chunk.get('year', 'N/A')}\n"
            f"Text: {chunk['text']}"
        )
        context_parts.append(context_part)
    return "\n\n".join(context_parts)

def generate_rag_response(user_input, history_text, model):
    """Generate response for RAG queries"""
    # Retrieve relevant chunks
    top_chunks = retrieve_top_k(user_input, k=5)
    
    # Build context
    context_text = build_context_text(top_chunks)
    
    # Create RAG prompt
    prompt = f"""
You are AURA - Artificial Understanding of Research Articles, an Agentic AI and a smart research paper Q&A assistant.

Begin your response with 'RAG Route:' and answer based **only** on the [Context] and [Previous conversation]. 

IMPORTANT GUIDELINES:
1. If the question is not clearly answered in the context, say: "The context does not provide enough information to answer this question."
2. If you reference mathematical equations in your answer, include them naturally in your response
3. Always end your response with a section titled "📚 **Sources Referenced:**" listing which papers and sections were used
4. Be precise and detailed in your explanations
5. Maintain conversation flow by referencing previous context when relevant

[Previous conversation]
{history_text if history_text else "None"}

[Context]
{context_text}

[User query]
{user_input}
"""
    
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Extract math equations from the retrieved chunks
        all_math = []
        for chunk in top_chunks:
            math_equations = extract_math_equations(chunk['text'])
            all_math.extend(math_equations)
        
        return {
            'text': answer,
            'math_equations': list(set(all_math)),  # Remove duplicates
            'chunks_used': len(top_chunks)
        }
        
    except Exception as e:
        return {
            'text': f"⚠️ Sorry, an error occurred while processing your query: {str(e)}",
            'math_equations': [],
            'chunks_used': 0
        }

def generate_chat_response(user_input, history_text, model):
    """Generate response for general chat queries"""
    
    prompt = f"""
You are AURA - Artificial Understanding of Research Articles, an Agentic AI and a smart research paper Q&A assistant.

Begin your response with 'Normal Route:' and continue the conversation naturally using only the [Previous conversation] and [User query].

You are knowledgeable about research papers and AI/ML topics, but for this query, you're responding as a general conversational AI since it doesn't require specific paper retrieval.

[Previous conversation]
{history_text if history_text else "None"}

[User query]
{user_input}
"""
    
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        return {
            'text': answer,
            'math_equations': [],
            'chunks_used': 0
        }
        
    except Exception as e:
        return {
            'text': f"⚠️ Sorry, an error occurred while processing your query: {str(e)}",
            'math_equations': [],
            'chunks_used': 0
        }

def format_math_for_display(math_equations):
    """Format math equations for Streamlit display"""
    if not math_equations:
        return ""
    
    formatted_math = "### 🧮 **Mathematical References:**\n\n"
    for i, equation in enumerate(math_equations, 1):
        formatted_math += f"**Equation {i}:**\n"
        formatted_math += f"$$\n{equation}\n$$\n\n"
    
    return formatted_math

import re

def convert_backticks_to_latex(text: str) -> str:
    """
    Converts inline `...` LaTeX to $...$ for MathJax rendering.
    """
    return re.sub(r"```([^`]+?)```", r"$$\1$$", text)