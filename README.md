# 📄 AURA – Artificial Understanding of Research Articles

**AURA** is an intelligent AI assistant that helps users effortlessly understand complex research papers using natural language questions. It leverages **RAG (Retrieval-Augmented Generation)** architecture and is powered by **Google Gemini** to deliver highly relevant and precise answers grounded in the research context.

> Ask AURA about papers it has access to, and it will fetch and summarize the most relevant sections. Ask it anything else, and it becomes a conversational agent like any smart chatbot!

---

## 🌟 Overview

AURA (Artificial Understanding of Research Articles) is designed to make reading academic papers less daunting. It’s an **agentic, dual-behavior AI** system that:
- Uses **RAG-based document retrieval** to ground answers in the original text.
- Switches between **RAG Mode** and **Normal Mode** based on the query type.
- Offers an intuitive, chat-style interface with clean UI and structured responses.

Whether you're a researcher, student, or enthusiast, AURA can help you:
- Understand key points of a paper.
- Ask detailed questions about methodology, results, or assumptions.
- Get quick and trustworthy responses grounded in actual content.

---

## ⚙️ Features

### 🧠 Dual-Mode Intelligence
- **RAG Route**: Retrieves relevant chunks from research papers and generates grounded answers.
- **Normal Route**: Behaves like a typical Gemini-powered conversational assistant when queries aren't paper-specific.

### 🔍 Intelligent Retrieval
- Uses **FAISS** for semantic similarity search.
- Dynamically fetches top-k relevant sections from parsed papers based on user input.

### 📄 Paper Metadata Awareness
- Each RAG response includes citations of the paper and section (heading) it used for transparency.

### 🗂️ Organized Architecture
- Modular file structure separating:
  - Retrieval logic (`query_rag.py`)
  - Generation & classification logic (`generation.py`)
  - Streamlit frontend (`app.py`)

### 💬 Beautiful Chat UI
- Custom styled chat bubbles using HTML + CSS inside Streamlit.
- Distinct user and bot messages with proper alignment and colors.

---

## 🚀 How to Use

### 🔗 Use Online

Try the live demo (hosted on [Streamlit Community Cloud](https://share.streamlit.io/)):

> **[🌐 Launch AURA](https://aura-vdhkcheems.streamlit.app/)**


---

### 💻 Run Locally

#### 1. Clone the repository
```bash
git clone https://github.com/your-username/aura.git
cd aura
```

#### 2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Add your Google Gemini API credentials
Make a .env file and add the gemini key as 'GEMINI_API_KEY'

#### 5. Run
```bash
streamlit run app.py
```

---

### 🚧 Future Improvements

Here are a few planned upgrades to AURA:

- **In-app upload**: Let users upload their own PDFs and instantly ask questions about them.
- **Source highlighting**: Show exactly which chunks from the paper were used to generate the answer.
- **Conversation memory**: Persist chat history between sessions and improve context handling.
- **UI Enhancements**: Improve chat interface responsiveness and add dark/light mode toggle.
- **Model selection**: Option to switch between Gemini, GPT, or local open-source LLMs.
- **Analytics Dashboard**: Track user queries, paper usage, and model response performance.

---

### 🔗 Demo

![Screenshot_20250514_125409](https://github.com/user-attachments/assets/545a8783-baff-4fa6-bdbb-946bb8ad1b17)


![Screenshot_20250514_125437](https://github.com/user-attachments/assets/e025a650-3fe9-465f-9d2b-196a2cacc812)



### 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

