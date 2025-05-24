# 📄 Agentic RAG using Agno and LangChain

This project is a **Document Question & Answering Web Application** built using **Streamlit**, **LangChain**, and **Agno SDK**. It allows users to upload documents and ask natural language questions. The app uses a **Retrieval-Augmented Generation (RAG)** architecture enhanced with agentic reasoning to provide accurate, context-aware answers using both document content and real-time web data.

---
## 📸 Screenshot

Here’s how the running application looks:

<p align="center">
  <img src="assets/screenshot.png" width="600"/>
</p>


## 🧠 Key Features

- 🔍 Upload and process documents (`.txt`, `.pdf`, `.docx`, `.csv`, `.ppt`, etc.)
- 💬 Ask natural language questions about uploaded documents
- 🤖 Smart RAG-powered assistant using **LangChain** + **Agno Agent**
- 🌐 Web search agent using **DuckDuckGoTools** (for out-of-context questions)
- 🔗 Website content extraction from URLs in documents
- 🧠 Local vector store created using **Chroma** for fast retrieval
- 📚 Answers with citations and references
- 💡 Intelligent agent instructions for multi-agent collaboration

---

## 🚀 Technologies Used

| Component                | Technology         |
|--------------------------|--------------------|
| Frontend UI              | Streamlit          |
| LLM Integration          | Agno SDK           |
| Document QA              | LangChain          |
| Vector Database          | Chroma             |
| Embeddings               | SentenceTransformer |
| Tools & Plugins          | DuckDuckGo, WebsiteTools |
| Styling                  | Custom HTML/CSS    |

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies

```bash 
pip install -r requirements.txt
```

### 4. Set environment variables
Create a .env file in the root directory:

```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_CHAT_MODEL=llama3-8b-8192   # or whichever Groq model you're using
```
---

## 📂 Project Structure

<pre>
├── assets/
│   └── Screenshot.png
├── uploads/           # Folder for uploaded files  
├── chroma_db/         # Local vector database directory  
├── main.py            # Main Streamlit application  
├── .env               # Environment variables  
├── requirements.txt   # Python dependencies  
└── README.md          # Project documentation  
</pre>

---

## 🧪 How It Works

- **Upload Documents:** Files are uploaded and saved to the uploads/ directory.

- **Vectorization:** Documents are split and embedded using SentenceTransformerEmbeddings, and stored in a Chroma vector store.

- **Agentic Reasoning:** The app uses Agno agents with LangChain-based retrievers to answer queries.

- **Tool Use:** If the question is unrelated to the document, it uses DuckDuckGo to search the web or fetches content from URLs in the document.

- **UI Display:** Answers, citations, and conversation history are shown with styled chat bubbles.

---

## 🤖 Agents Defined

### 1. Document Agent
- Primary agent.
- Uses LangChain retriever and Agno knowledge base.

### 2. Web Search Agent
- Uses DuckDuckGo for external info.
- Triggered when the question is unrelated to documents.

### 3. Website Agent
- Extracts content from URLs in the uploaded documents.

---

## 💬 Example Prompt Flow
**User:** What is covered in the third slide of the PowerPoint?<br>
**Assistant:** [Analyzes slide content and replies with summarized info]

**User:** Who wrote this document?<br>
**Assistant:** This question is not related to the uploaded documents. Searching the web for relevant information...

---

## 📎 **Credits**

- [Agno SDK](https://github.com/agnos-ai/agnos)
- [LangChain](https://www.langchain.com/)
- [Chroma Vector Store](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
