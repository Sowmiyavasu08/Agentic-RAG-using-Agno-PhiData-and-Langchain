import os
import hashlib
import shutil
import streamlit as st
from dotenv import load_dotenv

# LangChain and AGNO imports
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from agno.agent import Agent
from agno.models.groq import Groq
from agno.knowledge.langchain import LangChainKnowledgeBase
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.website import WebsiteTools

# ------------------------ Configuration ------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_CHAT_MODEL")

CHROMA_DB_DIR = "./chroma_db"
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

llm_model = Groq(id=GROQ_MODEL_NAME)
embedding_function = SentenceTransformerEmbeddings()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------ Styling ------------------------

st.set_page_config(page_title="AGNO Document Q&A", layout="wide", page_icon="📄")
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .chat-bubble-user {
            background-color: #DDEEFF;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .chat-bubble-assistant {
            background-color: #E6FFE6;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .section-header {
            font-size: 20px;
            font-weight: bold;
            margin-top: 30px;
            border-bottom: 2px solid #bbb;
            padding-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------ Helper Functions ------------------------

def hash_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def save_uploaded_files(uploaded_files):
    file_hashes = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        file_hashes.append(hash_file(file_path))
    return file_hashes

def load_and_split_documents(directory: str):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter()
    return splitter.split_documents(documents)

def create_or_load_vectorstore(doc_hashes: list):
    hash_file_path = os.path.join(CHROMA_DB_DIR, "hashes.txt")
    if os.path.exists(hash_file_path):
        with open(hash_file_path, "r") as f:
            existing_hashes = f.read().splitlines()
        if set(existing_hashes) == set(doc_hashes):
            return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)

    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    documents = load_and_split_documents(UPLOAD_DIR)
    db = Chroma.from_documents(documents, embedding=embedding_function, persist_directory=CHROMA_DB_DIR)

    with open(hash_file_path, "w") as f:
        f.writelines([h + "\n" for h in doc_hashes])

    return db

# ------------------------ Agents ------------------------

web_search_agent = Agent(
    model=llm_model,
    tools=[DuckDuckGoTools()],
    name="Web Search Agent",
    role="Search the internet for external information",
    instructions=[
        "Use this agent only when the user's question is not related to the uploaded documents.",
        "Always include the source URLs in your response.",
    ],
    show_tool_calls=True,
    markdown=True
)

website_agent = Agent(
    model=llm_model,
    tools=[WebsiteTools()],
    name="Website Agent",
    role="Visit URLs in the uploaded documents to extract relevant information",
    instructions=[
        "Use this agent only when the user's question is related to content found in URLs present in the uploaded documents.",
        "Always include the source or quote from the website in your response.",
    ],
    show_tool_calls=True,
    markdown=True
)

# ------------------------ UI Layout ------------------------

# st.sidebar.image("https://img.icons8.com/ios/452/pdf-2--v1.png", width=80)
# st.sidebar.title("📄 AGNO Document Q&A")
# st.sidebar.image("https://img.icons8.com/?size=100&id=U8CnD8A3IkcL&format=png&color=000000", width=80)
st.sidebar.image("https://img.icons8.com/?size=100&id=21mIvThYLAyM&format=png&color=000000", width=80)


st.markdown("""
    <h1 style="
        text-align: center; 
        background: linear-gradient(90deg, #7928CA, #FF0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    ">
        Agentic RAG using Agno and Langchain
    </h1>
""", unsafe_allow_html=True)

st.sidebar.markdown("Upload your documents and ask questions.")
uploaded_files = st.sidebar.file_uploader("Choose documents", type=["txt", "pdf", "md","doc","docx","xlsx","csv","ppt"], accept_multiple_files=True)

# Main Title
st.markdown("<div class='section-header'>🧠 Document Knowledge Assistant</div>", unsafe_allow_html=True)

# ------------------------ Core Functionality ------------------------

if uploaded_files:
    doc_hashes = save_uploaded_files(uploaded_files)
    st.success("✅ Files uploaded. Initializing vectorstore...")

    db = create_or_load_vectorstore(doc_hashes)
    st.success("🧠 Vectorstore created successfully!")

    retriever = db.as_retriever()
    knowledge_base = LangChainKnowledgeBase(retriever=retriever)

    agent = Agent(
        model=llm_model,
        knowledge=knowledge_base,
        add_references=True,
        add_history_to_messages=True,
        team=[web_search_agent, website_agent],
        instructions=[
    "You are an assistant that first tries to answer from the uploaded documents.",
    "If the user's question is not related to the documents, respond with: '📢 This question is not related to the uploaded documents. Searching the web for relevant information...'",
    "When the question is unrelated to the documents, use DuckDuckGo search to find a reliable answer, and always include source links in the response.",
    "If the document contains relevant URLs and the user's question is related to one of those links, use WebsiteTools to search the content of the site.",
    "Always return concise, accurate, and source-backed answers.",
    "If no information is found, politely mention that you couldn't find a reliable answer.",
]

    )

    # ------------------------ Question & Answer ------------------------

    user_query = st.text_input("💬 Ask a question about the uploaded documents:", key="user_query")

    if user_query:
        with st.spinner("🔍 Thinking..."):
            result = agent.run(user_query)

        st.session_state.chat_history.extend([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": result.get_content_as_string()}
        ])

        st.markdown("<div class='section-header'>📬 Answer</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-assistant'>{result.get_content_as_string()}</div>", unsafe_allow_html=True)

        if result.citations:
            st.markdown("<div class='section-header'>📚 References</div>", unsafe_allow_html=True)
            for ref in result.citations:
                st.markdown(f"- {ref}")

        st.markdown("<div class='section-header'>🕘 Conversation History</div>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            role_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-assistant"
            st.markdown(f"<div class='{role_class}'><strong>{msg['role'].capitalize()}:</strong> {msg['content']}</div>", unsafe_allow_html=True)

else:
    st.info("⬅️ Upload documents using the sidebar to get started.")
