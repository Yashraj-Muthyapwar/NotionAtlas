import streamlit as st
import asyncio
import aiohttp
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="NotionAtlas — AI Semantic Search & RAG for Notion",
    page_icon="🧭",
    layout="wide"
)

# ---------- HEADER ----------
st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; 
                padding: 28px 0 10px 0; margin-top: 24px;
                background: #fff; 
                box-shadow: 0 4px 32px rgba(30,30,40,0.07); 
                border-radius: 25px;
                max-width: 1100px;
                margin-left: auto; margin-right: auto;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png" width="55" style="margin-right:15px">
        <div>
            <span style="font-size:2.6em; font-weight:900; letter-spacing:-2px; font-family: 'Inter', sans-serif; color:#222;">
                NotionAtlas
            </span>
            <span style="font-size:1.25em; color:#868686; font-weight:400; margin-left:13px;">
                AI Semantic Search & RAG Assistant for Notion
            </span>
        </div>
        <a href="https://github.com/Yashraj-Muthyapwar/NotionAtlas-AI-Semantic-Search-And-RAG-Assistant-for-Notion" target="_blank" 
            title="GitHub Repo"
            style="margin-left:25px;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="32" style="vertical-align:middle;opacity:0.89"/>
        </a>
    </div>
""", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; font-size:1.16em; color:#888; margin-bottom:36px; margin-top:14px;'>"
    "Turn your Notion workspace into an intelligent, searchable knowledge hub."
    "</div>",
    unsafe_allow_html=True
)

# ---------- TRY ASKING SECTION ----------
st.markdown("""
<div style='
    padding: 28px 32px 22px 32px;
    background: linear-gradient(90deg, #ece9ff 88%, #f9f9ff 100%);
    border-radius: 26px;
    margin: 0 auto 2.4em auto;
    max-width: 950px;
    border: 1px solid #eee;
    box-shadow: 0 6px 32px rgba(90,90,160,0.08);
    '>
    <span style='font-weight:800; color:#745bf6; font-size:1.29em;'>💡 Try asking:</span>
    <ul style='margin:0.9em 0 0 1.3em; font-size:1.13em; color:#333;'>
        <li>How can I scrape tables from web pages using Pandas?</li>
        <li>Compare Decision Trees and Neural Networks for classification.</li>
        <li>Explain vocabulary and feature extraction in NLP.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png", width=54)
    st.markdown("""
        <div style="font-size:1.25em; font-weight:700; margin-bottom:7px;">NotionAtlas</div>
        <div style="color:#5a5a5a; font-size:0.99em; margin-bottom:12px;">
            Your intelligent Notion knowledge base
        </div>
        <hr style="margin:10px 0 13px 0;"/>
        <a href="https://github.com/Yashraj-Muthyapwar/NotionAtlas-AI-Semantic-Search-And-RAG-Assistant-for-Notion" target="_blank" 
            style="text-decoration:none;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="19" style="vertical-align:middle;opacity:0.84; margin-right:6px;"/>
            <span style="color:#222;font-size:0.97em;">View on GitHub</span>
        </a>
        <div style="margin-top:22px;color:#8a8a8a;font-size:0.99em;">Built by Yashraj Muthyapwar</div>
    """, unsafe_allow_html=True)

# ---------- Load Secrets and Models ----------
LLAMA_API_URL = "https://api.llama.com/v1/chat/completions"
LLAMA_API_KEY = st.secrets["LLAMA_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

COLLECTION_NAME = "notion_content"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(EMBEDDING_MODEL, use_auth_token=HF_TOKEN)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = ""

# ---------- Chat Bubble Function ----------
def chat_bubble(msg, sender="user"):
    color = "#fafafd" if sender == "user" else "#f3f7ff"
    border = "1px solid #edeef3" if sender == "user" else "1px solid #e1e8fb"
    shadow = "0 2px 8px rgba(40,60,100,0.04)"
    icon = "🧑‍💻" if sender == "user" else "🤖"
    align = "flex-start" if sender == "user" else "flex-end"
    st.markdown(
        f"""
        <div style='
            background:{color};
            border-radius:15px;
            border:{border};
            margin:10px 0;
            align-self:{align};
            max-width:75%;
            font-size:1.14em;
            box-shadow:{shadow};
            padding: 15px 22px 14px 18px;
            display: flex; align-items: center;
            '>
            <span style='font-size:1.24em; margin-right:10px'>{icon}</span>{msg}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Async Chat with RAG ----------
async def chat_with_memory(user_input: str):
    vector = embedder.encode(user_input).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5
    )
    context = "\n".join([
        hit.payload.get('chunk_text', '') for hit in results.points
    ]) or "No relevant context found."
    st.session_state.conversation_context += f"\nUser: {user_input}"
    combined_context = (
        f"Conversation history:\n{st.session_state.conversation_context}\n\n"
        f"Relevant Notion context:\n{context}"
    )
    payload = {
        "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are NotionAtlas, an AI assistant for Notion workspace queries. "
                    "Use the context if available. If context is insufficient, "
                    "start with: 'Note: This answer is **generated by LLAMA** "
                    "& falls **outside the scope of the Notion workspace data.**'"
                )
            },
            {"role": "user", "content": combined_context}
        ],
        "max_tokens": 500,
        "temperature": 0.2
    }
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(LLAMA_API_URL, json=payload, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                answer = data["completion_message"]["content"]["text"].strip()
            else:
                answer = f"Error: {await resp.text()}"
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.conversation_context += f"\nAssistant: {answer}"
    return answer

# ---------- CHAT MAIN CARD ----------
st.markdown("""
<div style='
    background: rgba(246,247,251,0.89);
    border-radius: 25px;
    padding: 35px 38px 25px 38px;
    margin: 0 auto 1.2em auto;
    max-width: 900px;
    box-shadow: 0 8px 32px rgba(80,90,120,0.08);
    border: 1px solid #f0f1f7;
    '>
""", unsafe_allow_html=True)

user_input = st.chat_input("Ask anything about your Notion workspace...")
if user_input:
    with st.spinner("Thinking..."):
        asyncio.run(chat_with_memory(user_input))

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        chat_bubble(msg["content"], sender="user")
    else:
        chat_bubble(msg["content"], sender="assistant")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER & INPUT CSS ----------
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .stTextInput>div>div>input {
        border-radius: 13px;
        border: 1.5px solid #e5e8ee;
        padding: 13px;
        font-size: 1.12em;
        margin-bottom: 8px;
        box-shadow: 0 1.5px 8px rgba(120,120,160,0.06);
        background: #fafaff;
    }
    .st-emotion-cache-1r4qj8v {padding-top: 1.4rem;}
    </style>
    """,
    unsafe_allow_html=True
)
