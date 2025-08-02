import streamlit as st
import asyncio
import aiohttp
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="NotionAtlas — AI Semantic Search & RAG for Notion",
    page_icon="🧭",
    layout="wide"
)

# -------------------- HEADER --------------------
st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; 
                padding: 18px 0 10px 0; background: #fff; box-shadow: 0 4px 16px rgba(30,30,40,0.05); border-radius: 14px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png" width="44" style="margin-right:13px">
        <div>
            <span style="font-size:2.1em; font-weight:800; letter-spacing:-2px; font-family: 'Inter', sans-serif; color:#222;">
                NotionAtlas
            </span>
            <span style="font-size:1.1em; color:#888; font-weight:400; margin-left:10px;">
                AI Semantic Search & RAG Assistant for Notion
            </span>
        </div>
        <a href="https://github.com/Yashraj-Muthyapwar/NotionAtlas-AI-Semantic-Search-And-RAG-Assistant-for-Notion" target="_blank" 
            title="GitHub Repo"
            style="margin-left:22px;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="27" style="vertical-align:middle;opacity:0.88"/>
        </a>
    </div>
""", unsafe_allow_html=True)
st.caption("Turn your Notion workspace into an intelligent, searchable knowledge hub.")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png", width=52)
    st.markdown("""
        <div style="font-size:1.2em; font-weight:600;">NotionAtlas</div>
        <div style="color:#555; font-size:0.99em; margin-bottom:9px;">
            Your intelligent Notion knowledge base
        </div>
        <hr style="margin:10px 0 8px 0;"/>
        <a href="https://github.com/Yashraj-Muthyapwar/NotionAtlas-AI-Semantic-Search-And-RAG-Assistant-for-Notion" target="_blank" 
            style="text-decoration:none;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="18" style="vertical-align:middle;opacity:0.8; margin-right:5px;"/>
            <span style="color:#222;font-size:0.97em;">View on GitHub</span>
        </a>
        <div style="margin-top:18px;color:#888;font-size:0.96em;">Built by Yashraj Muthyapwar</div>
    """, unsafe_allow_html=True)

# -------------------- TRY ASKING --------------------
st.markdown("""
<div style='padding:16px 24px; 
     background:linear-gradient(90deg, #ece9ff 85%, #fff 100%);
     border-radius:15px; 
     margin-bottom:1.5em; 
     border:1px solid #eee; 
     text-align:left;'>
    <span style='font-weight:600; color:#664ef7;'>💡 Try asking:</span>
    <ul style='margin:0.5em 0 0 1.1em;'>
        <li>How can I scrape tables from web pages using Pandas?</li>
        <li>Compare Decision Trees and Neural Networks for classification.</li>
        <li>Explain vocabulary and feature extraction in NLP.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# -------------------- Load Secrets and Models --------------------
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

# -------------------- Chat Bubble Function --------------------
def chat_bubble(msg, sender="user"):
    color = "#f1f3f4" if sender == "user" else "#e0f7fa"
    shadow = "0 2px 8px rgba(0,0,0,0.04)"
    icon = "🧑‍💻" if sender == "user" else "🤖"
    align = "flex-start" if sender == "user" else "flex-end"
    st.markdown(
        f"""
        <div style='background:{color}; padding:13px 20px 13px 20px; border-radius:15px;
        margin:9px 0; align-self:{align}; max-width:75%; 
        font-size:1.12em; box-shadow:{shadow}; transition: box-shadow 0.2s;'>
            <span style='font-size:1.25em; margin-right:8px'>{icon}</span>{msg}
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------- Async Chat with RAG --------------------
async def chat_with_memory(user_input: str):
    # 1️⃣ Encode query
    vector = embedder.encode(user_input).tolist()
    # 2️⃣ Query Qdrant
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5
    )
    # 3️⃣ Prepare semantic context
    context = "\n".join([
        hit.payload.get('chunk_text', '') for hit in results.points
    ]) or "No relevant context found."
    # 4️⃣ Update conversation memory
    st.session_state.conversation_context += f"\nUser: {user_input}"
    combined_context = (
        f"Conversation history:\n{st.session_state.conversation_context}\n\n"
        f"Relevant Notion context:\n{context}"
    )
    # 5️⃣ Call LLAMA API
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
    # 6️⃣ Save to chat history and memory
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.conversation_context += f"\nAssistant: {answer}"
    return answer

# -------------------- Chat Card (Main Area) --------------------
with st.container():
    st.markdown("""
    <div style="background:rgba(246,246,251,0.95);border-radius:18px;padding:24px 28px 20px 28px;
                box-shadow:0 8px 24px rgba(70,70,100,0.07);margin-bottom:28px;">
    """, unsafe_allow_html=True)
    
    user_input = st.chat_input("Ask anything about your Notion workspace...")
    if user_input:
        with st.spinner("Thinking..."):
            asyncio.run(chat_with_memory(user_input))

    # --- Display chat as beautiful chat bubbles
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_bubble(msg["content"], sender="user")
        else:
            chat_bubble(msg["content"], sender="assistant")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- CUSTOM FOOTER & CSS ---
    st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        padding: 10px;
        font-size: 1.08em;
    }
    </style>
    """,
    unsafe_allow_html=True
    )


