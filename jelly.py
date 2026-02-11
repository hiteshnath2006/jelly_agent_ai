import streamlit as st
import ollama
import json
from datetime import datetime

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Jelly AI", layout="wide")
st.markdown(
    """
    <style>
    /* Dark theme */
    body {background-color: #0e1117; color: #e0e0e0;}
    .css-18e3th9 {background-color: #0e1117;}  /* main area */
    .css-1d391kg {background-color: #0e1117;}  /* sidebar */
    .stButton>button {background-color: #1f222b; color: #e0e0e0;}
    .stSelectbox>div>div {background-color: #1f222b; color: #e0e0e0;}
    .user-msg {background:#1e1e2f; padding:8px; border-radius:5px; margin:5px 0;}
    .ai-msg {background:#2b2b3c; padding:8px; border-radius:5px; margin:5px 0;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LANGUAGES ---------------- #
LANGUAGES = {
    "as": {"name": "অসমীয়া (Assamese)", "title": "🪼 Jelly", "threads": "🧵 থ্ৰেডসমূহ", "new_thread": "➕ নতুন থ্ৰেড",
           "history": "📚 ইতিহাস", "share": "🔗 শেয়াৰ", "delete": "🗑️", "edit": "✏️",
           "talk": "💬 জেলীৰ সৈতে কথা পাতক...", "created": "সৃষ্টি"},
    "bn": {"name": "বাংলা (Bengali)", "title": "🪼 Jelly", "threads": "🧵 থ্রেড", "new_thread": "➕ নতুন থ্রেড",
           "history": "📚 ইতিহাস", "share": "🔗 শেয়ার", "delete": "🗑️", "edit": "✏️",
           "talk": "💬 জেলির সাথে কথা বলুন...", "created": "তৈরি"},
    "de": {"name": "Deutsch", "title": "🪼 Jelly", "threads": "🧵 Threads", "new_thread": "➕ Neuer Thread",
           "history": "📚 Verlauf", "share": "🔗 Teilen", "delete": "🗑️", "edit": "✏️",
           "talk": "💬 Mit Jelly sprechen...", "created": "Erstellt"},
    "en": {"name": "English", "title": "🪼 Jelly", "threads": "🧵 Threads", "new_thread": "➕ New Thread",
           "history": "📚 History", "share": "🔗 Share", "delete": "🗑️", "edit": "✏️",
           "talk": "💬 Talk to Jelly...", "created": "Created"},
    "es": {"name": "Español", "title": "🪼 Jelly", "threads": "🧵 Hilos", "new_thread": "➕ Nuevo Hilo",
           "history": "📚 Historial", "share": "🔗 Compartir", "delete": "🗑️", "edit": "✏️",
           "talk": "💬 Habla con Jelly...", "created": "Creado"},
    "fr": {"name": "Français", "title": "🪼 Jelly", "threads": "🧵 Fils", "new_thread": "➕ Nouveau Fil",
           "history": "📚 Historique", "share": "🔗 Partager", "delete": "🗑️", "edit": "✏️",
           "talk": "💬 Parlez à Jelly...", "created": "Créé"},
    "hi": {"name": "हिंदी (Hindi)", "title": "🪼 Jelly", "threads": "🧵 थ्रेड्स", "new_thread": "➕ नया थ्रेड",
           "history": "📚 इतिहास", "share": "🔗 शेयर", "delete": "🗑️", "edit": "✏️",
           "talk": "💬 जेली से बात करें...", "created": "बनाया"},
    "ja": {"name": "日本語", "title": "🪼 Jelly", "threads": "🧵 スレッド", "new_thread": "➕ 新規スレッド",
           "history": "📚 履歴", "share": "🔗 共有", "delete": "🗑️", "edit": "✏️",
           "talk": "💬 ジェリーと話す...", "created": "作成"},
    "zh": {"name": "中文", "title": "🪼 Jelly", "threads": "🧵 线程", "new_thread": "➕ 新线程",
           "history": "📚 历史", "share": "🔗 分享", "delete": "🗑️", "edit": "✏️",
           "talk": "💬 与Jelly聊天...", "created": "创建"}
}

# ---------------- SESSION INIT ---------------- #
if "language" not in st.session_state:
    st.session_state.language = "en"
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "current_thread" not in st.session_state:
    st.session_state.current_thread = None
if "agents" not in st.session_state:
    st.session_state.agents = {}
if "model" not in st.session_state:
    st.session_state.model = "llama3.1"
if "menu_open" not in st.session_state:
    st.session_state.menu_open = {}

# Only create a default thread if no threads exist
if not st.session_state.threads:
    thread_id = "main"
    st.session_state.threads[thread_id] = {
        "id": thread_id,
        "title": "Main Conversation",
        "messages": [],
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "created_ts": datetime.now().timestamp()
    }
    st.session_state.current_thread = thread_id

lang = LANGUAGES[st.session_state.language]

# ---------------- AGENT ---------------- #
class JellyAgent:
    def __init__(self, model):
        self.model = model

    def chat(self, messages):
        try:
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "num_predict": 256,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            full_response = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    full_response += chunk['message']['content']
                    yield full_response
        except Exception as e:
            yield f"Error: {str(e)}"

# ---------------- HEADER ---------------- #
header_col1, header_col2, header_col3 = st.columns([1,2,1])

with header_col1:
    st.session_state.model = st.selectbox(
        "Model",
        ["llama3.1","llama3","mistral"],
        index=["llama3.1","llama3","mistral"].index(st.session_state.model),
        label_visibility="collapsed"
    )


with header_col2:
    st.markdown(f"<h1 style='text-align:center'>{lang['title']}</h1>", unsafe_allow_html=True)

with header_col3:
    current_thread = st.session_state.threads[st.session_state.current_thread]

    # Generate shareable link
    share_url = f"https://jelly.com/?thread_id={current_thread['id']}"

    if st.button(lang["share"]):
        # One-click copy using st.markdown and JS
        st.markdown(f"""
        <input type="text" value="{share_url}" id="shareLink" style="position:absolute; left:-9999px;">
        <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)">Link copied to clipboard!</button>
        """, unsafe_allow_html=True)



# ---------------- LAYOUT ---------------- #
left_col, center_col = st.columns([1.2,3.8])

# ---------------- LEFT PANEL ---------------- #
with left_col:
    st.markdown(f"### {lang['threads']}", unsafe_allow_html=True)

    if st.button(lang['new_thread']):
        thread_id = f"thread_{int(datetime.now().timestamp())}"
        st.session_state.threads[thread_id] = {
            "id": thread_id,
            "title": f"New Chat {len(st.session_state.threads)}",
            "messages": [],
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "created_ts": datetime.now().timestamp()
        }
        st.session_state.current_thread = thread_id
        st.rerun()

    st.divider()
    st.markdown(f"### {lang['history']}", unsafe_allow_html=True)

    sorted_threads = sorted(
        st.session_state.threads.items(),
        key=lambda x: x[1]["created_ts"],
        reverse=True
    )

    for thread_id, thread_data in sorted_threads:
        col1, col2 = st.columns([4,1])

        with col1:
            if st.button(f"📄 {thread_data['title']}", key=f"thread_{thread_id}"):
                st.session_state.current_thread = thread_id
                st.rerun()

        with col2:
            if st.button("⋮", key=f"menu_{thread_id}"):
                st.session_state.menu_open[thread_id] = not st.session_state.menu_open.get(thread_id, False)

            if st.session_state.menu_open.get(thread_id, False):
                new_title = st.text_input("Rename", value=thread_data["title"], key=f"rename_{thread_id}")
                if new_title and new_title != thread_data["title"]:
                    st.session_state.threads[thread_id]["title"] = new_title
                    st.session_state.menu_open[thread_id] = False
                    st.rerun()

                if st.button("Delete", key=f"delete_{thread_id}"):

                    # Delete the thread safely
                    if thread_id in st.session_state.threads:
                        del st.session_state.threads[thread_id]

                    if thread_id in st.session_state.agents:
                        del st.session_state.agents[thread_id]

                    st.session_state.menu_open[thread_id] = False

                    # Switch current_thread if needed
                    if st.session_state.current_thread == thread_id:
                        if st.session_state.threads:
                            # Pick the most recent thread
                            st.session_state.current_thread = max(
                                st.session_state.threads.items(),
                                key=lambda x: x[1]["created_ts"]
                            )[0]
                        else:
                            # No threads left, create default main
                            new_id = "main"
                            st.session_state.threads[new_id] = {
                                "id": new_id,
                                "title": "Main Conversation",
                                "messages": [],
                                "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "created_ts": datetime.now().timestamp()
                            }
                            st.session_state.current_thread = new_id

                    st.rerun()

# ---------------- CENTER (CHAT AREA) ---------------- #
with center_col:
    current_thread_id = st.session_state.current_thread
    if current_thread_id not in st.session_state.threads:
        st.session_state.current_thread = list(st.session_state.threads.keys())[0]
        current_thread_id = st.session_state.current_thread

    current_thread = st.session_state.threads[current_thread_id]

    st.markdown(f"**{current_thread['title']}** | {len(current_thread['messages'])} msgs")

    agent = JellyAgent(st.session_state.model)

    # Chat Box
    for msg in current_thread["messages"]:
        if msg["role"]=="user":
            st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-msg">🪼 {msg["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input(lang['talk']):
        current_thread["messages"].append({"role":"user","content":prompt})
        st.markdown(f'<div class="user-msg">👤 {prompt}</div>', unsafe_allow_html=True)

        trimmed = current_thread["messages"][-8:]
        messages_for_model = [{"role":"system","content":"You are 🪼 Jelly, a helpful AI assistant."}] + trimmed

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for chunk in agent.chat(messages_for_model):
                full_response = chunk
                placeholder.markdown(f'<div class="ai-msg">🪼 {full_response}</div>', unsafe_allow_html=True)

            current_thread["messages"].append({"role":"assistant","content":full_response})

# ---------------- TRANSLATOR BOTTOM-LEFT ---------------- #
translator_col1, translator_col2 = st.columns([9,1])
with translator_col2:
    selected_lang = st.selectbox(
        "🌐",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.language),
        format_func=lambda x: LANGUAGES[x]["name"],
        key="translator_dropdown",
        label_visibility="collapsed"
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

st.markdown("---")
st.markdown("*Jelly AI — Llama3.1*")
st.markdown("*Jelly uses only LLama3.1 for now, further models shall be added later.*")
