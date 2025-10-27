import streamlit as st
from st_keyup import st_keyup
from backend import conversational_search_and_generate, get_suggestions
import time

# Page Configuration 
st.set_page_config(
    page_title="arXiv Research Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    iframe { border: none !important; }
    hr { display: none !important; }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0 10px auto;
        max-width: 70%;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        float: right;
        clear: both;
    }
    
    .assistant-message {
        background: #f0f2f6;
        color: #1f2937;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px auto 10px 0;
        max-width: 75%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        float: left;
        clear: both;
    }
    
    .message-container {
        margin: 15px 0;
        overflow: auto;
    }
    
    .reformulation-note {
        background: #fff3cd;
        color: #856404;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.85em;
        margin: 5px 0;
        border-left: 3px solid #ffc107;
    }
    
    .suggestion-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0 20px 0;
    }
    
    .stButton>button {
        border-radius: 50px;
        background-color: #667eea;
        color: white;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #764ba2;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .about-section {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin-top: 15px;
        border: 1px solid #dee2e6;
        font-size: 0.85em;
    }
    
    .about-section h4 {
        color: #667eea;
        margin-bottom: 8px;
        font-size: 1em;
    }
    
    .about-section p {
        font-size: 0.85em;
        line-height: 1.5;
        color: #4b5563;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization 
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        "conversation": [],
        "history": [],
        "current_query": "",
        "suggestions": [],
        "show_suggestions": False,
        "last_processed_input": "",
        "autocomplete_ready": False,
        "component_key_counter": 0,
        "processing": False  # Preventing duplicate processing
    }

# Initializing autocomplete system
if not st.session_state.app_state['autocomplete_ready']:
    with st.spinner("🔧 Initializing research bot..."):
        from backend import get_autocomplete_system
        get_autocomplete_system()
        st.session_state.app_state['autocomplete_ready'] = True
        st.success(" Ready!")
        time.sleep(0.5)
        st.rerun()

st.title("🔬 arXiv Research Chatbot")
st.markdown("*Ask questions about research papers. I'll remember our conversation!*")

def apply_suggestion(suggestion):
    """Replace the last word in the query with the selected suggestion."""
    current_input = st.session_state.app_state['current_query']
    words = current_input.split()
    
    if words:
        words[-1] = suggestion
        new_query = " ".join(words) + " "
    else:
        new_query = suggestion + " "
    
    st.session_state.app_state['current_query'] = new_query
    st.session_state.app_state['last_processed_input'] = ""
    st.session_state.app_state['show_suggestions'] = False
    st.session_state.app_state['suggestions'] = []
    st.session_state.app_state['component_key_counter'] += 1

def start_new_chat():
    """Save current conversation and start a new one."""
    if st.session_state.app_state['conversation']:
        first_user_message = next((msg['content'] for msg in st.session_state.app_state['conversation'] if msg['role'] == 'user'), "New Chat")
        st.session_state.app_state['history'].insert(0, {
            'title': first_user_message[:50] + "..." if len(first_user_message) > 50 else first_user_message,
            'conversation': st.session_state.app_state['conversation'].copy(),
            'timestamp': time.time()
        })
        st.session_state.app_state['history'] = st.session_state.app_state['history'][:10]
    
    st.session_state.app_state['conversation'] = []
    st.session_state.app_state['current_query'] = ""
    st.session_state.app_state['suggestions'] = []
    st.session_state.app_state['show_suggestions'] = False

# Display conversation
chat_container = st.container()
with chat_container:
    for message in st.session_state.app_state['conversation']:
        if message['role'] == 'user':
            st.markdown(f'<div class="message-container"><div class="user-message">👤 {message["content"]}</div></div>', unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.markdown(f'<div class="message-container"><div class="assistant-message"> {message["content"]}</div></div>', unsafe_allow_html=True)
            
            if message.get('reformulated_query') and message.get('original_query') != message.get('reformulated_query'):
                st.markdown(f'<div class="reformulation-note">💡 I understood this as: "{message["reformulated_query"]}"</div>', unsafe_allow_html=True)
            
            if message.get('papers'):
                with st.expander(f"📚 View {len(message['papers'])} Related Papers"):
                    for i, paper in enumerate(message['papers']):
                        st.markdown(f"**{i+1}. {paper['title']}**")
                        st.markdown(f"[View on arXiv]({paper['link']})")
                        with st.expander("Show Abstract"):
                            st.write(paper['summary'])
                        if i < len(message['papers']) - 1:
                            st.divider()

# Input section
if st.session_state.app_state['autocomplete_ready'] and not st.session_state.app_state['processing']:
    input_key = f"query_input_{st.session_state.app_state['component_key_counter']}"
    
    user_query = st_keyup(
        "Type your research question here...",
        value=st.session_state.app_state['current_query'],
        placeholder="e.g., What are the latest techniques for anomaly detection?",
        debounce=250,
        key=input_key
    )

    # Only updating if actually changed
    if user_query != st.session_state.app_state['last_processed_input']:
        st.session_state.app_state['current_query'] = user_query
        st.session_state.app_state['last_processed_input'] = user_query
        
        if user_query and len(user_query.strip()) > 1:
            try:
                suggestions = get_suggestions(user_query)
                st.session_state.app_state['suggestions'] = suggestions if suggestions else []
                st.session_state.app_state['show_suggestions'] = len(suggestions) > 0
            except Exception as e:
                st.session_state.app_state['suggestions'] = []
                st.session_state.app_state['show_suggestions'] = False
        else:
            st.session_state.app_state['suggestions'] = []
            st.session_state.app_state['show_suggestions'] = False

    # Display suggestions
    if st.session_state.app_state['show_suggestions'] and st.session_state.app_state['suggestions']:
        st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
        st.markdown("**💡 Suggestions:**")
        
        cols = st.columns(min(len(st.session_state.app_state['suggestions']), 5))
        for idx, suggestion in enumerate(st.session_state.app_state['suggestions']):
            with cols[idx]:
                if st.button(f"✨ {suggestion}", key=f"sug_{suggestion}_{idx}_{st.session_state.app_state['component_key_counter']}", use_container_width=True):
                    apply_suggestion(suggestion)
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Send button
    col1, col2, col3 = st.columns([4, 1, 4])
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)

    if send_button and user_query.strip() and not st.session_state.app_state['processing']:
        # Set processing flag to prevent duplicate clicks
        st.session_state.app_state['processing'] = True
        
        st.session_state.app_state['conversation'].append({
            "role": "user",
            "content": user_query.strip()
        })
        
        with st.spinner("🔍 Searching and analyzing papers..."):
            result = conversational_search_and_generate(
                user_query.strip(),
                st.session_state.app_state['conversation'][:-1]
            )
        
        st.session_state.app_state['conversation'].append({
            "role": "assistant",
            "content": result['generated_answer'],
            "original_query": result['original_query'],
            "reformulated_query": result['reformulated_query'],
            "papers": result['pinecone_results']
        })
        
        st.session_state.app_state['current_query'] = ""
        st.session_state.app_state['last_processed_input'] = ""
        st.session_state.app_state['show_suggestions'] = False
        st.session_state.app_state['component_key_counter'] += 1
        st.session_state.app_state['processing'] = False
        st.rerun()

# Sidebar
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.app_state['history']:
        st.markdown("### 💬 Recent Chats")
        for idx, hist_item in enumerate(st.session_state.app_state['history'][:5]):
            if st.button(f"{hist_item['title']}", key=f"hist_{idx}", use_container_width=True):
                st.session_state.app_state['conversation'] = hist_item['conversation'].copy()
                st.rerun()
        st.markdown("---")
    
    st.markdown("""
    <div class="about-section">
        <h4>ℹ️ About</h4>
        <p><strong>arXiv Research Chatbot</strong> - Discover research papers through conversation.</p>
        <p><strong>Features:</strong></p>
        <p>• Context-aware chat<br>
        • Fast spell checking<br>
        • Query reformulation<br>
        • 130K+ papers</p>
        <p style="margin-top: 8px; font-size: 0.75em; color: #9ca3af;">
        Built with ❤️ for researchers
        </p>
    </div>
    """, unsafe_allow_html=True)

# .\venv\Scripts\activate
# streamlit run ui.py
# What is the difference between BERT and Transformer models?
# What are the main applications of reinforcement learning ?
# How do Graph Neural Networks handle node classification tasks?
# Computers that can understand human language and text.
# Different kinds of neural networks?

