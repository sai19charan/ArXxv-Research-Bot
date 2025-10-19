import streamlit as st
from backend import search_and_generate

st.set_page_config(
    page_title="arXiv RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 50px;
    }
    .ai-answer-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---

# --- Enhanced Session State for Chat Log ---
if 'chat_state' not in st.session_state:
    st.session_state.chat_state = {
        'step': 'greet',
        'query': '',
        'rag_results': {},
        'paper_index': 0,
        'max_papers': 15,
        'history': [],
        'round': 0,
        'chat_log': []  # List of {'role': 'user'/'bot', 'content': str}
    }


def reset_for_new_query():
    st.session_state.chat_state['step'] = 'ask_query'
    st.session_state.chat_state['query'] = ''
    st.session_state.chat_state['rag_results'] = {}
    st.session_state.chat_state['paper_index'] = 0
    st.session_state.chat_state['round'] = 0

# --- Chatbot UI Flow ---
st.title("arXiv Research Chatbot")


chat = st.container()
with chat:
    state = st.session_state.chat_state

    # --- Display Chat Log (no duplicates, no random white boxes) ---
    for msg in state['chat_log']:
        if msg['role'] == 'user':
            st.markdown(f"""
                <div class='ai-answer-box' style='margin-bottom:10px;max-width:80%;margin-left:auto;text-align:right;'>
                    <b>You:</b> {msg['content']}
                </div>
            """, unsafe_allow_html=True)
        elif msg['role'] == 'bot':
            st.markdown(f"<div class='ai-answer-box' style='margin-bottom:10px;max-width:80%;'>{msg['content']}</div>", unsafe_allow_html=True)

    # Step 1: Greet
    if state['step'] == 'greet':
        greet_msg = "Hello! Welcome to the arXiv Research Chatbot.<br>How can I help you today? Please enter your research question below."
        # Only append greet message if chat_log is empty
        if not state['chat_log']:
            state['chat_log'].append({'role': 'bot', 'content': greet_msg})
        if st.button("Start", key="start_btn"):
            state['step'] = 'ask_query'
            st.rerun()

    # Step 2: Ask for Query
    elif state['step'] == 'ask_query':
        user_query = st.text_input(
            "Ask a research question:",
            value=state['query'],
            placeholder="e.g., What are the latest techniques for anomaly detection?",
            key="query_input"
        )
        if st.button("Submit Query", key="submit_query_btn") and user_query.strip():
            state['query'] = user_query.strip()
            # Only append user query if last message is not the same
            if not state['chat_log'] or state['chat_log'][-1].get('content') != state['query']:
                state['chat_log'].append({'role': 'user', 'content': state['query']})
            with st.spinner("Searching and generating summary..."):
                rag_results = search_and_generate(state['query'])
            state['rag_results'] = rag_results
            state['paper_index'] = 0
            state['round'] = 1
            state['history'].insert(0, {'query': state['query'], 'results': rag_results})
            # Always use generated_answer from backend.py
            summary = rag_results.get('generated_answer', 'No summary available.')
            # Only append summary if last bot message is not the same
            if not state['chat_log'] or state['chat_log'][-1].get('content') != summary:
                state['chat_log'].append({'role': 'bot', 'content': summary})
            state['step'] = 'show_summaries'
            st.rerun()

    # Step 3: Show Summary of Summaries (only the generated answer)
    elif state['step'] == 'show_summaries':
        ask_msg = "Would you like to see the links and abstracts for the top 5 papers?"
        # Always print the question immediately before the options
        st.markdown(f"<div class='ai-answer-box' style='margin-bottom:10px;max-width:80%;'>{ask_msg}</div>", unsafe_allow_html=True)
        # Always show options unless just answered in THIS rerun
        show_options = True
        if len(state['chat_log']) >= 2:
            last_user = state['chat_log'][-1]
            last_bot = state['chat_log'][-2]
            if last_user['role'] == 'user' and last_user['content'] in ['Yes, show me the papers.', 'No, not now.'] and last_bot['role'] == 'bot' and last_bot['content'] == ask_msg:
                show_options = False
        if show_options:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes", key=f"show_abstracts_yes_{state['paper_index']}v2"):
                    state['chat_log'].append({'role': 'user', 'content': 'Yes, show me the papers.'})
                    state['step'] = 'show_abstracts'
                    st.rerun()
            with col2:
                if st.button("No", key=f"show_abstracts_no_{state['paper_index']}v2"):
                    state['chat_log'].append({'role': 'user', 'content': 'No, not now.'})
                    state['step'] = 'ask_new'
                    st.rerun()

    # Step 4: Show Links and Abstracts
    elif state['step'] == 'show_abstracts':
        rag_results = state['rag_results']
        papers = rag_results.get('pinecone_results', [])
        idx = state['paper_index']
        abstracts = papers[idx:idx+5]
        if abstracts:
            papers_msg = "Here are the links and abstracts for the selected papers:" + "<br>" + "<br>".join([
                f"<b>{i+1}. Title:</b> <code>{result['title']}</code><br>"
                f"<b>Link:</b> <a href='{result['link']}' target='_blank'>View on arXiv</a><br>"
                f"<details><summary>Show Abstract</summary>{result['summary']}</details>"
                for i, result in enumerate(abstracts, start=idx)
            ])
            if not state['chat_log'] or state['chat_log'][-1].get('content') != papers_msg:
                state['chat_log'].append({'role': 'bot', 'content': papers_msg})
            st.markdown(f"<div class='ai-answer-box' style='margin-bottom:10px;max-width:80%;'>{'Here are the links and abstracts for the selected papers:'}</div>", unsafe_allow_html=True)
            for i, result in enumerate(abstracts, start=idx):
                st.markdown(f"<div class='ai-answer-box' style='margin-bottom:10px;max-width:80%;'><b>{i+1}. Title:</b> <code>{result['title']}</code><br><b>Link:</b> <a href='{result['link']}' target='_blank'>View on arXiv</a></div>", unsafe_allow_html=True)
                with st.expander("Show Abstract"):
                    st.write(result['summary'])
        if idx + 5 < min(len(papers), state['max_papers']):
            ask_next_msg = "Would you like to see the next 5 related papers?"
            st.markdown(f"<div class='ai-answer-box' style='margin-bottom:10px;max-width:80%;'>{ask_next_msg}</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes", key=f"next5_yes_{idx}v2"):
                    if not state['chat_log'] or state['chat_log'][-1].get('content') != 'Yes, show me more papers.':
                        state['chat_log'].append({'role': 'user', 'content': 'Yes, show me more papers.'})
                    state['paper_index'] += 5
                    state['round'] += 1
                    state['step'] = 'show_abstracts'
                    st.rerun()
            with col2:
                if st.button("No", key=f"next5_no_{idx}v2"):
                    if not state['chat_log'] or state['chat_log'][-1].get('content') != 'No, that is enough.':
                        state['chat_log'].append({'role': 'user', 'content': 'No, that is enough.'})
                    state['step'] = 'ask_new'
                    st.rerun()
        else:
            state['step'] = 'ask_new'
            st.rerun()

    # Step 5: Ask for New Query or Exit
    elif state['step'] == 'ask_new':
        ask_new_msg = "Would you like to enter another query or exit?"
        st.markdown(f"<div class='ai-answer-box' style='margin-bottom:10px;max-width:80%;'>{ask_new_msg}</div>", unsafe_allow_html=True)
        # Always show options unless just answered in THIS rerun
        show_options = True
        if len(state['chat_log']) >= 2:
            last_user = state['chat_log'][-1]
            last_bot = state['chat_log'][-2]
            if last_user['role'] == 'user' and last_user['content'] in ['One more query.', 'Exit.'] and last_bot['role'] == 'bot' and last_bot['content'] == ask_new_msg:
                show_options = False
        if show_options:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("One More Query", key="one_more_query"):
                    state['chat_log'].append({'role': 'user', 'content': 'One more query.'})
                    reset_for_new_query()
                    st.rerun()
            with col2:
                if st.button("Exit", key="exit_btn"):
                    state['chat_log'].append({'role': 'user', 'content': 'Exit.'})
                    state['step'] = 'exit'
                    st.rerun()

    # Step 6: Exit
    elif state['step'] == 'exit':
        exit_msg = "Thank you for using the arXiv Research Chatbot! 👋"
        if not state['chat_log'] or state['chat_log'][-1].get('content') != exit_msg:
            state['chat_log'].append({'role': 'bot', 'content': exit_msg})
        st.markdown(f"<div class='ai-answer-box' style='margin-bottom:10px'>{exit_msg}</div>", unsafe_allow_html=True)

# --- Optional: Sidebar for History ---
with st.sidebar:
    st.header("Search History")
    if not st.session_state.chat_state['history']:
        st.info("Your search history will appear here.")
    else:
        for item in st.session_state.chat_state['history']:
            if st.button(item['query'], key=f"history_{item['query']}"):
                st.session_state.chat_state['query'] = item['query']
                st.session_state.chat_state['rag_results'] = item['results']
                st.session_state.chat_state['paper_index'] = 0
                st.session_state.chat_state['round'] = 1
                st.session_state.chat_state['step'] = 'show_summaries'
                st.rerun()


# .\venv\Scripts\activate
# streamlit run ui.py
# What is the difference between BERT and Transformer models?
# What are the main applications of reinforcement learning ?
# How do Graph Neural Networks handle node classification tasks?
# Computers that can understand human language and text.
# Different kinds of neural networks?


# embedding model?
# text preprocessing
# pinecone?
