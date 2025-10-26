import streamlit as st
from backend import search_and_generate 

# Page Configuration 
st.set_page_config(
    page_title="arXiv RAG Search",
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


# Session State Initialization 
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        "history": [],
        "rag_results": {},
        "current_query": ""
    }
    
# UI Layout 
st.title("arXiv Search Engine")
st.write("An intelligent search engine that finds relevant papers and answers.")

# Search Function 
def run_search(query):
    """A helper function to run the search and update the state."""
    if query:
        with st.spinner("Performing retrieval and generation... This may take a moment."):
            rag_results = search_and_generate(query)
            st.session_state.app_state['rag_results'] = rag_results
            st.session_state.app_state['current_query'] = query
            
            query_in_history = False
            for item in st.session_state.app_state['history']:
                if item['query'] == query:
                    query_in_history = True
                    break
            
            if not query_in_history:
                 st.session_state.app_state['history'].insert(0, {
                     "query": query, 
                     "results": rag_results  
                 })
    else:
        st.warning("Please enter a question to search.")

# Search Bar and Button 
user_query = st.text_input(
    "Ask a research question:",
    value=st.session_state.app_state['current_query'],
    # value="",
    placeholder="e.g., What are the latest techniques for anomaly detection?"
)


search_clicked = st.button("Search")
if search_clicked and user_query.strip():
    run_search(user_query.strip())


# Display RAG Results 
if st.session_state.app_state['rag_results']:
    # Display the AI-Generated Answer
    st.markdown(f'<div class="ai-answer-box">{st.session_state.app_state["rag_results"]["generated_answer"]}</div>', unsafe_allow_html=True)

    # Display the Retrieved Papers
    st.header("Retrieved Papers")
    retrieved_papers = st.session_state.app_state["rag_results"]["pinecone_results"]
    if retrieved_papers:
        for i, result in enumerate(retrieved_papers):
            with st.container():
                st.markdown(f"**{i+1}. Title:** `{result['title']}`")
                # st.markdown(f"**Similarity Score:** {result['score']}")
                st.markdown(f"**Link:** [View on arXiv]({result['link']})")
                with st.expander("Show Abstract"):
                    st.write(result['summary'])
                st.divider()
    else:
        st.info("No relevant papers were found in the database.")


# Display Search History 
with st.sidebar:
    st.header("Search History")
    if not st.session_state.app_state['history']:
        st.info("Your search history will appear here.")
    else:
        for item in st.session_state.app_state['history']:
            if st.button(item['query'], key=f"history_{item['query']}"):
                st.session_state.app_state['current_query'] = item['query']
                st.session_state.app_state['rag_results'] = item['results'] 
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

