import streamlit as st
from backend import search_and_generate , preprocess_user_query

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

with st.sidebar:
    st.header("Preprocessing Options")
    spell_check = st.checkbox("Enable LLM Spell & Grammar Correction", value=True)
    clean_text = st.checkbox("Enable Basic Text Cleaning", value=True)
    expand_query = st.checkbox("Enable LLM Query Expansion", value=False)
    st.divider()

# Search Function 
def run_search(query):
    """Runs preprocessing, retrieval, and generation."""
    if query:
        with st.spinner("Preprocessing your query..."):
            processed_query = preprocess_user_query(
                user_query=query,
                spell_check=spell_check,
                clean_text=clean_text,
                expand_query=expand_query
            )

        # If query expansion is enabled, it returns multiple variants
        if isinstance(processed_query, list):
            # For now, we’ll use the first variant for main retrieval
            main_query = processed_query[0]
            st.info(f"✅ Query expanded into variants: {processed_query}")
        else:
            main_query = processed_query

        # If query changed after correction/cleaning, notify user
        if main_query != query:
            st.info(f"✅ Preprocessed query: **{main_query}**")

        with st.spinner("Performing retrieval and generation... This may take a moment."):
            rag_results = search_and_generate(main_query)
            st.session_state.app_state['rag_results'] = rag_results
            st.session_state.app_state['current_query'] = main_query

            # Store in history
            query_in_history = any(
                item['query'] == main_query for item in st.session_state.app_state['history']
            )
            if not query_in_history:
                st.session_state.app_state['history'].insert(0, {
                    "query": main_query,
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

