import streamlit as st
from pinecone import Pinecone
import re
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os 
from dotenv import load_dotenv 
import pandas as pd 

load_dotenv()

# CONFIGURATION 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
INDEX_NAME = 'arxiv-index'
MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_FILE = 'arXiv_scientific dataset.csv'

# Arxiv Link generation function
def get_arxiv_link(paper_id):
    """
    Generates a valid arXiv URL from various dataset ID formats.
    """
    base_url = "https://arxiv.org/abs/"
    if isinstance(paper_id, str) and paper_id.lower().startswith('abs-'):
        cleaned_id = paper_id[4:]
    else:
        cleaned_id = str(paper_id)
    cleaned_id = re.split(r'v\d+$', cleaned_id)[0]
    if cleaned_id and cleaned_id[0].isdigit():
        return f"{base_url}{cleaned_id}"
    else:
        match = re.match(r'(.*)-(\d{7})$', cleaned_id)
        if match:
            category = match.group(1)
            number_part = match.group(2)
            return f"{base_url}{category}/{number_part}"
        else:
            return f"Could not parse ID: {paper_id}"

# Using st.cache_resource to load the model and initialize Pinecone only once.
@st.cache_resource
def get_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    print("Loading embedding model...")
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def get_pinecone_index():
    """Initializes and caches the Pinecone index connection."""
    print("Initializing Pinecone connection...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    print("Pinecone connection ready.")
    return index

@st.cache_resource
def get_gemini_model():
    """Initializes and caches the Gemini model."""
    print("Initializing Gemini model...")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    print("Gemini model ready.")
    return model

@st.cache_resource
def get_title_lookup():
    """
    Loads the arXiv dataset and creates a dictionary mapping paper ID to title.
    This is cached so it only runs once per session.
    """
    print("Loading arXiv dataset for title lookup...")
    try:
        # Load only the necessary columns to save memory
        df = pd.read_csv(DATA_FILE, usecols=['id', 'title'])
        df.dropna(inplace=True)
        # Create a dictionary for fast lookups: { 'id': 'title' }
        title_lookup = df.set_index('id')['title'].to_dict()
        print("Title lookup dictionary created.")
        return title_lookup
    except FileNotFoundError:
        st.error(f"Dataset file not found at {DATA_FILE}. Cannot perform title lookups.")
        return {}
    except Exception as e:
        st.error(f"Error loading title data: {e}")
        return {}

def perform_search(query):
    """
    Takes a user query, embeds it, and performs a search in Pinecone.
    """
    model = get_embedding_model()
    index = get_pinecone_index()
    
    # Creating embedding for the user's query
    query_embedding = model.encode(query).tolist()

    # Query Pinecone to find the top 5 most similar papers
    search_results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Format the results for display
    formatted_results = []
    if search_results['matches']:
        for i, match in enumerate(search_results['matches']):
            paper_id = match['id']
            summary = match['metadata']['summary']
            score = match['score']
            link = get_arxiv_link(paper_id)
            
            result_item = {
                "id": paper_id,
                "summary": summary,
                "score": f"{score:.4f}",
                "link": link
            }
            formatted_results.append(result_item)
            
    return formatted_results

def create_augmented_prompt(query, search_results):
    context = "\n\n".join([f"Paper ID: {res['id']}\nSummary: {res['summary']}" for res in search_results])
    prompt_template = f"""
        You are a helpful AI research assistant. Your task is to answer the user's question based ONLY on the provided context from the research paper abstracts below. Do not use any external knowledge.

        ## User's Question:
        "{query}"

        ## Context from Research Papers:
        {context}

        ## Final Instruction:
        Based only on the context provided, synthesize a comprehensive, conversational answer to the user's question. Cite the relevant paper IDs (e.g., [Paper ID: cs-9308101v1]) for any claims you make. If the context does not contain enough information to answer, state that clearly.
        """
    return prompt_template

def generate_answer(query, search_results):
    """Generates an answer using the Gemini model and an augmented prompt."""
    if not search_results:
        return "No relevant papers were found to answer your question."
        
    llm = get_gemini_model()
    prompt = create_augmented_prompt(query, search_results)
    
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return "Sorry, there was an error generating the answer. Please try again."
    

def search_and_generate(query):
    """
    The main RAG pipeline function.
    1. Retrieves relevant documents from Pinecone.
    2. Generates a synthesized answer using Gemini.
    """
    model = get_embedding_model()
    index = get_pinecone_index()
    title_lookup = get_title_lookup()
    
    # 1. Retrieval
    query_embedding = model.encode(query).tolist()
    pinecone_results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Formatting Pinecone results for display
    formatted_results = []
    if pinecone_results['matches']:
        for match in pinecone_results['matches']:
            paper_id = match['id']
            # Safely get the title from our lookup dictionary
            title = title_lookup.get(paper_id, 'Title Not Found in Dataset')
            
            formatted_results.append({
                "id": paper_id,
                "summary": match['metadata']['summary'],
                "title": title, 
                "score": f"{match['score']:.4f}",
                "link": get_arxiv_link(paper_id)
            })

    # 2. Augmentation & Generation
    generated_answer = generate_answer(query, formatted_results)
    
    return {
        "pinecone_results": formatted_results,
        "generated_answer": generated_answer
    }


