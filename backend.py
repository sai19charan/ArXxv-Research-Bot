import streamlit as st
from pinecone import Pinecone
import re
from sentence_transformers import SentenceTransformer
import google.genai as genai 
from google.genai import Client # <-- New explicit import for clarity
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
def get_gemini_model_client(): # <-- Renamed function for clarity
    """Initializes and caches the Gemini Client connection."""
    print("Initializing Gemini Client...")
    
    # **FIX:** Replace genai.configure() with client instantiation
    # The client will handle the connection logic
    client = Client(api_key=GEMINI_API_KEY) 

    print("Gemini Client ready.")
    return client

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
    (This function is currently unused in your main RAG pipeline but kept for integrity.)
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
        
    # Get the Gemini Client object
    llm_client = get_gemini_model_client()
    prompt = create_augmented_prompt(query, search_results)
    
    try:
        # **FIX:** Call generate_content on the client object and pass the model name
        response = llm_client.models.generate_content(
            model='gemini-2.5-flash', # Use a stable model name
            contents=prompt
        )
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



# ================================================
# SPELL / GRAMMAR CORRECTION USING GEMINI LLM
# ================================================
# def correct_query_with_llm(user_query: str) -> str:
#     """
#     Uses Gemini to correct spelling and grammar in the user's query.
#     Keeps domain-specific terms (e.g., ML, AI, arXiv) unchanged.
#     """
#     if not user_query or not user_query.strip():
#         return user_query

#     try:
#         # Get Gemini client (cached)
#         llm_client = get_gemini_model_client()

#         # Construct a prompt to preserve technical terms
#         prompt = f"""
#         You are a precise academic writing assistant.
#         Correct any spelling or grammatical errors in the following query
#         while preserving domain-specific and technical words such as
#         machine learning terms, model names, or dataset names.
        
#         After correction, convert query to a concise and clear form which is suitable for a research paper search engine that uses scientific text embeddings.

#         Return only the corrected and final query, no explanations.

#         Query: "{user_query}"
#         """

#         # Call Gemini
#         response = llm_client.models.generate_content(
#             model='gemini-2.5-flash',
#             contents=prompt
#         )

#         # Extract corrected text safely
#         corrected_query = response.text.strip() if hasattr(response, 'text') else user_query

#         # Handle empty or invalid responses
#         if not corrected_query:
#             corrected_query = user_query

#         print(f"[Spell Check] Original: {user_query} --> Corrected: {corrected_query}")
#         return corrected_query

#     except Exception as e:
#         print(f"Spell correction error: {e}")
#         return user_query


# ================================================
# ADVANCED QUERY PREPROCESSING PIPELINE
# ================================================
import re

def basic_clean(text: str) -> str:
    """
    Performs minimal text cleaning for embedding stability.
    - Lowercases
    - Removes URLs and unwanted characters
    - Normalizes whitespace
    """
    if not text:
        return text
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)                # remove URLs
    text = re.sub(r"[_\-]+", " ", text)               # replace _ or - with space
    text = re.sub(r"[^a-z0-9\s\+\#\.\,\%\(\)]", "", text)  # keep useful symbols
    text = re.sub(r"\s+", " ", text)                  # normalize spaces
    return text


def expand_query_with_llm(user_query: str) -> list:
    """
    Optionally uses Gemini to expand a query with related terms or paraphrases.
    Returns a list of variations including the original query.
    """
    if not user_query.strip():
        return [user_query]
    try:
        llm_client = get_gemini_model_client()
        prompt = f"""
        Generate 3 concise paraphrases or related variations of this scientific query,
        preserving technical meaning and domain-specific terms.
        Return them as a simple numbered list, no explanations.

        Query: "{user_query}"
        """
        response = llm_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        raw_text = response.text.strip() if hasattr(response, 'text') else user_query
        # Extract each line as a variation
        variations = [line.strip("- ").strip() for line in raw_text.split("\n") if line.strip()]
        # Always include the original query at index 0
        return [user_query] + variations[:3]
    except Exception as e:
        print(f"Query expansion error: {e}")
        return [user_query]


def correct_query_with_llm(user_query: str) -> str:
    """
    Uses Gemini to correct spelling and grammar in the user's query.
    Keeps domain-specific terms and optimizes for embedding input.
    """
    if not user_query or not user_query.strip():
        return user_query

    try:
        llm_client = get_gemini_model_client()
        prompt = f"""
        You are a precise academic writing assistant.
        Correct any spelling or grammatical errors in the following query while preserving
        domain-specific and technical words (e.g., model names, algorithms, dataset names).

        Then, rephrase it into a concise, clear, and semantically rich version suitable for
        a research paper search engine that uses scientific text embeddings.

        Return only the corrected and final query text — no explanations, no formatting.

        Query: "{user_query}"
        """
        response = llm_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        corrected_query = response.text.strip() if hasattr(response, 'text') else user_query
        if not corrected_query:
            corrected_query = user_query
        print(f"[Spell Check] Original: {user_query} --> Corrected: {corrected_query}")
        return corrected_query

    except Exception as e:
        print(f"Spell correction error: {e}")
        return user_query


def preprocess_user_query(user_query: str, spell_check=True, clean_text=True, expand_query=False):
    """
    Complete preprocessing pipeline that sequentially:
    1. Corrects spelling and grammar (LLM)
    2. Cleans text (regex-based)
    3. Optionally expands the query with LLM
    """
    final_query = user_query

    if spell_check:
        final_query = correct_query_with_llm(final_query)

    if clean_text:
        final_query = basic_clean(final_query)

    if expand_query:
        expanded = expand_query_with_llm(final_query)
        print(f"[Query Expansion] Generated variations: {expanded}")
        return expanded  # list of queries

    return final_query  # single cleaned string

