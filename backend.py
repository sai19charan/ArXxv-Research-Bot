import streamlit as st
from pinecone import Pinecone
import re
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os 
from dotenv import load_dotenv 
import pandas as pd 
from prefix_checker import NGramTrie  
from auto_complete import ContextAwareAutoComplete

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
    """Generates a valid arXiv URL from various dataset ID formats."""
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

# Using st.cache_resource to load models
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
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("Gemini model ready.")
    return model

@st.cache_resource
def get_autocomplete_system():
    """Initializes and caches the autocomplete system."""
    print("Initializing autocomplete system...")
    corpus_files = ['big.txt', 'arxiv_corpus.txt', 'question_corpus.txt']
    autocomplete = ContextAwareAutoComplete(corpus_files)
    print("Autocomplete system ready.")
    return autocomplete

@st.cache_resource
def get_title_lookup():
    """Loads the arXiv dataset and creates a dictionary mapping paper ID to title."""
    print("Loading arXiv dataset for title lookup...")
    try:
        df = pd.read_csv(DATA_FILE, usecols=['id', 'title'])
        df.dropna(inplace=True)
        title_lookup = df.set_index('id')['title'].to_dict()
        print("Title lookup dictionary created.")
        return title_lookup
    except FileNotFoundError:
        st.error(f"Dataset file not found at {DATA_FILE}. Cannot perform title lookups.")
        return {}
    except Exception as e:
        st.error(f"Error loading title data: {e}")
        return {}

def get_suggestions(current_input):
    """Get autocomplete suggestions based on current input."""
    if not current_input or not current_input.strip():
        return []
    
    autocomplete = get_autocomplete_system()
    words = current_input.strip().split()
    
    if not words:
        return []
    
    current_word = words[-1]
    previous_words = words[:-1][-2:] if len(words) > 1 else []
    suggestions = autocomplete.get_suggestions(current_word, previous_words)
    
    return suggestions[:5]

# Query reformulation with conversation context
def reformulate_query(current_query, conversation_history):
    """
    Reformulates the current query using LLM by considering conversation history.
    Returns a complete, standalone query.
    """
    llm = get_gemini_model()
    
    # Building conversation context
    context_messages = []
    for msg in conversation_history[-3:]:  # Last 3 exchanges for context
        if msg['role'] == 'user':
            context_messages.append(f"User: {msg['content']}")
        elif msg['role'] == 'assistant':
            context_messages.append(f"Assistant: {msg['content'][:200]}...")  # Truncating long responses
    
    context = "\n".join(context_messages) if context_messages else "No previous conversation."
    
    reformulation_prompt = f"""You are a query reformulation assistant for a research paper search system.

Given the conversation history and the user's current query, reformulate the current query into a complete, standalone, and well-formed research question.

**Conversation History:**
{context}

**Current User Query:**
{current_query}

**Instructions:**
1. If the current query is incomplete or uses pronouns (it, that, this, etc.), resolve them using conversation history
2. Make the query specific and research-focused
3. Preserve the user's original intent
4. Return ONLY the reformulated query, nothing else
5. If the query is already complete and clear, return it as-is

**Reformulated Query:**"""
    
    try:
        response = llm.generate_content(reformulation_prompt)
        reformulated = response.text.strip()
        return reformulated
    except Exception as e:
        print(f"Error during query reformulation: {e}")
        return current_query  # Fallback to original query

def perform_search(query):
    """Takes a user query, embeds it, and performs a search in Pinecone."""
    model = get_embedding_model()
    index = get_pinecone_index()
    
    query_embedding = model.encode(query).tolist()
    search_results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    formatted_results = []
    if search_results['matches']:
        for match in search_results['matches']:
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

def create_augmented_prompt(query, search_results, conversation_history):
    """Creates augmented prompt with conversation context."""
    context = "\n\n".join([f"Paper ID: {res['id']}\nSummary: {res['summary']}" for res in search_results])
    
    # Adding conversation history for context
    conv_context = ""
    if conversation_history:
        recent_conv = conversation_history[-2:]  # Last exchange
        for msg in recent_conv:
            if msg['role'] == 'user':
                conv_context += f"\nPrevious User Question: {msg['content']}"
            elif msg['role'] == 'assistant':
                conv_context += f"\nPrevious Assistant Response: {msg['content'][:300]}..."
    
    prompt_template = f"""You are a helpful AI research assistant. Your task is to answer the user's question based ONLY on the provided context from the research paper abstracts below. Do not use any external knowledge.

{conv_context}

## Current User's Question:
"{query}"

## Context from Research Papers:
{context}

## Final Instruction:
Based only on the context provided, synthesize a comprehensive, conversational answer to the user's question. Cite the relevant paper IDs (e.g., [Paper ID: cs-9308101v1]) for any claims you make. If the context does not contain enough information to answer, state that clearly. If this is a follow-up question, acknowledge the conversation context."""
    return prompt_template

def generate_answer(query, search_results, conversation_history):
    """Generates an answer using the Gemini model with conversation context."""
    if not search_results:
        return "No relevant papers were found to answer your question."
        
    llm = get_gemini_model()
    prompt = create_augmented_prompt(query, search_results, conversation_history)
    
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return "Sorry, there was an error generating the answer. Please try again."

# Main conversational search function
def conversational_search_and_generate(user_query, conversation_history):
    """
    Main RAG pipeline with conversational context.
    1. Reformulates query using conversation history
    2. Retrieves relevant documents from Pinecone
    3. Generates contextual answer using Gemini
    """
    model = get_embedding_model()
    index = get_pinecone_index()
    title_lookup = get_title_lookup()
    
    # 1. Query Reformulation
    reformulated_query = reformulate_query(user_query, conversation_history)
    
    # 2. Retrieval using reformulated query
    query_embedding = model.encode(reformulated_query).tolist()
    pinecone_results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Format results
    formatted_results = []
    if pinecone_results['matches']:
        for match in pinecone_results['matches']:
            paper_id = match['id']
            title = title_lookup.get(paper_id, 'Title Not Found in Dataset')
            
            formatted_results.append({
                "id": paper_id,
                "summary": match['metadata']['summary'],
                "title": title, 
                "score": f"{match['score']:.4f}",
                "link": get_arxiv_link(paper_id)
            })

    # 3. Generation with conversation context
    generated_answer = generate_answer(reformulated_query, formatted_results, conversation_history)
    
    return {
        "original_query": user_query,
        "reformulated_query": reformulated_query,
        "pinecone_results": formatted_results,
        "generated_answer": generated_answer
    }
