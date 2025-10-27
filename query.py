import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import re
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import time


def get_arxiv_link(paper_id):
    """
    Generates a valid arXiv URL from various dataset ID formats, removing any version numbers.
    """
    base_url = "https://arxiv.org/abs/"
    
    # Clean the ID by removing any potential 'abs-' prefix.
    if isinstance(paper_id, str) and paper_id.lower().startswith('abs-'):
        cleaned_id = paper_id[4:]
    else:
        cleaned_id = str(paper_id)
        
    # Remove the version number (e.g., 'v1', 'v2') from the end.
    cleaned_id = re.split(r'v\d+$', cleaned_id)[0]
        
    # Distinguish between old and new formats.
    if cleaned_id and cleaned_id[0].isdigit():
        return f"{base_url}{cleaned_id}"
    else:
        # Regex to find the category part and the number part.
        match = re.match(r'(.*)-(\d{7})$', cleaned_id)
        if match:
            category = match.group(1)
            number_part = match.group(2)
            return f"{base_url}{category}/{number_part}"
        else:
            # Fallback for IDs that don't match the old format regex
            return f"Could not parse ID: {paper_id}"


# --- Pinecone Configuration ---
PINECONE_API_KEY = "" 
PINECONE_CLOUD = "aws" 
PINECONE_REGION = "us-east-1"
INDEX_NAME = 'arxiv-index'

# --- Model Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' # A fast and effective model for semantic search

# --- Data Configuration ---
DATA_FILE = 'arXiv_scientific dataset.csv'
BATCH_SIZE = 100 # Process and upsert in batches of 100 for efficiency


print("Initializing Pinecone connection...")
pc = Pinecone(api_key=PINECONE_API_KEY)
print("Pinecone connection initialized.")

print(f"Loading embedding model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dimension: {embedding_dim}")

# Creating Pinecone Index
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Index '{INDEX_NAME}' not found. Creating a new one...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=embedding_dim,
        metric='cosine',
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )
    print(f"Index '{INDEX_NAME}' created successfully. Waiting for it to be ready...")
    # Wait for the index to be ready
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)
else:
    print(f"Index '{INDEX_NAME}' already exists. Using it.")

index = pc.Index(INDEX_NAME)


# Performing Semantic Serach
print("\n--- Performing a Test Search ---")
user_query = "technique developed is a variant of dependency-directed backtracking"
print(f"User Query: '{user_query}'")

#Create embedding for the user's query
query_embedding = model.encode(user_query).tolist()

# Query Pinecone to find the top 5 most similar papers
search_results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

print("\n--- Top 5 Similar Research Papers ---")
if search_results['matches']:
    for i, match in enumerate(search_results['matches']):
        paper_id = match['id']
        summary = match['metadata']['summary']
        score = match['score']
        link = get_arxiv_link(paper_id)
        
        print(f"\nResult {i+1}:")
        print(f"  ID: {paper_id}")
        print(f"  Similarity Score: {score:.4f}")
        print(f"  Link: {link}")
        print(f"  Summary: {summary[:250]}...") # Display first 250 characters
else:
    print("No results found.")


