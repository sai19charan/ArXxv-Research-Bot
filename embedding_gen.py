import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import re
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import time
import os 
from dotenv import load_dotenv 

load_dotenv()

# Text Pre-Processing        
def clean_arxiv_abstract(text):
    """
    Performs NLP pre-processing on a raw arXiv abstract to remove
    non-linguistic noise and normalize the text.
    """
    if not isinstance(text, str):
        return ""
    # Removing LaTeX equations (anything between $$...$$ or $...$)
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', '', text)
    # Removing citations (e.g., [1], [cs.AI], (Smith et al., 2023))
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[[a-zA-Z\.\-]+\]', '', text)
    text = re.sub(r'\s*\([A-Za-z\s]+\s*(et al\.)?,\s*\d{4}\)', '', text)
    # Removing newlines and excess whitespace
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Pinecone Configuration 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = "aws" 
PINECONE_REGION = "us-east-1"
INDEX_NAME = 'arxiv-index'

# Model Configuration 
MODEL_NAME = 'all-MiniLM-L6-v2' 

# Data Configuration 
DATA_FILE = 'arXiv_scientific dataset.csv'
BATCH_SIZE = 100 # Processing and upserting in batches of 100 for efficiency


print("Initializing Pinecone connection...")
pc = Pinecone(api_key=PINECONE_API_KEY)
print("Pinecone connection initialized.")

print(f"Loading embedding model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dimension: {embedding_dim}")

# Pinecone Index Creation
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
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)
else:
    print(f"Index '{INDEX_NAME}' already exists. Using it.")

index = pc.Index(INDEX_NAME)

# Inserting Data in Batches
print(f"Loading data from '{DATA_FILE}'...")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded data with {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: The file '{DATA_FILE}' was not found. Make sure it's in the same directory.")
    exit()

# Drop rows with missing summaries, if any
df.dropna(subset=['summary'], inplace=True)
df['summary'] = df['summary'].astype(str)
print(f"Processing {len(df)} rows after removing entries with no summary.")

# Pre Processing the summary
print("Applying text pre-processing to summaries...")
tqdm.pandas(desc="Cleaning text")
df['summary'] = df['summary'].progress_apply(clean_arxiv_abstract)
print("Pre-processing complete.")

print("\nGenerating embeddings and upserting to Pinecone in batches...")
# Processing the entire dataframe in batches
for i in tqdm(range(0, 50000, BATCH_SIZE)):
    i_end = min(i + BATCH_SIZE, 50000)
    batch_df = df.iloc[i:i_end]

    # Generating embeddings for the summary text
    summaries = batch_df['summary'].tolist()
    embeddings = model.encode(summaries, show_progress_bar=False).tolist()

    # Preparing data for upsertion
    vectors_to_upsert = []
    for j, row in batch_df.iterrows():
        paper_id = str(row['id'])
        vector = embeddings[j - i] 
        metadata = {'summary': row['summary']}
        vectors_to_upsert.append((paper_id, vector, metadata))
    
    # Upserting the batch to Pinecone
    index.upsert(vectors=vectors_to_upsert)
    print(i)

print("All batches have been upserted.")

print("Your dataset is now fully loaded and indexed in Pinecone!")





