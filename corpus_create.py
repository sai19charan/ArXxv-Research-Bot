import pandas as pd
from tqdm import tqdm
import re

DATA_FILE = 'arXiv_scientific dataset.csv'
OUTPUT_FILE = 'arxiv_corpus.txt'

print(f"Loading data from '{DATA_FILE}' to create a text corpus...")
try:
    # Loading only the summary column to save memory
    df = pd.read_csv(DATA_FILE, usecols=['summary'])
    df.dropna(subset=['summary'], inplace=True)
    df['summary'] = df['summary'].astype(str)
    print(f"Loaded {len(df)} summaries.")

    # Combining all summaries into a single block of text
    print("Combining summaries into a single text block...")
    text_corpus = ' '.join(summary for summary in tqdm(df['summary'], desc="Combining summaries"))

    # Replacing any occurrences of multiple whitespace characters (including newlines, tabs, and multiple spaces) with a single space.
    print("Cleaning corpus by normalizing whitespace...")
    text_corpus = re.sub(r'\s+', ' ', text_corpus).strip()
    print("Whitespace normalization complete.")

    # Writing the combined text to the output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(text_corpus)
        
    print(f"\nSuccessfully created '{OUTPUT_FILE}' with all combined summaries.")

except FileNotFoundError:
    print(f"Error: The file '{DATA_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

