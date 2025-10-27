import pandas as pd
import re

def create_question_corpus():
    """
    Create a question-focused corpus for spell checking.
    Combines multiple sources with question patterns.
    """
    
    # Downloading SQuAD 
    print("Downloading SQuAD questions...")
    import requests
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    response = requests.get(url)
    squad_data = response.json()
    
    questions = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                questions.append(qa['question'])
    
    # Write to file
    with open('question_corpus.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(questions))
    
    print(f"Created question_corpus.txt with {len(questions)} questions")
    print("Add this file to your corpus list!")

# Run this once
create_question_corpus()
