import sqlite3
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from annoy import AnnoyIndex
import os
import sys

# Function to get embeddings from the API
def get_embedding(text):
    headers = {
        'Content-Type': 'application/json',  # Changed to application/json
    }
    data = json.dumps({
        "model": "llama3.1",
        "input": text 
    })
    response = requests.post('http://localhost:11434/api/embed', headers=headers, data=data)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    response_data = response.json()
    #print(f"API Response: {json.dumps(response_data, indent=2)}")
    
    if 'embedding' in response_data:
        return np.array(response_data['embedding'], dtype=np.float32)
    elif 'embeddings' in response_data:
        if response_data['embeddings']:
            return np.array(response_data['embeddings'][0], dtype=np.float32)
        else:
            raise ValueError(f"'embeddings' key is present but contains an empty list. Full response: {response_data}")
    else:
        raise KeyError(f"No embedding found in API response. Response: {response_data}")

# Function to create the database and table
def create_database():
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (id INTEGER PRIMARY KEY, text TEXT, embedding BLOB)''')
    conn.commit()
    return conn

# Function to insert data into the database
def insert_data(conn, text, embedding):
    c = conn.cursor()
    c.execute("INSERT INTO embeddings (text, embedding) VALUES (?, ?)",
              (text, sqlite3.Binary(np.array(embedding).tobytes())))
    conn.commit()

# Function to process the text file and generate embeddings
def process_file(filename, conn):
    # Count total lines in the file
    with open(filename, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    with open(filename, 'r') as f:
        for i, line in enumerate(f, 1):
            text = line.strip()
            embedding = get_embedding(text)
            insert_data(conn, text, embedding)
            
            # Show progress for first 10 lines, then every 10th line
            if i <= 10 or i % 10 == 0:
                progress = (i / total_lines) * 100
                print(f"Processed {i}/{total_lines} lines ({progress:.2f}%)")

    print(f"Completed processing {total_lines} lines (100.00%)")

# Function to create an index for faster similarity search
def create_index(conn):
    c = conn.cursor()
    c.execute("CREATE INDEX IF NOT EXISTS idx_embedding ON embeddings (embedding)")
    conn.commit()

# Function to build the Annoy index
def build_annoy_index(conn, vector_size=4096, n_trees=10):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM embeddings")
    total_vectors = c.fetchone()[0]
    
    annoy_index = AnnoyIndex(vector_size, 'angular')
    c.execute("SELECT id, embedding FROM embeddings")
    
    for i, (id, embedding_blob) in enumerate(c.fetchall()):
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        if len(embedding) != vector_size:
            print(f"Warning: Embedding size mismatch. Expected {vector_size}, got {len(embedding)}. Skipping this vector.")
            continue
        annoy_index.add_item(id - 1, embedding)  # Annoy uses 0-based indexing
        
        if i % 1000 == 0:
            print(f"Added {i}/{total_vectors} vectors to the index")
    
    print("Building index...")
    annoy_index.build(n_trees)
    annoy_index.save('embeddings.ann')
    print("Index built and saved")

# Function to find similar texts using Annoy
def find_similar(conn, query_text, top_k=5):
    query_embedding = get_embedding(query_text)
    
    annoy_index = AnnoyIndex(4096, 'angular')  # Change this to 4096
    annoy_index.load('embeddings.ann')
    
    similar_ids, distances = annoy_index.get_nns_by_vector(query_embedding, top_k, include_distances=True)
    
    c = conn.cursor()
    results = []
    for id, distance in zip(similar_ids, distances):
        c.execute("SELECT text FROM embeddings WHERE id = ?", (id + 1,))  # SQLite uses 1-based indexing
        text = c.fetchone()[0]
        similarity = 1 - distance  # Convert distance to similarity
        results.append((id + 1, text, similarity))
    
    return results

# Function to add a question
def add_question(conn, text):
    embedding = get_embedding(text)
    insert_data(conn, text, embedding)
    print("Question added successfully.")

    # Rebuild Annoy index
    print("Rebuilding Annoy index...")
    build_annoy_index(conn, vector_size=4096)
    print("Annoy index updated.")

# Function to rebuild the Annoy index
def rebuild_annoy_index(conn):
    print("Rebuilding Annoy index...")
    build_annoy_index(conn, vector_size=4096)
    print("Annoy index rebuilt.")

# Main execution
def main():
    db_file = 'embeddings.db'
    first_run = not os.path.exists(db_file)

    conn = create_database()

    if first_run:
        print("First run detected. Processing initial file and building indices...")
        process_file('large_text_file.txt', conn)
        create_index(conn)
        build_annoy_index(conn)
        print("Initial setup completed.")
    else:
        print("Rebuilding Annoy index...")
        rebuild_annoy_index(conn)

    while True:
        print("\n1. Add a question")
        print("2. Find similar questions")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            question = input("Enter your question: ")
            add_question(conn, question)
        elif choice == '2':
            query = input("Enter your query: ")
            similar_texts = find_similar(conn, query)
            print(f"\nTop 5 similar texts to '{query}':")
            for id, text, similarity in similar_texts:
                print(f"ID: {id}, Similarity: {similarity:.4f}")
                print(f"Text: {text[:100]}...")
                print()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

    conn.close()

if __name__ == "__main__":
    main()