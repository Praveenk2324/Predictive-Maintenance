import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

LOGS_PATH = "data/raw/maintenance_logs.csv"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "maintenance_knowledge_base"

def main():
    print("Loading maintenance logs...")
    df = pd.read_csv(LOGS_PATH)

    print("Initializing ChromaDB Persistent Client...")

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted existing collection to start fresh.")
    
    except Exception:
        pass

    print("Creating new vector collection and downloading embedding model...")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )

    print("Embedding and storing documents (This might take a minute)...")

    documents = df['text'].tolist()
    metadatas = [{"engine_id": int(row['engine_id']), "component": row['component']} for _, row in df.iterrows()]
    ids = df['log_id'].tolist()

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully ingested {collection.count()} logs into ChromaDB at ./{CHROMA_DB_DIR}")

if __name__ == "__main__":
    main()