import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "maintenance_knowledge_base"

def query_knowledge_base(query_text, n_results=3):
    print(f"Searching for: '{query_text}'\n")

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    print("-" * 60)
    print("                RAG RETRIEVAL RESULTS")
    print("-" * 60)
    
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i] # Lower distance = closer match
        
        print(f"Result {i+1} (Distance: {distance:.4f})")
        print(f"Component: {metadata['component']}")
        print(f"Log: {doc}\n")
    print("-" * 60 + "\n")

def main():
    # Let's test a few realistic queries an engineer might type
    test_queries = [
        "High EGT and bleed air temperature deviation",
        "Excessive vibration on N1 and fan blade damage",
        "Oil pressure drop with metallic debris in the filter"
    ]
    
    for query in test_queries:
        query_knowledge_base(query)

if __name__ == "__main__":
    main()