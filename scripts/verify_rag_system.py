import sys
import os
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import chromadb

def test_rag():
    print("--- RAG SYSTEM VERIFICATION ---")
    
    # 1. Setup Models (MUST MATCH main.py)
    print("1. Configuring Models...")
    try:
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        print("   -> Embedding Model: nomic-embed-text (OK)")
    except Exception as e:
        print(f"   -> FATAL: Could not init embedding model. {e}")
        return

    # 2. Connect to DB
    db_path = Path("data/chroma_db")
    print(f"2. Connecting to ChromaDB at {db_path}...")
    if not db_path.exists():
        print("   -> FAIL: DB directory does not exist. Run ingest_data.py first.")
        return

    try:
        db_client = chromadb.PersistentClient(path=str(db_path))
        collection = db_client.get_collection("patient_therapies")
        count = collection.count()
        print(f"   -> Connection Success. Found collection 'patient_therapies'.")
        print(f"   -> Document Count: {count}")
        
        if count == 0:
            print("   -> WARNING: Collection is empty!")
    except Exception as e:
        print(f"   -> FAIL: DB Connection Error: {e}")
        return

    # 3. Retrieval Test
    print("3. Performing Retrieval Tests...")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=3)

    test_queries = [
        "Aulin", 
        "Ginocchio", 
        "Psicologa",
        "Progetto Digital Transformation" # Check if PDF content is there
    ]

    for q in test_queries:
        print(f"\n   Query: '{q}'")
        results = retriever.retrieve(q)
        if not results:
            print("     -> No results found.")
        else:
            for i, res in enumerate(results):
                # Print score and snippet
                print(f"     [{i+1}] Score: {res.score:.4f} | Content: {res.node.text[:100]}...")

if __name__ == "__main__":
    test_rag()
