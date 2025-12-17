import json
from pathlib import Path
from typing import List

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from src.models import Activity, Therapy

# Dati di esempio basati sul PDF "Progetto Digital Transformation - Alessandro De Martini.pdf"
# Rappresentano la terapia di un paziente con diverse attività.
sample_therapy_data = [
    {
        "activity_id": "act_001",
        "name": "Assunzione Aulin",
        "description": "Assumi l\'Aulin con acqua",
        "day_of_week": ["Lunedì", "Mercoledì", "Venerdì"],
        "time": "08:00",
        "dependencies": ["Colazione"]
    },
    {
        "activity_id": "act_002",
        "name": "Riabilitazione ginocchio",
        "description": "Fare gli esercizi con il fisioterapista per il ginocchio",
        "day_of_week": ["Martedì", "Giovedì", "Venerdì"],
        "time": "09:00-09:30",
        "dependencies": []
    },
    {
        "activity_id": "act_003",
        "name": "Esercizi cognitivi",
        "description": "Leggere un libro, oppure fare gli esercizi proposti dalla psicologa",
        "day_of_week": ["Lunedì", "Martedì", "Giovedì", "Venerdì"],
        "time": "10:00-10:30",
        "dependencies": []
    }
]

def ingest_data(output_dir: str = "data"):
    """
    Ingests sample therapy data into a ChromaDB vector store using LlamaIndex.
    Configures LlamaIndex to use Ollama for LLM and embeddings.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure Ollama as the default LLM and embedding model for LlamaIndex
    # Ensure Ollama server is running and both models are available locally
    Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # Transform sample data into LlamaIndex Document objects
    documents: List[Document] = []
    for activity_data in sample_therapy_data:
        activity = Activity(**activity_data)
        # We store each activity as a separate document for fine-grained retrieval
        document_text = f"Attività: {activity.name}\nDescrizione: {activity.description}\nGiorni: {', '.join(activity.day_of_week)}\nOrario: {activity.time}\nDipendenze: {', '.join(activity.dependencies) if activity.dependencies else 'Nessuna'}"
        documents.append(Document(text=document_text, metadata={"activity_id": activity.activity_id, "name": activity.name}))

    # Initialize ChromaDB client and collection
    db = chromadb.PersistentClient(path=str(Path(output_dir) / "chroma_db"))
    
    # Using a simple collection name, can be made dynamic
    collection_name = "patient_therapies"
    
    # Delete the collection if it already exists to ensure a clean slate
    try:
        db.delete_collection(name=collection_name)
        print(f"Collezione '{collection_name}' eliminata (se esistente).")
    except Exception as e:
        print(f"Errore durante l'eliminazione della collezione '{collection_name}': {e} (potrebbe non esistere, procedo).")

    chroma_collection = db.get_or_create_collection(collection_name)
    
    # Initialize ChromaVectorStore
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Wrap the vector store in a StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create a VectorStoreIndex from the documents
    # This will generate embeddings for the documents and store them in ChromaDB
    print("Creazione dell\'indice vettoriale e popolamento di ChromaDB...")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print(f"Indice creato con {len(documents)} documenti. Database salvato in {Path(output_dir) / 'chroma_db'}")
    
    
    return index

if __name__ == "__main__":
    print("Inizio fase di ingestion dati...")
    ingest_data()
    print("Ingestion dati completata.")
    print("\nRicorda di aver avviato Ollama e scaricato il modello 'llama3' ('ollama run llama3') prima di eseguire questo script.")
    print("Ora puoi implementare la logica di querying o la chat Streamlit.")
