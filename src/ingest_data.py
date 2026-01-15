import json
from pathlib import Path
from typing import Iterable

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


DATA_DIR = Path("data")
COLLECTION_NAME = "patient_therapies"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_activity_doc(activity: dict, patient_id: str) -> Document | None:
    name = activity.get("name")
    if not name:
        return None
    description = activity.get("description") or ""
    days = activity.get("day_of_week") or []
    time = activity.get("time") or ""
    dependencies = activity.get("dependencies") or []
    valid_from = activity.get("valid_from")
    valid_until = activity.get("valid_until")

    lines = [
        f"Attività: {name}",
        f"Descrizione: {description}",
        f"Giorni: {', '.join(days)}",
        f"Orario: {time}",
        f"Dipendenze: {', '.join(dependencies) if dependencies else 'Nessuna'}",
    ]
    if valid_from:
        lines.append(f"Validità dal: {valid_from}")
    if valid_until:
        lines.append(f"Validità fino al: {valid_until}")

    meta = {
        "type": "therapy_activity",
        "source": "therapy_json",
        "patient_id": patient_id,
        "activity_id": activity.get("activity_id"),
    }
    return Document(text="\n".join(lines), metadata=meta)


def _iter_therapy_docs() -> Iterable[Document]:
    therapy_dir = DATA_DIR / "therapies"
    if not therapy_dir.exists():
        return []

    docs: list[Document] = []
    for path in therapy_dir.glob("*.json"):
        data = _load_json(path)
        if not data:
            continue
        if isinstance(data, dict) and "activities" in data:
            activities = data.get("activities") or []
            patient_id = data.get("patient_id") or path.stem
        elif isinstance(data, list):
            activities = data
            patient_id = path.stem
        else:
            continue

        for activity in activities:
            if not isinstance(activity, dict):
                continue
            doc = _build_activity_doc(activity, patient_id)
            if doc:
                docs.append(doc)
    return docs


def _iter_patient_docs() -> Iterable[Document]:
    patient_dir = DATA_DIR / "patients"
    if not patient_dir.exists():
        return []

    docs: list[Document] = []
    for path in patient_dir.glob("*.json"):
        data = _load_json(path)
        if not data or not isinstance(data, dict):
            continue
        patient_id = data.get("patient_id") or path.stem
        for cond in data.get("medical_conditions") or []:
            docs.append(
                Document(
                    text=f"[Always] {cond}",
                    metadata={
                        "type": "patient_condition",
                        "category": "conditions",
                        "source": "patient_profile",
                        "patient_id": patient_id,
                    },
                )
            )
        for pref in data.get("preferences") or []:
            docs.append(
                Document(
                    text=f"[Always] {pref}",
                    metadata={
                        "type": "patient_preference",
                        "category": "preferences",
                        "source": "patient_profile",
                        "patient_id": patient_id,
                    },
                )
            )
        for habit in data.get("habits") or []:
            docs.append(
                Document(
                    text=f"[Always] {habit}",
                    metadata={
                        "type": "patient_habit",
                        "category": "habits",
                        "source": "patient_profile",
                        "patient_id": patient_id,
                    },
                )
            )
        for note in data.get("notes") or []:
            if not isinstance(note, dict):
                continue
            content = note.get("content")
            if not content:
                continue
            day = note.get("day")
            label = day or "Always"
            docs.append(
                Document(
                    text=f"[{label}] {content}",
                    metadata={
                        "type": "patient_note",
                        "category": "notes",
                        "source": "patient_profile",
                        "patient_id": patient_id,
                    },
                )
            )
    return docs


def _iter_caregiver_docs() -> Iterable[Document]:
    caregiver_dir = DATA_DIR / "caregivers"
    if not caregiver_dir.exists():
        return []

    docs: list[Document] = []
    for path in caregiver_dir.glob("*.json"):
        data = _load_json(path)
        if not data or not isinstance(data, dict):
            continue
        caregiver_id = data.get("caregiver_id") or path.stem
        for pref in data.get("semantic_preferences") or []:
            docs.append(
                Document(
                    text=f"[Always] {pref}",
                    metadata={
                        "type": "caregiver_preference",
                        "category": "caregiver",
                        "source": "caregiver_profile",
                        "caregiver_id": caregiver_id,
                    },
                )
            )
        for note in data.get("notes") or []:
            if not isinstance(note, dict):
                continue
            content = note.get("content")
            if not content:
                continue
            day = note.get("day")
            label = day or "Always"
            docs.append(
                Document(
                    text=f"[{label}] {content}",
                    metadata={
                        "type": "caregiver_note",
                        "category": "caregiver",
                        "source": "caregiver_profile",
                        "caregiver_id": caregiver_id,
                    },
                )
            )
    return docs


def ingest_data(output_dir: str = "data") -> VectorStoreIndex:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    documents: list[Document] = []
    documents.extend(list(_iter_therapy_docs()))
    documents.extend(list(_iter_patient_docs()))
    documents.extend(list(_iter_caregiver_docs()))

    db = chromadb.PersistentClient(path=str(Path(output_dir) / "chroma_db"))
    try:
        db.delete_collection(name=COLLECTION_NAME)
        print(f"Collezione '{COLLECTION_NAME}' resettata.")
    except Exception:
        pass

    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Indicizzazione di {len(documents)} documenti totali...")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("✅ Ingestion completata con successo.")
    return index


if __name__ == "__main__":
    print("Inizio fase di ingestion dati...")
    ingest_data()
    print("Ingestion dati completata.")
    print("\nRicorda di avere Ollama in esecuzione (nomic-embed-text) prima di eseguire questo script.")
