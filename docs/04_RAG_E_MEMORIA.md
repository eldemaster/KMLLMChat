# 4. RAG (Retrieval-Augmented Generation)

## Obiettivo
Fornire al chatbot una "memoria semantica" per rispondere a domande non strutturate e recuperare informazioni dai dati di progetto (attività, profili, note).

## Tecnologia: ChromaDB + LlamaIndex
Utilizziamo **ChromaDB** come database vettoriale locale.
*   **Embedding Model:** `nomic-embed-text`. Trasforma il testo in vettori numerici.
*   **Retrieval:** Quando l'utente fa una domanda, cerchiamo i 3 "chunk" di testo più simili nel database.

## Flusso RAG
1.  **Ingestione (`src/ingest_data.py`):** I dati JSON (terapie, profili paziente/caregiver, note) vengono trasformati in documenti e indicizzati.
2.  **Query:**
    *   Utente: "Quali sono le controindicazioni per l'Aulin?"
    *   Sistema: Calcola il vettore della domanda.
    *   ChromaDB: Restituisce i chunk pertinenti dai dati indicizzati.
3.  **Augmentation:** Il testo recuperato viene inserito nel Prompt di Sistema ("CONTESTO RECUPERATO: ...").
4.  **Generation:** L'LLM usa quel contesto per rispondere in modo informato.

## Integrazione con Knowledge Extraction
Quando il sistema apprende una nuova informazione (es. "Il paziente odia il rumore"), questa viene salvata in ChromaDB. In futuro, se l'utente chiederà "Cosa devo sapere sull'ambiente per il paziente?", il RAG recupererà la nota pertinente.
