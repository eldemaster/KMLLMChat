# 1. Architettura Ibrida "Neuro-Simbolica"

## Visione d'Insieme
Il progetto KMChat non si affida ciecamente all'Intelligenza Artificiale Generativa per ogni compito. Abbiamo scelto un approccio **Ibrido (Neuro-Simbolico)** per bilanciare la flessibilità del linguaggio naturale con la rigidità necessaria in ambito medico e nella gestione della terapia.

### Perché questa scelta?
Le specifiche del progetto richiedono che il sistema non risolva conflitti automaticamente e garantisca la coerenza dei dati. Gli LLM puri soffrono di "allucinazioni" e non sono affidabili per calcoli precisi (es. sovrapposizione oraria), quindi serve una componente simbolica deterministica.

## Schema Architetturale

```mermaid
graph TD
    User[Utente/Caregiver] -->|Chat| Router[Agente LLM (Ollama)]
    
    subgraph "Componente Neurale (Cervello)"
        Router -->|Analisi Intento| Prompt[System Prompt]
        Router -->|Ragionamento Medico| SemanticCheck[Check Semantico]
    end
    
    subgraph "Componente Simbolica (Braccio)"
        Router -->|JSON Command| KM[Knowledge Manager (Python)]
        KM -->|Validazione| ConflictEngine[Motore Conflitti Temporali]
        KM -->|CRUD| JSON_DB[(File JSON)]
    end
    
    subgraph "Memoria Semantica"
        KM -->|Indexing| Chroma[ChromaDB (Vector Store)]
        Chroma -->|Retrieval| Router
    end
```

## Stack Tecnologico
*   **LLM Engine:** Ollama (modelli principali `kmchat-14b` e `kmchat-8b`). Esecuzione locale per privacy.
*   **Framework:** Python 3.13 puro per la logica, LlamaIndex per l'interfaccia LLM.
*   **Database:**
    *   *Strutturato:* JSON Files (per Terapie e Profili).
    *   *Vettoriale:* ChromaDB (per attività, note e profili indicizzati).
