# 5. Gestione dei Conflitti

## Requisito Fondamentale
Le specifiche del progetto richiedono: *"Il sistema deve identificare i conflitti ma NON deve risolverli autonomamente."*

Abbiamo implementato due livelli di controllo distinti.

## 1. Conflitti Temporali (Deterministici)
Gestiti dal codice Python (`src/knowledge_manager.py`).
*   **Algoritmo:** Calcola l'intervallo `[Start, End]` della nuova attività. Scansiona tutte le attività esistenti nel giorno target.
*   **Logica:** Se `(New.Start < Existing.End) AND (New.End > Existing.Start)`, c'è sovrapposizione.
*   **Output:** Genera un warning: *"ATTENZIONE: Conflitto temporale con 'Pranzo' (12:00-12:30)"*.

## 2. Conflitti Semantici (Probabilistici)
Gestiti dall'LLM (`check_semantic_conflict` in `src/main.py`).
*   **Workflow:**
    1.  Recuperiamo le condizioni mediche e le preferenze del paziente dal profilo.
    2.  Creiamo un prompt specifico per un LLM "Analista":
        > "Il paziente ha DIABETE. L'utente vuole aggiungere TORTA. C'è conflitto?"
    3.  Se l'LLM risponde "SÌ", blocchiamo l'azione con un "AI SAFETY ALERT".

## Interazione con l'Utente
In entrambi i casi, l'agente:
1.  Sospende l'azione.
2.  Mostra l'avviso all'utente.
3.  Chiede esplicitamente: *"Vuoi confermare comunque (forzando) o annullare?"*.
