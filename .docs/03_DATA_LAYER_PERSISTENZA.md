# 3. Data Layer e Persistenza

## Il `KnowledgeManager`
Il cuore logico del sistema è la classe `KnowledgeManager` (`src/knowledge_manager.py`). Questo componente agisce come un'interfaccia CRUD (Create, Read, Update, Delete) verso il file system.

## Organizzazione dei Dati
I dati sono salvati in formato JSON per garantire leggibilità umana e facilità di debug. La struttura è gerarchica:

*   `data/patients/{id}.json`: Contiene i dati anagrafici, condizioni mediche, preferenze e abitudini del paziente.
*   `data/caregivers/{id}.json`: Contiene le preferenze dell'operatore (es. "mattina = 08:00").
*   `data/therapies/{id}.json`: Contiene il calendario effettivo delle attività.

## Modello Dati (Pydantic)
Usiamo Pydantic per validare rigorosamente i dati prima del salvataggio.

```python
class Activity(BaseModel):
    activity_id: str
    name: str
    description: str
    day_of_week: List[str]
    time: str
    dependencies: List[str] = []
    # ...
```

## Persistenza Atomica
Ogni volta che un'azione viene confermata dall'utente (`confirm_action`), il `KnowledgeManager`:
1.  Aggiorna l'oggetto in memoria RAM.
2.  Scrive immediatamente il dump JSON su disco.
3.  Invia il frammento a ChromaDB per l'indicizzazione.

Questo garantisce che se il sistema viene riavviato, lo stato è preservato perfettamente.
