# 6. Knowledge Extraction (Estrazione Conoscenza)

## Active Listening
Il sistema non si limita a eseguire ordini, ma ascolta per arricchire la sua base di conoscenza.

## Categorie di Informazione
Il tool `save_knowledge` categorizza i dati in:
*   **Conditions:** Dati medici critici ("Ha il diabete", "Allergico alle noci").
*   **Preferences:** Preferenze soggettive ("Odia svegliarsi presto").
*   **Habits:** Abitudini ricorrenti ("Fa una passeggiata dopo pranzo").
*   **Caregiver:** Definizioni linguistiche ("Per me 'sera' inizia alle 21").

## Meccanismo di Estrazione
Abbiamo istruito l'LLM (nel Prompt di Sistema) a riconoscere pattern discorsivi:
*   *"Il paziente ha..."* -> Trigger per `save_knowledge(conditions)`.
*   *"Preferisce..."* -> Trigger per `save_knowledge(preferences)`.

## Chiusura del Cerchio (Feedback Loop)
1.  Estratta l'informazione, viene salvata nel JSON del paziente o del caregiver.
2.  Viene anche indicizzata in ChromaDB.
3.  Alla prossima interazione, il RAG o il Semantic Check useranno questa nuova conoscenza per validare le azioni future (es. impedire di aggiungere dolci se abbiamo appena imparato che è diabetico).
