# 2. Agente LLM e Routing (Pure AI)

## Il Ruolo di `main.py`
Il file `src/main.py` funge da cervello e orchestratore. Non esegue direttamente le azioni, ma decide *quale* strumento utilizzare in base all'input dell'utente.

## Strategia "Pure LLM"
Il sistema usa un approccio interamente LLM‑based per l'estrazione dei dati, basato su **Function Calling simulato via JSON**. Non sono previste euristiche o regex di parsing nella pipeline di routing.

### Il Ciclo di Ragionamento (Agent Loop)
1.  **Input:** L'utente scrive "Aggiungi fisioterapia lunedì alle 10".
2.  **System Prompt Dinamico:** Costruiamo un prompt che include:
    *   La cronologia recente.
    *   La definizione degli strumenti (`add_activity`, `get_schedule`, ecc.).
    *   **Few-Shot Examples:** Esempi concreti di come trasformare frasi in JSON.
3.  **Generazione:** L'LLM analizza la richiesta e genera un oggetto JSON:
    ```json
    {
      "action": "call_tool",
      "tool_name": "add_activity",
      "arguments": {
        "name": "Fisioterapia",
        "day": ["Lunedì"],
        "time": "10:00"
      }
    }
    ```
4.  **Esecuzione:** Il codice Python parsa il JSON, convalida la presenza degli argomenti ed esegue la funzione corrispondente.

## Gestione Robustezza
Per mitigare i limiti dei modelli piccoli:
*   **Strict Mode (prompt):** Regole più stringenti su formati, conferme e mancanze di campi.
*   **Stop Sequences:** Ollama si ferma appena generato il JSON, evitando output prolissi.
*   **Fallback sugli Argomenti:** Se mancano campi obbligatori (es. orario), il sistema chiede chiarimenti all'utente invece di eseguire.

## Routing del Modello
La generazione principale usa sempre il modello SMART, evitando instradamenti euristici. Il modello piccolo viene usato nei test comparativi con prompt più rigido.
