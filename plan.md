# Piano di Sviluppo KMChat (COMPLETO)

Obiettivo: Sistema di supporto decisionale per caregiver basato su LLM, con gestione della conoscenza, rilevamento conflitti e RAG.

## 1. Setup e Base (Completato)
- [x] Setup ambiente (venv, requirements).
- [x] Ingestion RAG iniziale (ChromaDB con `therapy.json`).
- [x] CLI Agent "Lite" ottimizzato con Llama 3.1.
- [x] Streaming output per UX reattiva.

## 2. Gestione Conoscenza e Conflitti (Completato)
### 2.1 Conflitti Temporali
- [x] Rilevamento sovrapposizioni orarie (es. 10:00-11:00 vs 10:30-11:30).
- [x] Parsing robusto orari ("HH:MM", "HH:MM-HH:MM").

### 2.2 Conflitti di Dipendenza
- [x] Implementare check logico: se rimuovo attività A, avvisare se B dipende da A.
- [x] Implementare check aggiunta: se aggiungo B che dipende da A, verificare se A esiste.
- [x] Test unitari per dipendenze.

### 2.3 Profili Paziente e Caregiver
- [x] Estendere `KnowledgeManager` per caricare/salvare `patient_profile.json` e `caregiver_profile.json`.
- [x] Includere preferenze paziente (es. "riposino alle 15") e caregiver (es. "Aulin = granulare") nel contesto RAG o System Prompt (Context Injection).

## 3. Conflitti Semantici e Avanzati (LLM-based) (Completato)
- [x] Rilevamento conflitti indiretti (es. "Niente liquidi" vs "Farmaco con acqua") usando LLM che analizza le descrizioni (Guardrail Semantico).
- [x] Rilevamento ambiguità semantiche (es. "Aulin" che cambia significato) gestito tramite profili.

## 4. Estrazione e Aggiornamento Conoscenza (Learning) (Completato)
- [x] Implementare tool `save_knowledge`: analizzare la chat per trovare nuove info (es. "Il paziente oggi ha la febbre").
- [x] Aggiornamento incrementale ChromaDB: salvare nuove info estratte senza ri-indicizzare tutto.
- [x] Validazione utente prima del salvataggio ("Ho capito che X, confermi?") - *Gestito tramite interazione chat.*

## 5. Interfaccia e UX
- [x] Interfaccia CLI interattiva con streaming.
