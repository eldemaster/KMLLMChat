import sys
import logging
import argparse
import asyncio
import json
from typing import List, Dict, Any, AsyncGenerator
from pathlib import Path

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, Document
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

# Importiamo il nostro cervello logico e i modelli
from src.knowledge_manager import KnowledgeManager
from src.models import Activity
from src.logging_utils import setup_logger

# --- CONFIGURAZIONE ---
MODEL_NAME = "llama3.1" 
DB_DIR = "data"

# Inizializziamo il Knowledge Manager e Logger
km = KnowledgeManager()
logger = setup_logger("cli", "cli")

# --- RAG HELPER ---
def get_rag_index():
    """Restituisce l'indice vettoriale (singleton-like)."""
    if hasattr(get_rag_index, "index"):
        return get_rag_index.index
        
    db_path = Path(DB_DIR) / "chroma_db"
    if not db_path.exists():
        return None
        
    db_client = chromadb.PersistentClient(path=str(db_path))
    # Usiamo get_or_create per evitare errori se non esiste ancora
    chroma_collection = db_client.get_or_create_collection("patient_therapies")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    get_rag_index.index = VectorStoreIndex.from_vector_store(vector_store)
    return get_rag_index.index

def check_semantic_conflict(name: str, description: str) -> str | None:
    """
    Usa l'LLM per verificare se l'attività proposta viola vincoli medici o note
    presenti nel profilo del paziente (Conflitto Semantico Indiretto).
    Ritorna una stringa di avviso se c'è conflitto, altrimenti None.
    """
    p_profile = km.patient_profile
    if not p_profile:
        return None

    # Raccogliamo i vincoli noti
    constraints = []
    if p_profile.medical_conditions:
        constraints.extend(p_profile.medical_conditions)
    if p_profile.notes:
        constraints.extend(p_profile.notes)
    
    if not constraints:
        return None

    constraints_text = "\n- ".join(constraints)
    
    prompt = f"""Sei un supervisore medico. Analizza se la NUOVA ATTIVITÀ entra in conflitto con i VINCOLI del paziente.\n\nVINCOLI PAZIENTE:\n- {constraints_text}\n\nNUOVA ATTIVITÀ:\nNome: {name}\nDescrizione: {description}\n\nTASK:\nC'è un conflitto semantico evidente (es. il paziente deve digiunare ma l'attività prevede cibo/acqua)?\nRispondi SOLO con "SI: [spiegazione]" se c'è conflitto, o "NO" se non c'è.\n"""
    
    # Usiamo l'LLM globale configurato in Settings
    response = Settings.llm.complete(prompt).text.strip()
    
    if response.upper().startswith("SI"):
        return response # Ritorna la spiegazione del conflitto
    return None

# --- TOOL DEFINITIONS ---

def save_knowledge_tool(category: str | dict, content: str | dict) -> str:
    """
    Salva una nuova informazione (nota) sul paziente o caregiver.
    Aggiorna sia il JSON (persistenza) che il DB Vettoriale (RAG).
    """
    # Robustezza
    if isinstance(content, dict):
        content = content.get("text", content.get("content", str(content)))
    if isinstance(category, dict):
        category = category.get("value", category.get("category", str(category)))
    
    # Casting finale a stringa
    content = str(content)
    category = str(category)
    
    # 1. Salva nel JSON tramite KM
    km_result = km.save_knowledge_note(category, content)
    
    # 2. Aggiorna RAG (ChromaDB)
    index = get_rag_index()
    if index:
        # Creiamo un documento LlamaIndex
        doc = Document(
            text=content, 
            metadata={
                "category": category, 
                "source": "chat_extraction",
                "type": "extracted_knowledge"
            }
        )
        index.insert(doc)
        logger.info(f"Nota indicizzata in ChromaDB: {content}")
        return f"{km_result} e indicizzata per ricerche future."
    else:
        return f"{km_result} (ATTENZIONE: RAG non aggiornato, DB non trovato)."

def add_activity_tool(name: str, description: str, days: List[str], time: str) -> str:
    """Aggiunge una nuova attività."""
    try:
        # 0. Check Semantico (AI Guardrail)
        semantic_warning = check_semantic_conflict(name, description)
        if semantic_warning:
            logger.warning(f"Blocco semantico attivato: {semantic_warning}")
            return f"BLOCCO SICUREZZA: {semantic_warning} (Se vuoi forzare l'inserimento, specifica esplicitamente che ignori il vincolo)."

        import time as t
        act_id = f"act_{{int(t.time())}}"
        new_activity = Activity(
            activity_id=act_id,
            name=name,
            description=description,
            day_of_week=days,
            time=time,
            dependencies=[]
        )
        result = km.add_activity(new_activity)
        return result
    except Exception as e:
        return f"Errore nell'aggiunta dell'attività: {str(e)}"

def get_schedule_tool(day: str) -> str:
    """Restituisce la lista delle attività per un giorno."""
    activities = km.get_activities_by_day(day)
    if not activities:
        return f"Nessuna attività prevista per {day}."
    output = f"Programma per {day}:\n"
    for act in activities:
        output += f"- [{act.time}] {act.name}: {act.description}\n"
    return output

def consult_guidelines_tool(query: str) -> str:
    """Cerca nel PDF/RAG."""
    index = get_rag_index()
    if not index:
        return "Errore: DB Vettoriale non trovato."
        
    # Creiamo il query engine al volo
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query)
    return str(response)

# --- LITE AGENT LOGIC ---

def build_system_prompt():
    """Costruisce il prompt di sistema iniettando il contesto dinamico."""
    
    # Recupera i profili dal Knowledge Manager
    p_profile = km.patient_profile
    c_profile = km.caregiver_profile
    
    context_str = ""
    if c_profile:
        context_str += f"\n\nCONTESTO CAREGIVER (Chi ti scrive):\n- Nome: {c_profile.name}\n- Ruolo: {c_profile.name}" # fix role
        if c_profile.semantic_preferences:
            context_str += f"\n- Preferenze Linguistiche: {', '.join(c_profile.semantic_preferences)}"
            
    if p_profile:
        context_str += f"\n\nCONTESTO PAZIENTE (Di chi si parla):\n- Nome: {p_profile.name}\n- Condizioni Mediche: {', '.join(p_profile.medical_conditions)}"
        if p_profile.preferences:
            context_str += f"\n- Abitudini/Note: {', '.join(p_profile.preferences)}"
        # Aggiungi le note dinamiche
        if p_profile.notes:
            context_str += f"\n- Note Recenti: {', '.join(p_profile.notes[-3:])}" # Ultime 3 note

    base_prompt = f"""Sei KMChat, un assistente medico intelligente per la gestione delle terapie.
Il tuo compito è aiutare il caregiver a gestire le attività del paziente e a consultare le linee guida.
{context_str}

REGOLE IMPORTANTI:
1. Analizza attentamente la richiesta dell'utente.
2. Se la richiesta può essere soddisfatta da uno dei TUOI TOOL, genera il JSON per chiamare quel tool.
3. Se la richiesta è una conversazione generale (saluti, ringraziamenti), rispondi direttamente.
4. Ogni output DEVE essere un JSON valido, non aggiungere altro testo prima o dopo il JSON.

HAI A DISPOSIZIONE QUESTI TOOL (DESCRIZIONE PER TE):
- `get_schedule(day: str)`: Restituisce la lista delle attività previste per un giorno specifico (es. "Lunedì", "Martedì"). Usa questo per sapere cosa c'è in programma.
- `add_activity(name: str, description: str, days: List[str], time: str)`: Aggiunge una nuova attività alla terapia. Richiede nome, descrizione, lista dei giorni (es. ["Lunedì", "Venerdì"]) e orario (es. "08:00" o "09:00-09:30").
- `consult_guidelines(query: str)`: Cerca informazioni dettagliate o regole mediche nelle linee guida. Usa questo quando l'utente chiede informazioni su procedure, farmaci o contesto medico generale.
- `save_knowledge(category: str, content: str)`: Salva una nuova informazione appresa dalla conversazione. 'category' può essere 'patient' (es. salute, abitudini) o 'caregiver' (es. preferenze). Usa questo quando l'utente ti dice qualcosa di nuovo e rilevante che dovresti ricordare.

FORMATO DI RISPOSTA (SEMPRE IN JSON):

CASO 1: Chiamare un tool.
{{
  "action": "call_tool",
  "tool_name": "NOME_DEL_TOOL",
  "arguments": {{ "nome_argomento1": "valore1", "nome_argomento2": "valore2", ... }}
}}

CASO 2: Rispondere direttamente all'utente.
{{
  "action": "reply",
  "message": "Il tuo messaggio qui."
}}

ESEMPI DI INTERAZIONE (Utente -> KMChat):

Utente: "Ciao"
KMChat (JSON):
{{
  "action": "reply",
  "message": "Ciao! Sono qui per aiutarti a gestire le terapie. Come posso esserti utile oggi?"
}}

Utente: "Il paziente oggi ha la febbre alta." (Informazione nuova rilevante)
KMChat (JSON):
{{
  "action": "call_tool",
  "tool_name": "save_knowledge",
  "arguments": {{ "category": "patient", "content": "Il paziente ha la febbre alta in data odierna." }}
}}

Utente: "Cosa c'è in programma per domani?"
KMChat (JSON):
{{
  "action": "call_tool",
  "tool_name": "get_schedule",
  "arguments": {{ "day": "Mercoledì" }}
}}

Utente: "Aggiungi fisioterapia il Giovedì dalle 14:00 alle 15:00, descrizione esercizi per la schiena."
KMChat (JSON):
{{
  "action": "call_tool",
  "tool_name": "add_activity",
  "arguments": {{
    "name": "Fisioterapia",
    "description": "Esercizi per la schiena",
    "days": ["Giovedì"],
    "time": "14:00-15:00"
  }}
}}

Utente: "Parlami dell'Assunzione Aulin"
KMChat (JSON):
{{
  "action": "call_tool",
  "tool_name": "consult_guidelines",
  "arguments": {{ "query": "Assunzione Aulin" }}
}}
"""
    return base_prompt

async def run_lite_agent(llm, user_input: str) -> AsyncGenerator[str, None]:
    # 1. Chiedi al LLM cosa fare
    system_prompt = build_system_prompt()
    prompt = f"{system_prompt}\n\nUtente: {user_input}\nKMChat (JSON):"
    try:
        response_gen = llm.stream_complete(prompt)
        
        full_response_text = ""
        for token in response_gen:
            full_response_text += token.delta or ""
            
        json_str = full_response_text.strip()
        # Clean markdown e prefissi
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx:end_idx+1]
        
        data = json.loads(json_str)
        
        action = data.get("action")
        
        if action == "reply":
            msg = data.get("message")
            if isinstance(msg, dict) and "text" in msg:
                msg = msg["text"]
            yield str(msg)
        
        elif action == "call_tool":
            tool_name = data.get("tool_name")
            args = data.get("arguments", {})
            logger.info(f"Esecuzione tool: {tool_name} con args {args}")
            print(f"[DEBUG] Args Tool: {args}") # Debug visibile
            
            tool_result = ""
            if tool_name == "get_schedule":
                tool_result = get_schedule_tool(**args)
            elif tool_name == "add_activity":
                tool_result = add_activity_tool(**args)
            elif tool_name == "consult_guidelines":
                tool_result = consult_guidelines_tool(**args)
            elif tool_name == "save_knowledge":
                tool_result = save_knowledge_tool(**args)
            else:
                tool_result = f"Tool {tool_name} non trovato."
            
            # 2. Risposta finale
            final_prompt = f"Sei KMChat. Rispondi all'utente basandoti sull'output del tool.\n\nUtente: {user_input}\nTool Output: {tool_result}\nRisposta finale (breve):"
            
            # Qui usiamo lo streaming VERO verso l'utente
            final_resp_gen = llm.stream_complete(final_prompt)
            for token in final_resp_gen:
                yield token.delta or ""
            
        else:
            yield f"Azione sconosciuta: {action}"

    except json.JSONDecodeError as e:
        logger.error(f"Errore parsing JSON LLM: {e}. Risposta grezza: {full_response_text if 'full_response_text' in locals() else 'N/A'}")
        yield f"Errore interno (JSON non valido). Controlla i log per i dettagli."
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.exception(f"Errore agente: {e}")
        yield f"Errore sistema: {e}"

# --- MAIN ---

async def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="KMChat CLI Lite")
    parser.add_argument("--test-prompt", type=str, help="Esegui un singolo prompt di test ed esci")
    args = parser.parse_args()

    # Inizializza LLM
    print(f"🔌 Connessione a {MODEL_NAME}...")
    llm = Ollama(model=MODEL_NAME, request_timeout=120.0, temperature=0.1)
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    if args.test_prompt:
        print(f"\n🧪 Modalità Test: {args.test_prompt}")
        full_response_test = ""
        async for chunk in run_lite_agent(llm, args.test_prompt):
            full_response_test += str(chunk)
        print(f"\nKMChat: {full_response_test}")
        return

    print("\n✅ KMChat Agente Attivo (Lite Mode)! (Digita 'exit' per uscire)")
    print("-" * 50)

    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nCaregiver: ")
            user_input = user_input.strip()
            
            if user_input.lower() in ["exit", "quit", "esci"]:
                break
            if not user_input:
                continue

            print("\nKMChat: ", end="", flush=True)
            async for chunk in run_lite_agent(llm, user_input):
                print(str(chunk), end="", flush=True)
            print() # Nuova riga alla fine della risposta

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("Errore chat: %s", e)
            print(f"\nErrore: {e}")

if __name__ == "__main__":
    asyncio.run(main())