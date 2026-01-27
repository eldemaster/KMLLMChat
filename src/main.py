import sys
import os
import logging
import argparse
import asyncio
import json
import re
from datetime import date, timedelta
from typing import List, Dict, Any, AsyncGenerator
from pathlib import Path

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, Document
try:
    from llama_index.core import MetadataFilters, ExactMatchFilter
except Exception:
    MetadataFilters = None
    ExactMatchFilter = None
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

# Importiamo il nostro cervello logico e i modelli
from src.knowledge_manager import KnowledgeManager
from src.models import Activity
from src.logging_utils import setup_logger

# --- CONFIGURAZIONE ---
# Modello Veloce: Per comandi diretti, JSON formatting, CRUD
MODEL_FAST = "kmchat-14b"
# Modello Smart: Per ragionamento semantico, analisi conflitti, empatia
# (In locale possiamo usare lo stesso se non ne abbiamo altri, o uno più grande tipo 'qwen2.5:32b' se hardware permette)
MODEL_SMART = "kmchat-14b"

DB_DIR = "data"
HISTORY_FILE = "session_history.md"

# Tool registry
VALID_TOOLS = [
    "get_schedule",
    "get_schedule_week",
    "get_patient_info",
    "get_caregiver_info",
    "add_activity",
    "modify_activity",
    "delete_activity",
    "consult_guidelines",
    "save_knowledge",
    "switch_context",
    "get_context",
    "confirm_action",
    "cancel_action",
    "debug_rag",
]

TOOL_ARG_WHITELIST = {
    "get_schedule": {"day", "date"},
    "get_schedule_week": set(),
    "get_patient_info": {"category"},
    "get_caregiver_info": {"category"},
    "add_activity": {"name", "description", "days", "time", "duration_minutes", "dependencies", "force", "valid_from", "valid_until", "duration_days"},
    "modify_activity": {"old_name", "day", "new_name", "new_description", "new_time", "new_days", "duration_minutes", "force", "valid_from", "valid_until", "duration_days"},
    "delete_activity": {"name", "day", "force"},
    "consult_guidelines": {"query"},
    "save_knowledge": {"category", "content", "day"},
    "switch_context": {"patient_id", "caregiver_id"},
    "get_context": set(),
    "confirm_action": set(),
    "cancel_action": set(),
    "debug_rag": {"query"},
}

def _normalize_duration_days(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None

def _normalize_duration_minutes(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None

def _parse_time_to_minutes(value: str) -> int | None:
    try:
        hours, minutes = map(int, value.split(":"))
        return hours * 60 + minutes
    except Exception:
        return None

def _normalize_time_and_duration(time_str: str | None, duration_minutes: int | None) -> tuple[str | None, int | None]:
    if not time_str:
        return time_str, duration_minutes
    cleaned = str(time_str).strip()
    if "-" not in cleaned:
        return cleaned, duration_minutes

    start_str, end_str = [part.strip() for part in cleaned.split("-", 1)]
    if duration_minutes is None:
        start_min = _parse_time_to_minutes(start_str)
        end_min = _parse_time_to_minutes(end_str)
        if start_min is not None and end_min is not None and end_min > start_min:
            duration_minutes = end_min - start_min
    return start_str, duration_minutes

def _expand_days_for_duration(days: List[str], duration_days: int | None) -> List[str]:
    if not duration_days or not days:
        return days
    if len(days) != 1:
        return days
    base_day = days[0]
    day_order = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
    if base_day not in day_order:
        return days
    start_idx = day_order.index(base_day)
    total = duration_days + 1
    return [day_order[(start_idx + i) % 7] for i in range(total)]

def _apply_duration(valid_from: str | None, valid_until: str | None, duration_days: int | None) -> tuple[str | None, str | None]:
    """Calcola valid_until basandosi sulla durata in giorni se specificata."""
    if duration_days is None:
        return valid_from, valid_until
    
    start_date = date.today()
    if valid_from:
        try:
            start_date = date.fromisoformat(valid_from)
        except ValueError:
            pass # Usa oggi se il formato è invalido o non specificato correttamente
    else:
        # Se c'è una durata ma non una data di inizio, assumiamo che inizi oggi
        valid_from = start_date.isoformat()
        
    end_date = start_date + timedelta(days=duration_days)
    valid_until = end_date.isoformat()
    return valid_from, valid_until

def _shorten_semantic_warning(text: str, max_len: int = 180) -> str:
    if not text:
        return text
    cleaned = " ".join(str(text).strip().split())
    if ":" in cleaned:
        prefix, rest = cleaned.split(":", 1)
        if prefix.strip().upper().startswith(("SI", "SÌ")):
            cleaned = f"SÌ: {rest.strip()}"
    for sep in (".", "!", "?"):
        idx = cleaned.find(sep)
        if idx != -1 and idx < max_len:
            cleaned = cleaned[: idx + 1]
            break
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3].rstrip() + "..."
    return cleaned

def _ensure_patient_context() -> str | None:
    if not km.current_patient_id or not km.patient_profile:
        return "Nessun paziente selezionato. Usa switch_context(patient_id, caregiver_id)."
    return None

# Inizializziamo il Knowledge Manager e Logger
km = KnowledgeManager(auto_discover=False)
logger = setup_logger("cli", "cli")
PENDING_ACTION: Dict[str, Any] | None = None

# --- SESSION MANAGER (SHARED MEMORY) ---
class SessionManager:
    def __init__(self, history_file: str):
        self.file_path = Path(history_file)
        if not self.file_path.exists():
            self.file_path.write_text("# KMChat Session History\n\n", encoding="utf-8")

    def append_interaction(self, role: str, content: str):
        """Salva l'interazione nel file condiviso."""
        timestamp = "" # Possiamo aggiungere timestamp se serve
        entry = f"\n**{role}**: {content}\n"
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def get_recent_history(self, limit_chars: int = 2000) -> str:
        """Recupera gli ultimi N caratteri per dare contesto all'LLM."""
        if os.getenv("KMCHAT_DISABLE_HISTORY") == "1":
            return ""
        if not self.file_path.exists(): return ""
        text = self.file_path.read_text(encoding="utf-8")
        if len(text) > limit_chars:
            return "...(cronologia precedente troncata)...\n" + text[-limit_chars:]
        return text

session = SessionManager(HISTORY_FILE)

# --- RAG HELPER ---
def get_rag_index():
    """Restituisce l'indice vettoriale (singleton-like)."""
    if hasattr(get_rag_index, "index"):
        return get_rag_index.index
        
    db_path = Path(DB_DIR) / "chroma_db"
    if not db_path.exists():
        db_path.mkdir(parents=True, exist_ok=True)
        
    db_client = chromadb.PersistentClient(path=str(db_path))
    chroma_collection = db_client.get_or_create_collection("patient_therapies")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    get_rag_index.index = VectorStoreIndex.from_vector_store(vector_store)
    return get_rag_index.index

def reset_rag_index() -> None:
    """Reset cached RAG index after re-ingest or collection changes."""
    if hasattr(get_rag_index, "index"):
        delattr(get_rag_index, "index")

def get_rag_context(query: str, patient_name: str | None, caregiver_name: str | None) -> str:
    if os.getenv("KMCHAT_DISABLE_RAG_CONTEXT") == "1":
        return ""
    if not query:
        return ""
    try:
        index = get_rag_index()
        retriever = index.as_retriever(similarity_top_k=3)
        scoped_query = query
        if patient_name:
            scoped_query = f"{scoped_query}\nPatient: {patient_name}"
        if caregiver_name:
            scoped_query = f"{scoped_query}\nCaregiver: {caregiver_name}"
        results = retriever.retrieve(scoped_query)
        if not results:
            return "Nessuna informazione specifica trovata nei documenti."
        snippets = []
        for res in results:
            text = res.node.text.replace("\n", " ").strip()
            snippets.append(f"- {text}")
        return "\n".join(snippets)
    except Exception:
        return "Errore nel recupero delle informazioni."

def debug_rag_tool(query: str = None) -> str:
    if not query or not str(query).strip():
        return "Errore: specifica una query per il debug RAG."
    try:
        index = get_rag_index()
        retriever = index.as_retriever(similarity_top_k=3)
        results = retriever.retrieve(str(query))
        if not results:
            return "RAG DEBUG: nessun risultato."
        lines = ["RAG DEBUG (top 3):"]
        for res in results:
            text = res.node.text.replace("\n", " ").strip()
            meta = res.node.metadata or {}
            meta_bits = []
            for key in ("type", "source", "patient_id", "caregiver_id", "activity_id", "category"):
                if key in meta:
                    meta_bits.append(f"{key}={meta[key]}")
            meta_str = f" | meta: {', '.join(meta_bits)}" if meta_bits else ""
            lines.append(f"- {text}{meta_str}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Errore nel debug RAG: {exc}"

def _index_activity_in_rag(activity: Activity, source: str) -> None:
    try:
        index = get_rag_index()
        text = (
            f"Attività: {activity.name}\n"
            f"Descrizione: {activity.description}\n"
            f"Giorni: {', '.join(activity.day_of_week)}\n"
            f"Orario: {activity.time}\n"
            f"Durata (minuti): {activity.duration_minutes if activity.duration_minutes is not None else 'Non specificata'}\n"
            f"Dipendenze: {', '.join(activity.dependencies) if activity.dependencies else 'Nessuna'}"
        )
        meta = {
            "type": "therapy_activity",
            "source": source,
            "activity_id": activity.activity_id,
            "patient_id": km.current_patient_id,
            "caregiver_id": km.current_caregiver_id,
        }
        if activity.valid_from:
            meta["valid_from"] = activity.valid_from
        if activity.valid_until:
            meta["valid_until"] = activity.valid_until
        index.insert(Document(text=text, metadata=meta))
    except Exception:
        pass

# --- TOOL ROUTING HELPERS ---
def _parse_action_string(text: str) -> Dict[str, Any] | None:
    """
    Parses strings like: tool_name(arg1=val1, arg2=[v1, v2])
    Used as fallback when LLM outputs text representation instead of JSON.
    """
    text = text.strip()
    # Basic matching: ToolName(...)
    match = re.search(r"^(\w+)\((.*)\)$", text, re.DOTALL)
    if not match:
        return None
        
    tool_name = match.group(1)
    if tool_name not in VALID_TOOLS:
        return None
        
    args_str = match.group(2)
    arguments = {}
    
    # Split by comma, ignoring commas inside brackets []
    tokens = []
    buffer = ""
    in_list = False
    
    for char in args_str:
        if char == '[':
            in_list = True
        elif char == ']':
            in_list = False
        
        if char == ',' and not in_list:
            tokens.append(buffer)
            buffer = ""
        else:
            buffer += char
    if buffer:
        tokens.append(buffer)
        
    for token in tokens:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        
        # Value parsing
        if v == "null": 
            val = None
        elif v.lower() == "true": 
            val = True
        elif v.lower() == "false": 
            val = False
        elif v.startswith("[") and v.endswith("]"):
            inner = v[1:-1]
            if not inner.strip():
                val = []
            else:
                # Handle list items (assuming string list)
                val = [x.strip().strip("'\"") for x in inner.split(",")]
        else:
            val = v.strip("'\"")
            
        arguments[k] = val
        
    return {"action": "call_tool", "tool_name": tool_name, "arguments": arguments}

def _extract_json_object(text: str) -> Dict[str, Any] | None:
    # 1. Cerca blocchi di codice Markdown
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    
    # 2. Cerca il primo oggetto JSON valido nel testo
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1:
        return None
    
    json_str = text[s : e + 1]
    
    # 3. Tentativo di pulizia per errori comuni LLM
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: prova a correggere virgolette singole (comune errore LLM) o True/False
        try:
            fixed_str = json_str.replace("'", '"').replace("True", "true").replace("False", "false")
            return json.loads(fixed_str)
        except:
            pass
    return None

def _normalize_tool_action(data: Dict[str, Any]) -> Dict[str, Any]:
    # Normalizzazione chiavi (Case Insensitive per sicurezza)
    data = {k.lower(): v for k, v in data.items()}
    
    # Mappatura Italiano -> Inglese
    mappings = {
        "azione": "action",
        "strumento": "tool_name",
        "tool": "tool_name",
        "parametri": "arguments",
        "argomenti": "arguments",
        "args": "arguments"
    }
    
    for it_key, en_key in mappings.items():
        if it_key in data and en_key not in data:
            data[en_key] = data.pop(it_key)

    if "reply" in data and "message" not in data:
        data["message"] = data.get("reply")
        data["action"] = "reply"

    # Se c'è 'action' ma non 'tool_name', spesso action È il tool_name
    action = data.get("action")
    if action in VALID_TOOLS:
        data["tool_name"] = action
        data["action"] = "call_tool"
    
    # Se abbiamo un tool_name valido, forziamo l'azione a 'call_tool'
    if data.get("tool_name") in VALID_TOOLS:
        data["action"] = "call_tool"
        
    return data

def _sanitize_tool_args(tool_name: str, args: Dict[str, Any], user_input: str | None = None) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    allowed = TOOL_ARG_WHITELIST.get(tool_name, set())
    filtered = {k: v for k, v in args.items() if k in allowed}

    if tool_name == "get_schedule":
        day = filtered.get("day")
        if isinstance(day, list):
            filtered["day"] = day[0] if day else None
        if isinstance(filtered.get("day"), str):
            filtered["day"] = filtered["day"].strip()
        if filtered.get("date") is not None:
            filtered["date"] = str(filtered["date"]).strip()
    elif tool_name == "add_activity":
        days = filtered.get("days")
        if isinstance(days, str):
            filtered["days"] = [days]
        elif days is None:
            filtered["days"] = []
        filtered["days"] = [d.strip() for d in filtered["days"] if isinstance(d, str) and d.strip()]
        if filtered.get("dependencies") is None:
            filtered["dependencies"] = []
        if isinstance(filtered.get("time"), str):
            filtered["time"] = filtered["time"].strip()
        filtered["time"], filtered["duration_minutes"] = _normalize_time_and_duration(
            filtered.get("time"),
            filtered.get("duration_minutes"),
        )
        if not filtered.get("description"):
            filtered["description"] = filtered.get("name", "")
        if "duration_minutes" in filtered:
            filtered["duration_minutes"] = _normalize_duration_minutes(filtered.get("duration_minutes"))
        if "duration_days" in filtered:
            filtered["duration_days"] = _normalize_duration_days(filtered.get("duration_days"))
    elif tool_name == "modify_activity":
        day = filtered.get("day")
        if isinstance(day, list):
            filtered["day"] = day[0] if day else None
        if isinstance(filtered.get("day"), str):
            filtered["day"] = filtered["day"].strip()
        new_days = filtered.get("new_days")
        if isinstance(new_days, str):
            filtered["new_days"] = [new_days]
        elif new_days is None:
            filtered["new_days"] = []
        filtered["new_days"] = [d.strip() for d in filtered.get("new_days", []) if isinstance(d, str) and d.strip()]
        if "new_time" in filtered:
            if isinstance(filtered.get("new_time"), str):
                filtered["new_time"] = filtered["new_time"].strip()
            filtered["new_time"], filtered["duration_minutes"] = _normalize_time_and_duration(
                filtered.get("new_time"),
                filtered.get("duration_minutes"),
            )
        if "duration_minutes" in filtered:
            filtered["duration_minutes"] = _normalize_duration_minutes(filtered.get("duration_minutes"))
        if "duration_days" in filtered:
            filtered["duration_days"] = _normalize_duration_days(filtered.get("duration_days"))
    elif tool_name == "delete_activity":
        day = filtered.get("day")
        if isinstance(day, list):
            filtered["day"] = day[0] if day else None
        if isinstance(filtered.get("day"), str):
            filtered["day"] = filtered["day"].strip()
    elif tool_name == "consult_guidelines":
        pass
    elif tool_name == "switch_context":
        pass

    return filtered

def _coerce_tool_call(llm_fast, user_input: str) -> Dict[str, Any] | None:
    prompt = (
        "Sei un router di tool molto rigido. Restituisci SOLO un JSON valido.\n"
        "Strumenti disponibili:\n"
        "- get_schedule(day)\n"
        "- add_activity(name, description, days, time, duration_minutes=None, dependencies=[], force=False)\n"
        "- modify_activity(old_name, day, new_name, new_description, new_time, new_days, duration_minutes=None, force=False)\n"
        "- delete_activity(name, day, force=False)\n"
        "- consult_guidelines(query)\n"
        "- save_knowledge(category, content, day=None)\n"
        "- get_patient_info(category)\n"
        "- get_caregiver_info(category)\n"
        "- switch_context(patient_id, caregiver_id)\n"
        "- confirm_action()\n"
        "- cancel_action()\n\n"
        "- debug_rag(query)\n\n"
        "Regole:\n"
        "1) Se l'utente chiede il programma di un giorno, usa get_schedule.\n"
        "2) Usa get_schedule_week SOLO se l'utente chiede la settimana intera.\n"
        "3) Se l'utente chiede condizioni/preferenze/abitudini/notes del paziente, usa get_patient_info.\n"
        "4) Se l'utente chiede info/notes del caregiver, usa get_caregiver_info.\n"
        "5) Se l'utente vuole cambiare paziente/caregiver, usa switch_context.\n"
        "6) Se l'utente dice \"salva questa informazione\" senza una pending action, chiedi quale informazione.\n"
        "7) Se l'utente risponde solo con \"no\" o \"annulla\" e non c'è un'azione in sospeso, rispondi \"Ok.\".\n"
        "8) Se l'utente dice \"per i prossimi N giorni\", usa duration_days=N nel tool di aggiunta/modifica.\n"
        "9) Se l'utente parla della propria routine (\"mia visita\", \"per me\"), usa save_knowledge con category caregiver.\n"
        "10) Se l'utente chiede note del paziente, usa get_patient_info con category notes.\n"
        "11) Se l'utente chiede note del caregiver, usa get_caregiver_info con category notes.\n"
        "12) Se l'utente chiede un debug RAG, usa debug_rag(query).\n"
        "13) Giorni ammessi: Lunedì, Martedì, Mercoledì, Giovedì, Venerdì, Sabato, Domenica (accento grave).\n"
        "14) Altrimenti, rispondi con {\"action\":\"reply\",\"message\":\"...\"}.\n\n"
        "Esempio modify_activity:\n"
        "{\"action\":\"call_tool\",\"tool_name\":\"modify_activity\",\"arguments\":{\"old_name\":\"Camminata\",\"day\":\"Lunedì\",\"new_name\":\"Cyclette al chiuso\",\"new_time\":\"18:00\"}}\n\n"
        "Esempio durata:\n"
        "{\"action\":\"call_tool\",\"tool_name\":\"add_activity\",\"arguments\":{\"name\":\"Controllo pressione\",\"days\":[\"Martedì\"],\"time\":\"09:00\",\"duration_days\":2}}\n\n"
        "Esempio caregiver:\n"
        "{\"action\":\"call_tool\",\"tool_name\":\"save_knowledge\",\"arguments\":{\"category\":\"caregiver\",\"content\":\"La mia visita abituale è alle 18:00\"}}\n\n"
        "Esempio note paziente:\n"
        "{\"action\":\"call_tool\",\"tool_name\":\"get_patient_info\",\"arguments\":{\"category\":\"notes\"}}\n\n"
        "Esempio note caregiver:\n"
        "{\"action\":\"call_tool\",\"tool_name\":\"get_caregiver_info\",\"arguments\":{\"category\":\"notes\"}}\n\n"
        f"Utente: {user_input}\n"
        "JSON:"
    )
    try:
        response = llm_fast.complete(prompt).text.strip()
    except Exception:
        return None
    data = _extract_json_object(response)
    if not data:
        return None
    if isinstance(data, dict) and not data:
        return {"action": "reply", "message": "Ok."}
    return _normalize_tool_action(data)

def _stage_action(tool_name: str, args: Dict[str, Any]) -> str:
    global PENDING_ACTION
    PENDING_ACTION = {"tool_name": tool_name, "arguments": args}
    return f"Azione in sospeso: {tool_name} con {args}. Scrivi 'conferma' per applicare o 'annulla' per annullare."

def _consume_pending_action() -> Dict[str, Any] | None:
    global PENDING_ACTION
    pending = PENDING_ACTION
    PENDING_ACTION = None
    return pending

def check_semantic_conflict(name: str, description: str) -> str | None:
    """
    Usa l'LLM SMART per verificare coerenza logica.
    """
    p_profile = km.patient_profile
    if not p_profile: return None

    constraints = []
    if p_profile.medical_conditions: constraints.extend(p_profile.medical_conditions)
    if p_profile.preferences: constraints.extend(p_profile.preferences)
    if p_profile.habits: constraints.extend(p_profile.habits)
    
    for note in p_profile.notes:
        prefix = f"[{note.day}] " if note.day else "[Sempre] "
        constraints.append(f"{prefix}{note.content}")
    
    if not constraints: return None
    constraints_text = "\n- ".join(constraints)
    
    prompt = f"""Ruolo: Controllo Coerenza Logica e Sicurezza Medica.
Compito: Verifica RIGOROSA se l'AZIONE PROPOSTA contraddice le REGOLE STABILITE (Condizioni Mediche, Preferenze).

REGOLE STABILITE:
- {constraints_text}

AZIONE PROPOSTA:
Nome: {name}
Descrizione: {description}

ANALISI DI SICUREZZA:
1. Se il paziente ha DIABETE/GLICEMIA e l'azione implica CIBO/ZUCCHERI/DOLCI -> È UN CONFLITTO.
2. Se c'è un'allergia e l'azione implica l'allergene -> È UN CONFLITTO.
3. Se l'azione viola un orario preferito -> È UN CONFLITTO.
4. Se c'è un vincolo tipo "non assumere liquidi" e l'azione implica acqua/liquidi/bevande -> È UN CONFLITTO.
5. Non considerare "farmaci/pillole generiche" un conflitto se le regole non citano farmaci o ingredienti specifici.
6. Se NON c'è una contraddizione esplicita con una regola, rispondi "NO". Non inventare rischi o dettagli mancanti.

Rispondi SOLO con:
- "SÌ: [Spiegazione breve, max 20 parole]" se c'è un rischio o contraddizione.
- "NO" se è sicuro o non ci sono informazioni sufficienti.
"""
    # Usiamo il modello SMART per il ragionamento
    llm_smart = Settings.llm  # Assumiamo che Settings.llm sia quello smart o riconfiguriamolo
    print(f"\n[SEMANTIC CHECK] Analyzing...")
    response = llm_smart.complete(prompt).text.strip()
    print(f"[SEMANTIC CHECK] LLM Response: {response}") # Log cruciale per debug
    
    # Parsing robusto: cerchiamo un "SI" o "SÌ" esplicito all'inizio
    # Se l'LLM dice "No, non c'è conflitto" o "Confermato", allora NON è un blocco.
    normalized_resp = response.strip().upper()
    if normalized_resp.startswith("SI") or normalized_resp.startswith("SÌ"):
        return response
    if normalized_resp.startswith("NO"):
        return None
    return None

# --- TOOL DEFINITIONS ---
def _execute_tool(tname: str, args: Dict[str, Any]) -> str:
    if tname == "get_schedule":
        return get_schedule_tool(**args)
    if tname == "get_schedule_week":
        return get_schedule_week_tool()
    if tname == "get_patient_info":
        return get_patient_info_tool(**args)
    if tname == "get_caregiver_info":
        return get_caregiver_info_tool(**args)
    if tname == "add_activity":
        return add_activity_tool(**args)
    if tname == "modify_activity":
        return modify_activity_tool(**args)
    if tname == "delete_activity":
        return delete_activity_tool(**args)
    if tname == "consult_guidelines":
        return consult_guidelines_tool(**args)
    if tname == "save_knowledge":
        return save_knowledge_tool(**args)
    if tname == "switch_context":
        return switch_context_tool(**args)
    if tname == "get_context":
        return get_context_tool()
    if tname == "confirm_action":
        return confirm_action_tool()
    if tname == "cancel_action":
        return cancel_action_tool()
    if tname == "debug_rag":
        return debug_rag_tool(**args)
    return "Tool not found."

def save_knowledge_tool(category: str, content: str | dict | None = None, day: str = None, confirm: bool = False, **kwargs) -> str:
    if content is None:
        return "Errore: specifica l'informazione da salvare."
    if isinstance(content, dict): content = content.get("text", str(content))
    if isinstance(category, dict): category = category.get("value", str(category))
    content, category = str(content), str(category).lower()
    if not content.strip():
        return "Errore: specifica l'informazione da salvare."

    if not confirm:
        return _stage_action("save_knowledge", {"category": category, "content": content, "day": day, "confirm": True})
    
    if any(x in category for x in ["paziente", "patient", "abitudin", "preferenz", "condizion", "condition", "habit", "preferenc"]):
        if "abitudin" in category or "habit" in category:
            category = "habits"
        elif "preferenz" in category or "preferenc" in category:
            category = "preferences"
        elif "condizion" in category or "condition" in category:
            category = "conditions"
        else: category = "patient"
    elif "caregiver" in category: category = "caregiver"
    else: category = "patient"
    
    km_result = km.save_knowledge_note(category, content, day=day)
    index = get_rag_index()
    if index:
        meta = {
            "category": category,
            "source": "chat",
            "type": "extracted",
            "patient_id": km.current_patient_id,
            "caregiver_id": km.current_caregiver_id,
        }
        if day: meta["validity_day"] = day
        doc = Document(text=f"[{day or 'Always'}] {content}", metadata=meta)
        index.insert(doc)
        return f"{km_result} e indicizzata."
    return km_result

def switch_context_tool(patient_id: str = None, caregiver_id: str = None) -> str:
    pid = patient_id or km.current_patient_id
    cid = caregiver_id or km.current_caregiver_id
    if not pid and not cid:
        return "Errore: specifica patient_id e caregiver_id."
    if isinstance(pid, dict): pid = pid.get("value", pid.get("id", str(pid)))
    if isinstance(cid, dict): cid = cid.get("value", cid.get("id", str(cid)))
    pid = str(pid).strip() if pid is not None else pid
    cid = str(cid).strip() if cid is not None else cid

    # Resolve names to IDs when a direct file match is missing.
    if pid:
        p_file = Path("data") / "patients" / f"{pid}.json"
        if not p_file.exists():
            resolved = km.find_patient_id_by_name(pid)
            if resolved:
                pid = resolved
    if cid:
        c_file = Path("data") / "caregivers" / f"{cid}.json"
        if not c_file.exists():
            resolved = km.find_caregiver_id_by_name(cid)
            if resolved:
                cid = resolved

    km.set_context(str(pid), str(cid))
    return f"Contesto aggiornato: {km.patient_profile.name}, {km.caregiver_profile.name}."

def delete_activity_tool(name: str, day: str, force: bool = False, confirm: bool = False) -> str:
    context_error = _ensure_patient_context()
    if context_error:
        return context_error
    if not confirm:
        return _stage_action("delete_activity", {"name": name, "day": day, "force": force, "confirm": True})
    return km.remove_activity(name, day, force=force)

def modify_activity_tool(
    old_name: str,
    day: str = None,
    new_name: str = None,
    new_description: str = None,
    new_time: str = None,
    new_days: List[str] = None,
    duration_minutes: int | None = None,
    force: bool = False,
    confirm: bool = False,
    valid_from: str | None = None,
    valid_until: str | None = None,
    duration_days: int | None = None,
) -> str:
    context_error = _ensure_patient_context()
    if context_error:
        return context_error
    # 1. Validazione
    if not old_name or not day:
        return "Errore: Devi specificare il nome dell'attività da modificare e il giorno."
    
    if new_name == old_name: new_name = None
    if new_description is not None and not str(new_description).strip(): new_description = None
    if new_name and not new_description:
        new_description = new_name

    # Preparazione dati update per controlli
    updates = {}
    if new_name: updates["name"] = new_name
    if new_description: updates["description"] = new_description
    if new_time:
        new_time, duration_minutes = _normalize_time_and_duration(new_time, duration_minutes)
        updates["time"] = new_time
    if new_days:
        updates["day_of_week"] = new_days
    if duration_minutes is not None:
        updates["duration_minutes"] = duration_minutes
    valid_from, valid_until = _apply_duration(valid_from, valid_until, duration_days)
    if valid_from: updates["valid_from"] = valid_from
    if valid_until: updates["valid_until"] = valid_until

    # 2. Controllo Conflitti (PRE-CONFERMA)
    warnings = []
    if not force:
        # Conflitti Tecnici (Temporali / Dipendenze)
        km_warnings = km.check_update_conflicts(old_name, day, updates)
        if km_warnings:
            warnings.extend(km_warnings)
        
        # Conflitti Semantici
        if new_name or new_description:
            sem_name = new_name or old_name
            sem_desc = new_description or "Invariata"
            sem_warning = check_semantic_conflict(sem_name, sem_desc)
            sem_warning = _shorten_semantic_warning(sem_warning)
            if sem_warning and confirm and not force:
                return (
                    "BLOCCO SEMANTICO: "
                    f"{sem_warning}. Se vuoi procedere comunque, ripeti con force=True."
                )
            if sem_warning:
                warnings.append(f"Avviso Semantico: {sem_warning}")

    # 4. Gestione Conferma / Staging
    if not confirm:
        warning_msg = ""
        if warnings:
            warning_msg = f"\n⚠️ ATTENZIONE: {'; '.join(warnings)}."
            warning_msg += "\nPer procedere comunque, conferma l'azione (verrà applicato force=True)."
            force = True

        return _stage_action(
            "modify_activity",
            {
                "old_name": old_name,
                "day": day,
                "new_name": new_name,
                "new_description": new_description,
                "new_time": new_time,
                "new_days": new_days,
                "duration_minutes": duration_minutes,
                "force": force,
                "confirm": True,
                "valid_from": valid_from,
                "valid_until": valid_until,
                "duration_days": duration_days,
            },
        ) + warning_msg

    # 5. Esecuzione Reale
    result = km.update_activity(old_name, day, updates, force=force)
    if "successo" in result.lower():
        updated = km.get_activity_by_name_day(new_name or old_name, day)
        if updated:
            _index_activity_in_rag(updated, "modify")
    return result

def add_activity_tool(
    name: str = None,
    description: str = "",
    days: List[str] = [],
    time: str = None,
    duration_minutes: int | None = None,
    dependencies: List[str] = [],
    force: bool = False,
    confirm: bool = False,
    valid_from: str | None = None,
    valid_until: str | None = None,
    duration_days: int | None = None,
) -> str:
    context_error = _ensure_patient_context()
    if context_error:
        return context_error
    # 1. Validazione Parametri Base
    if not name:
        return "Errore: Devi specificare il NOME dell'attività."
    if not days:
        return "Errore: Devi specificare almeno un GIORNO della settimana."
    if not time:
        return "Errore: Devi specificare l'ORARIO dell'attività."

    try:
        # 2. Creazione Oggetto Temporaneo per Controlli
        import time as t
        temp_id = f"temp_{int(t.time())}"
        valid_from, valid_until = _apply_duration(valid_from, valid_until, duration_days)
        time, duration_minutes = _normalize_time_and_duration(time, duration_minutes)
        
        # Pulizia giorni
        clean_days = [d.strip() for d in days if isinstance(d, str) and d.strip()]
        clean_days = _expand_days_for_duration(clean_days, duration_days)
        
        new_activity = Activity(
            activity_id=temp_id,
            name=name,
            description=description or name,
            day_of_week=clean_days,
            time=time,
            duration_minutes=duration_minutes,
            dependencies=dependencies,
            valid_from=valid_from,
            valid_until=valid_until,
        )

        # 3. Controllo Conflitti (PRE-CONFERMA)
        warnings = []
        if not force:
            # Conflitti Temporali
            t_conflicts = km.check_temporal_conflict(new_activity)
            if t_conflicts:
                warnings.extend(t_conflicts)
            
            # Conflitti Semantici (solo se non forzato)
            sem_warning = check_semantic_conflict(name, description or name)
            sem_warning = _shorten_semantic_warning(sem_warning)
            if sem_warning and confirm and not force:
                return (
                    "BLOCCO SEMANTICO: "
                    f"{sem_warning}. Se vuoi procedere comunque, ripeti con force=True."
                )
            if sem_warning:
                warnings.append(f"Avviso Semantico: {sem_warning}")

        # 4. Gestione Conferma / Staging
        if not confirm:
            warning_msg = ""
            if warnings:
                warning_msg = f"\n⚠️ ATTENZIONE: {'; '.join(warnings)}."
                warning_msg += "\nPer procedere comunque, conferma l'azione (verrà applicato force=True)."
                # Se ci sono conflitti, pre-impostiamo force=True nell'azione in sospeso
                force = True 
            
            return _stage_action(
                "add_activity",
                {
                    "name": name,
                    "description": description,
                    "days": days,
                    "time": time,
                    "duration_minutes": duration_minutes,
                    "dependencies": dependencies,
                    "force": force,
                    "confirm": True,
                    "valid_from": valid_from,
                    "valid_until": valid_until,
                    "duration_days": duration_days,
                },
            ) + warning_msg

        # 5. Esecuzione Reale (Post-Conferma)
        print(f"[DEBUG] Invoking KM.add_activity for {name}...")
        result = km.add_activity(new_activity, force=force)
        print(f"[DEBUG] KM result: {result}")
        
        if "successo" in result.lower():
            _index_activity_in_rag(new_activity, "add")
            
        return result

    except Exception as e:
        logger.exception("Error in add_activity_tool")
        return f"Errore interno: {str(e)}"

def get_schedule_tool(day: str = None, date: str = None) -> str:
    context_error = _ensure_patient_context()
    if context_error:
        return context_error
    if not day:
        return "Errore: specifica un giorno per il programma."
    activities = km.get_activities_by_day(day, date_str=date)
    if not activities: return f"Nessuna attività per {day}."
    output = f"Programma {day}:\n"
    for act in activities:
        time_label = act.time
        if act.duration_minutes and "-" not in act.time:
            time_label = f"{act.time} ({act.duration_minutes}m)"
        output += f"- [{time_label}] {act.name}: {act.description}\n"
    return output

def get_schedule_week_tool() -> str:
    context_error = _ensure_patient_context()
    if context_error:
        return context_error
    week = km.get_week_schedule()
    output = "Programma settimanale:\n"
    for day, activities in week.items():
        output += f"{day}:\n"
        if not activities:
            output += "- (nessuna attività)\n"
            continue
        for act in activities:
            time_label = act.time
            if act.duration_minutes and "-" not in act.time:
                time_label = f"{act.time} ({act.duration_minutes}m)"
            output += f"- [{time_label}] {act.name}: {act.description}\n"
    return output

def get_patient_info_tool(category: str = "conditions") -> str:
    profile = km.patient_profile
    if not profile:
        return "Profilo paziente non caricato."
    cat = (category or "conditions").lower()
    if "cond" in cat:
        items = profile.medical_conditions
        label = "Condizioni mediche"
    elif "prefer" in cat:
        items = profile.preferences
        label = "Preferenze"
    elif "habit" in cat or "abitud" in cat:
        items = profile.habits
        label = "Abitudini"
    elif "note" in cat:
        items = [n.content for n in profile.notes]
        label = "Note"
    elif "all" in cat:
        parts = []
        parts.append(f"Condizioni: {', '.join(profile.medical_conditions) or 'Nessuna'}")
        parts.append(f"Preferenze: {', '.join(profile.preferences) or 'Nessuna'}")
        parts.append(f"Abitudini: {', '.join(profile.habits) or 'Nessuna'}")
        notes = "; ".join(n.content for n in profile.notes) if profile.notes else "Nessuna"
        parts.append(f"Note: {notes}")
        return " | ".join(parts)
    else:
        items = profile.medical_conditions
        label = "Condizioni mediche"

    if not items:
        return f"{label}: Nessuna."
    return f"{label}: {', '.join(items)}."

def get_caregiver_info_tool(category: str = "notes") -> str:
    profile = km.caregiver_profile
    if not profile:
        return "Profilo caregiver non caricato."
    cat = (category or "notes").lower()
    if "semant" in cat or "prefer" in cat:
        items = profile.semantic_preferences
        label = "Preferenze semantiche"
    elif "note" in cat or "info" in cat:
        items = [n.content for n in profile.notes]
        label = "Note caregiver"
    elif "all" in cat:
        prefs = ", ".join(profile.semantic_preferences) or "Nessuna"
        notes = "; ".join(n.content for n in profile.notes) if profile.notes else "Nessuna"
        return f"Preferenze semantiche: {prefs}. Note caregiver: {notes}."
    else:
        items = [n.content for n in profile.notes]
        label = "Note caregiver"

    if not items:
        return f"{label}: Nessuna."
    return f"{label}: {', '.join(items)}."

def get_context_tool() -> str:
    if not km.patient_profile or not km.caregiver_profile:
        return "Contesto non disponibile."
    return f"Paziente: {km.patient_profile.name}. Caregiver: {km.caregiver_profile.name}."

def consult_guidelines_tool(query: str) -> str:
    index = get_rag_index()
    if not index: return "Errore: DB non trovato."
    if MetadataFilters and ExactMatchFilter:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="type", value="guideline")])
        return str(index.as_query_engine(similarity_top_k=3, filters=filters).query(query))
    return str(index.as_query_engine(similarity_top_k=3).query(query))

def confirm_action_tool() -> str:
    pending = _consume_pending_action()
    print(f"[DEBUG] Confirming pending action: {pending}")
    if not pending:
        return "Nessuna azione in sospeso."
    tname = pending.get("tool_name")
    args = pending.get("arguments", {})
    return _execute_tool(tname, args)

def cancel_action_tool() -> str:
    pending = _consume_pending_action()
    if not pending:
        return "Nessuna azione in sospeso."
    return "Azione annullata."

# --- ROUTER & AGENT LOGIC ---

def build_system_prompt(user_input: str, strict: bool = False):
    p_profile = km.patient_profile
    c_profile = km.caregiver_profile
    
    # 1. Definizione Entità (Chi è chi)
    entities_str = "ENTITÀ COINVOLTE:\n"
    if p_profile:
        entities_str += f"- PAZIENTE (Soggetto della cura): {p_profile.name}\n"
        if p_profile.medical_conditions: entities_str += f"  * Condizioni Mediche: {', '.join(p_profile.medical_conditions)}\n"
        if p_profile.preferences: entities_str += f"  * Preferenze: {', '.join(p_profile.preferences)}\n"
    else:
        entities_str += "- PAZIENTE: Non selezionato.\n"
        
    if c_profile:
        entities_str += f"- CAREGIVER (Tu parli con lui): {c_profile.name}\n"
        if c_profile.notes:
            notes = [n.content for n in c_profile.notes]
            entities_str += f"  * Note Operative: {'; '.join(notes)}\n"

    # 2. Contesto Dinamico
    chat_history = session.get_recent_history()
    rag_context = get_rag_context(user_input, p_profile.name if p_profile else None, c_profile.name if c_profile else None)
    available = km.get_available_users()

    strict_suffix = ""
    if strict:
        strict_suffix = (
            "\n--- STRICT MODE (Modello piccolo) ---\n"
            "Se non sei SICURO del giorno/orario, usa reply per chiedere chiarimenti.\n"
            "Non usare parole come 'sera' negli orari: converti sempre in HH:MM.\n"
            "Non inventare accenti nei giorni: usa solo i 7 giorni standard.\n"
            "Se non sei sicuro dell'accento, usa reply per chiedere il giorno corretto.\n"
            "Giorni esatti: Lunedì, Martedì, Mercoledì, Giovedì, Venerdì, Sabato, Domenica.\n"
            "Usa accento grave: ì/è (non í/é).\n"
            "Non chiamare save_knowledge senza content: se manca, usa reply.\n"
            "Se l'utente chiede due cose insieme, chiedi di separarle.\n"
            "Esempio strict:\n"
            "User: \"Aggiungi Ossigenoterapia mercoledì\"\n"
            "JSON: {\"action\":\"reply\",\"message\":\"A che ora vuoi aggiungere l'attività?\"}\n"
            "User: \"Dimmi le attività di mercoledi\"\n"
            "JSON: {\"action\":\"call_tool\",\"tool_name\":\"get_schedule\",\"arguments\":{\"day\":\"Mercoledì\"}}\n"
            "User: \"Passa al paziente Alessandro e dimmi le attività di mercoledì\"\n"
            "JSON: {\"action\":\"reply\",\"message\":\"Posso fare una sola azione per volta. Vuoi che cambi paziente o che mostri le attività?\"}\n"
            "User: \"Oggi visita anticipata alle 14:00\"\n"
            "JSON: {\"action\":\"call_tool\",\"tool_name\":\"save_knowledge\",\"arguments\":{\"category\":\"caregiver\",\"content\":\"Oggi visita anticipata alle 14:00\"}}\n"
        )

    return f"""SEI KMChat: Un assistente virtuale esperto per terapie mediche.
LINGUA: Rispondi SEMPRE in ITALIANO. Non usare mai l'inglese.
OBIETTIVO: Aiutare il Caregiver (operatore) a gestire il Paziente {p_profile.name if p_profile else ''}.

{entities_str}

CONOSCENZA RECUPERATA (RAG):
{rag_context or "Nessuna info specifica."}

STORICO RECENTE:
{chat_history}

UTENTI DISPONIBILI:
Pazienti: {available.get("patients")}
Caregivers: {available.get("caregivers")}

--- REGOLE MANDATORIE ---
1. RISPONDI SOLO IN JSON.
2. ESTRAZIONE ISTANTANEA: Se l'utente menziona un fatto (es. "è allergico", "non mangia X", "per me mattina è alle 7"), DEVI usare `save_knowledge` immediatamente.
3. NO ALLUCINAZIONI: Se mancano dati (ora/giorno), usa `reply` per chiederli.
4. CONFERME: Se un'azione è in sospeso e l'utente dice "sì", "ok", "conferma" o "salva", usa `confirm_action()`.
5. PROGRAMMA: Usa `get_schedule_week()` SOLO se l'utente chiede la settimana intera.
6. CONTESTO: Se non è selezionato un paziente, chiedi di usare `switch_context` con un ID dalla lista.
7. MODIFY: `modify_activity` richiede SEMPRE il giorno (`day`).
   - Se l'utente vuole aggiungere un giorno alla stessa attività, usa `new_days` con la lista aggiornata.
8. DURATA TEMPORANEA: se l'utente dice "per i prossimi N giorni", usa `duration_days: N` (non inventare date).
   - Se è indicato UN solo giorno, estendi automaticamente i giorni consecutivi (giorno iniziale + N giorni).
   - Se l'utente specifica la durata dell'attività in minuti, usa `duration_minutes`.
9. CATEGORIE KNOWLEDGE:
   - Paziente: salute, condizioni, abitudini, preferenze del paziente.
   - Caregiver: preferenze linguistiche/semantiche, routine, frasi in prima persona ("io", "per me", "quando dico").
   - Vincoli/Divieti ("non deve", "vietato", "non assumere") sono condizioni del paziente, NON abitudini.
   - Se l'utente parla della propria routine (es. "la mia visita abituale"), salva come caregiver, NON come attività.
   - Se l'utente dice "salva questa informazione" senza una pending action, chiedi quale informazione salvare.
   - Se l'utente chiede "note del paziente", usa `get_patient_info(category="notes")`.
   - Se l'utente chiede "note del caregiver", usa `get_caregiver_info(category="notes")`.
10. FORMATI: Usa SOLO questi giorni: Lunedì, Martedì, Mercoledì, Giovedì, Venerdì, Sabato, Domenica. Accento grave su ì/è (non í/é).
11. FORMATI ORA: Se l'utente fornisce un intervallo "HH:MM-HH:MM", converti in orario iniziale + `duration_minutes`. Non usare parole come "sera".
12. MULTI-AZIONE: Se l'utente chiede più azioni nello stesso messaggio, usa `reply` per chiedere di separarle.
13. STILE: Risposte concise, senza saluti o firme.

STRUMENTI:
- `get_schedule(day)`
- `get_schedule_week()`
- `get_patient_info(category)`
- `get_caregiver_info(category)`
- `add_activity(name, days, time, duration_minutes=None, dependencies=[], force=False)`
- `modify_activity(...)`
- `delete_activity(name, day)`
- `save_knowledge(category, content)`: Categorie: 'conditions', 'preferences', 'habits', 'caregiver'.
- `switch_context(patient_id, caregiver_id)`
- `confirm_action()` / `cancel_action()`
- `debug_rag(query)`
- `reply(message)`: Per domande generali o dati mancanti.

ESEMPIO ESTRAZIONE:
User: "Il paziente è celiaco"
JSON: {{"action": "call_tool", "tool_name": "save_knowledge", "arguments": {{"category": "conditions", "content": "Il paziente è celiaco"}}}}
User: "Quando dico Aulin intendo la forma granulare"
JSON: {{"action": "call_tool", "tool_name": "save_knowledge", "arguments": {{"category": "caregiver", "content": "Quando dico Aulin intendo la forma granulare"}}}}
User: "Il paziente non può bere latte a colazione"
JSON: {{"action": "call_tool", "tool_name": "save_knowledge", "arguments": {{"category": "conditions", "content": "Il paziente non può bere latte a colazione"}}}}

ESEMPIO PROGRAMMA:
User: "Dimmi le attività di martedì"
JSON: {{"action": "call_tool", "tool_name": "get_schedule", "arguments": {{"day": "Martedì"}}}}
User: "Dimmi le attività della settimana"
JSON: {{"action": "call_tool", "tool_name": "get_schedule_week", "arguments": {{}}}}
User: "Aggiungi attività 'Ossigenoterapia' mercoledì"
JSON: {{"action": "reply", "message": "A che ora vuoi aggiungere l'attività?"}}
User: "Quali sono le note del paziente?"
JSON: {{"action": "call_tool", "tool_name": "get_patient_info", "arguments": {{"category": "notes"}}}}
User: "Quali sono le note del caregiver?"
JSON: {{"action": "call_tool", "tool_name": "get_caregiver_info", "arguments": {{"category": "notes"}}}}
User: "Debug RAG: Ossigenoterapia mercoledì alle 11:00"
JSON: {{"action": "call_tool", "tool_name": "debug_rag", "arguments": {{"query": "Ossigenoterapia mercoledì alle 11:00"}}}}

ESEMPIO MODIFICA:
User: "Sostituisci l'attività Camminata di lunedì con Cyclette al chiuso alle 18:00"
JSON: {{"action": "call_tool", "tool_name": "modify_activity", "arguments": {{"old_name": "Camminata", "day": "Lunedì", "new_name": "Cyclette al chiuso", "new_time": "18:00"}}}}
User: "Aggiungi controllo pressione martedì alle 09:00 per i prossimi 2 giorni"
JSON: {{"action": "call_tool", "tool_name": "add_activity", "arguments": {{"name": "Controllo pressione", "days": ["Martedì", "Mercoledì", "Giovedì"], "time": "09:00", "duration_days": 2}}}}
User: "Aggiungi camomilla mercoledì di sera"
JSON: {{"action": "call_tool", "tool_name": "add_activity", "arguments": {{"name": "Camomilla", "days": ["Mercoledì"], "time": "21:00"}}}}
User: "La mia visita abituale è alle 18:00"
JSON: {{"action": "call_tool", "tool_name": "save_knowledge", "arguments": {{"category": "caregiver", "content": "La mia visita abituale è alle 18:00"}}}}
{strict_suffix}
"""


import traceback

async def run_agent_step(llms: Dict, user_input: str) -> AsyncGenerator[str, None]:
    # 1. Routing
    selected_llm = llms["SMART"]
    logger.info(f"Routing: '{user_input}' -> SMART Model")
    
    # 2. Aggiorna Memoria Condivisa
    session.append_interaction("Utente", user_input)

    data = None
    full_text = ""
    auto_confirm_msg = ""

    try:
        pending = PENDING_ACTION
        if pending:
            normalized = user_input.strip().lower()
            confirm_tokens = {"si", "sì", "ok", "conferma", "salva"}
            cancel_tokens = {"no", "annulla", "stop", "cancella"}
            if normalized in confirm_tokens or normalized.startswith("salva"):
                res = confirm_action_tool()
                session.append_interaction("KMChat", res)
                yield res
                return
            if normalized in cancel_tokens:
                res = cancel_action_tool()
                session.append_interaction("KMChat", res)
                yield res
                return
            if pending.get("tool_name") == "save_knowledge":
                auto_confirm_msg = confirm_action_tool()
            else:
                msg = (
                    f"Hai un'azione in sospeso: {pending.get('tool_name')}. "
                    "Conferma o annulla per proseguire."
                )
                session.append_interaction("KMChat", msg)
                yield msg
                return

        # --- LLM CALL (PURE AI APPROACH) ---
        
        model_name = str(getattr(selected_llm, "model", "")).lower()
        strict_hint = os.getenv("KMCHAT_STRICT", "").strip() == "1"
        strict = strict_hint or any(tag in model_name for tag in ("1b", "2b", "3b", "4b", "7b", "8b"))
        system_prompt = build_system_prompt(user_input, strict=strict)
        prompt = f"{system_prompt}\n\nUtente: {user_input}\nJSON:"
        
        response_gen = selected_llm.stream_complete(prompt)
        for token in response_gen: full_text += token.delta or ""
        
        data = _extract_json_object(full_text) or _parse_action_string(full_text)
        if data:
            data = _normalize_tool_action(data)
        else:
            data = {"action": "reply", "message": full_text}

        # --- EXECUTION ---
        action = data.get("action", "reply")
        if action != "call_tool" and data.get("action") != "reply":
            action = "reply"

        if action == "reply":
            coerced = _coerce_tool_call(selected_llm, user_input)
            if coerced:
                if coerced.get("action") == "call_tool":
                    data = coerced
                    action = "call_tool"
                else:
                    msg = str(data.get("message", full_text)).strip()
                    if not msg or msg == "{}" or "respond only in json" in msg.lower():
                        data = coerced

        final_reply = ""

        if action == "reply":
            msg = data.get("message", full_text)
            final_reply = str(msg)
            if auto_confirm_msg:
                if final_reply.strip().lower().startswith("nessuna azione in sospeso"):
                    final_reply = auto_confirm_msg
                else:
                    final_reply = f"{auto_confirm_msg}\n{final_reply}"
            yield final_reply
        
        elif action == "call_tool":
            tname = data.get("tool_name")
            args = data.get("arguments", {})
            if not args and "arguments" not in data:
                args = {k: v for k, v in data.items() if k not in ("action", "tool_name")}
            
            args = _sanitize_tool_args(tname, args, user_input=user_input)
            logger.info(f"Tool: {tname} Args: {args}")
            print(f"[DEBUG] Args: {args}")
            
            if auto_confirm_msg and tname in {"confirm_action", "cancel_action"}:
                final_reply = auto_confirm_msg
                yield final_reply
                session.append_interaction("KMChat", final_reply)
                return

            res = _execute_tool(tname, args)
            
            if tname == "consult_guidelines":
                final_reply = str(res)
                if auto_confirm_msg:
                    final_reply = f"{auto_confirm_msg}\n{final_reply}"
                yield final_reply
            elif tname == "cancel_action":
                final_reply = "Azione annullata come richiesto."
                if auto_confirm_msg:
                    final_reply = f"{auto_confirm_msg}\n{final_reply}"
                yield final_reply
            elif (
                tname in {"get_schedule", "get_schedule_week", "get_patient_info", "get_caregiver_info", "switch_context", "confirm_action", "get_context"}
                or str(res).startswith("Azione in sospeso:")
                or str(res).startswith("Errore")
                or str(res).startswith("BLOCCO SEMANTICO")
            ):
                final_reply = str(res)
                if auto_confirm_msg:
                    final_reply = f"{auto_confirm_msg}\n{final_reply}"
                yield final_reply
            else:
                final_prompt = (
                    f"SISTEMA: Risultato dell'azione: {res}\n"
                    f"COMPITO: Rispondi in modo conciso e professionale in ITALIANO.\n"
                    f"- Niente saluti o firme.\n"
                    f"- Se hai salvato conoscenza, di: 'Ho registrato questa informazione.'\n"
                    f"- Se hai aggiunto attività, conferma orario e giorno.\n"
                    f"- Se c'è un errore o conflitto, spiegalo chiaramente.\n"
                    f"Utente: {user_input}\n"
                    f"Risposta:"
                )
                if auto_confirm_msg:
                    final_reply += f"{auto_confirm_msg}\n"
                    yield f"{auto_confirm_msg}\n"
                for token in llms["SMART"].stream_complete(final_prompt): 
                    chunk = token.delta or ""
                    final_reply += chunk
                    yield chunk
        else:
             final_reply = f"Azione sconosciuta: {action}"
             if auto_confirm_msg:
                 final_reply = f"{auto_confirm_msg}\n{final_reply}"
             yield final_reply

        session.append_interaction("KMChat", final_reply)

    except Exception as e:
        traceback.print_exc()
        logger.exception(f"Err: {e}")
        yield f"Errore: {e}"

async def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-prompt", type=str)
    parser.add_argument("--model-fast", type=str, default="kmchat-14b", help="Model for simple tasks")
    parser.add_argument("--model-smart", type=str, default="kmchat-14b", help="Model for complex reasoning")
    args = parser.parse_args()

    print(f"🔌 Init Models... FAST: {args.model_fast}, SMART: {args.model_smart}")

    max_tokens = int(os.getenv("KMCHAT_MAX_TOKENS", "0") or "0")

    # Configurazione ottimizzata per Ollama
    llm_fast = Ollama(
        model=args.model_fast, 
        request_timeout=60.0, 
        temperature=0.1, 
        context_window=8192,
        additional_kwargs={"stop": ["Utente:", "\nUtente", "Caregiver:", "\nCaregiver"]},
        ollama_additional_kwargs={"keep_alive": "60m", "num_predict": max_tokens}
    )
    llm_smart = Ollama(
        model=args.model_smart, 
        request_timeout=120.0, 
        temperature=0.2, 
        context_window=8192,
        additional_kwargs={"stop": ["Utente:", "\nUtente", "Caregiver:", "\nCaregiver"]},
        ollama_additional_kwargs={"keep_alive": "60m", "num_predict": max_tokens}
    )
    
    # Global Settings use SMART by default for internal logic checks
    Settings.llm = llm_smart 
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    llms = {"FAST": llm_fast, "SMART": llm_smart}

    if args.test_prompt:
        print(f"\n🧪 Test: {args.test_prompt}")
        full = ""
        async for chunk in run_agent_step(llms, args.test_prompt): full += str(chunk)
        print(f"\nKMChat: {full}")
        return

    print(f"\n✅ KMChat Router Active. Shared Memory: {HISTORY_FILE}")
    while True:
        try:
            inp = await asyncio.to_thread(input, "\nCaregiver: ")
            if inp.lower().strip() in ["exit", "quit"]: break
            if not inp.strip(): continue
            print("\nKMChat: ", end="", flush=True)
            async for chunk in run_agent_step(llms, inp): print(str(chunk), end="", flush=True)
            print()
        except KeyboardInterrupt: break
        except Exception as e: print(f"Errore: {e}")

if __name__ == "__main__":
    asyncio.run(main())
