import streamlit as st
import sys
import asyncio
import logging
from pathlib import Path
import pandas as pd
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

# Ensure project root is on sys.path when running via `streamlit run src/app.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Importiamo la logica di business
from src.knowledge_manager import KnowledgeManager
from src.models import Activity
from src.logging_utils import setup_logger

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="KMChat - Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

# --- COSTANTI & STATO ---
MODEL_NAME = "kmchat-14b"
DB_DIR = "data"
logger = setup_logger("app", "app")

if "km" not in st.session_state:
    st.session_state.km = KnowledgeManager(auto_discover=False)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ciao! Sono KMChat. Come posso aiutarti con la gestione della terapia oggi?"}
    ]
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

# --- DEFINIZIONE TOOLS (Adattati per Streamlit) ---
# Nota: Li ridefiniamo qui per garantire accesso allo stato della sessione se necessario

def add_activity_tool(
    name: str,
    description: str,
    days: list,
    time: str,
    duration_minutes: int | None = None,
    confirm: bool = False,
) -> str:
    """Aggiunge una nuova attività alla terapia."""
    try:
        if not st.session_state.km.current_patient_id:
            return "Nessun paziente selezionato. Imposta il contesto prima di aggiungere attività."
        if not confirm:
            st.session_state.pending_action = {
                "tool": "add_activity",
                "args": {
                    "name": name,
                    "description": description,
                    "days": days,
                    "time": time,
                    "duration_minutes": duration_minutes,
                    "confirm": True,
                },
            }
            return "Azione in sospeso. Scrivi 'conferma' per applicare o 'annulla' per annullare."
        import time as t_lib
        act_id = f"act_{int(t_lib.time())}"
        if isinstance(time, str) and "-" in time:
            parts = [p.strip() for p in time.split("-", 1)]
            if len(parts) == 2:
                try:
                    start_h, start_m = map(int, parts[0].split(":"))
                    end_h, end_m = map(int, parts[1].split(":"))
                    start_min = start_h * 60 + start_m
                    end_min = end_h * 60 + end_m
                    if end_min > start_min and duration_minutes is None:
                        duration_minutes = end_min - start_min
                    time = parts[0]
                except Exception:
                    pass
        new_activity = Activity(
            activity_id=act_id, name=name, description=description,
            day_of_week=days, time=time, duration_minutes=duration_minutes, dependencies=[]
        )
        result = st.session_state.km.add_activity(new_activity)
        return result
    except Exception as e:
        return f"Errore: {str(e)}"

def get_schedule_tool(day: str) -> str:
    """Restituisce le attività per un giorno specifico."""
    if not st.session_state.km.current_patient_id:
        return "Nessun paziente selezionato. Imposta il contesto prima di richiedere il programma."
    activities = st.session_state.km.get_activities_by_day(day)
    if not activities:
        return f"Nessuna attività prevista per {day}."
    output = f"Programma per {day}:\n"
    for act in activities:
        time_label = act.time
        if act.duration_minutes and "-" not in act.time:
            time_label = f"{act.time} ({act.duration_minutes}m)"
        output += f"- [{time_label}] {act.name}\n"
    return output

def confirm_action_tool() -> str:
    pending = st.session_state.pending_action
    if not pending:
        return "Nessuna azione in sospeso."
    st.session_state.pending_action = None
    tool = pending.get("tool")
    args = pending.get("args", {})
    if tool == "add_activity":
        return add_activity_tool(**args)
    return "Azione non valida."

def cancel_action_tool() -> str:
    if not st.session_state.pending_action:
        return "Nessuna azione in sospeso."
    st.session_state.pending_action = None
    return "Azione annullata."

def run_agent_sync(agent, prompt: str):
    """Esegue l'agente ReAct in modo sincrono partendo da un prompt."""
    async def _run():
        handler = agent.run(user_msg=prompt)
        return await handler
    return asyncio.run(_run())

# --- INIZIALIZZAZIONE AGENTE (Lazy Loading) ---
@st.cache_resource
def get_agent():
    # 1. LLM
    llm = Ollama(model=MODEL_NAME, request_timeout=120.0)
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # 2. RAG Tool
    db_path = Path(DB_DIR) / "chroma_db"
    if not db_path.exists():
        logger.error("DB vettoriale non trovato a %s", db_path)
        return None # Gestito nella UI

    db_client = chromadb.PersistentClient(path=str(db_path))
    chroma_collection = db_client.get_collection("patient_therapies")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    rag_engine = index.as_query_engine(similarity_top_k=3)
    rag_tool = QueryEngineTool(
        query_engine=rag_engine,
        metadata=ToolMetadata(
            name="consult_guidelines",
            description="Cerca linee guida mediche e info storiche nel database."
        )
    )

    # 3. Logic Tools
    schedule_tool = FunctionTool.from_defaults(fn=get_schedule_tool)
    add_tool = FunctionTool.from_defaults(fn=add_activity_tool)
    confirm_tool = FunctionTool.from_defaults(fn=confirm_action_tool)
    cancel_tool = FunctionTool.from_defaults(fn=cancel_action_tool)

    # 4. Agente
    logger.info("Agente inizializzato (UI) con modello %s", MODEL_NAME)
    return ReActAgent(
        tools=[rag_tool, schedule_tool, add_tool, confirm_tool, cancel_tool],
        llm=llm,
        verbose=True,
        streaming=False,
        system_prompt="""Sei KMChat, un assistente per caregiver.
Usa 'get_schedule_tool' prima di aggiungere attività.
Usa 'consult_guidelines' per domande mediche.
Se rilevi conflitti, avvisa l'utente.
Richiedi conferma prima di modificare i dati; l'utente può dire 'conferma' o 'annulla'.
Parla Italiano."""
    )

agent = get_agent()

# --- INTERFACCIA GRAFICA ---

# SIDEBAR: Visualizzazione Dati Strutturati
with st.sidebar:
    st.title("📅 Terapia Attuale")
    st.markdown("---")

    available = st.session_state.km.get_available_users()
    patients = available.get("patients") or []
    caregivers = available.get("caregivers") or []
    patient_map = {p.get("id"): p for p in patients if p.get("id")}
    caregiver_map = {c.get("id"): c for c in caregivers if c.get("id")}

    if not patient_map:
        st.warning("Nessun paziente disponibile.")
    if not caregiver_map:
        st.warning("Nessun caregiver disponibile.")

    if patient_map:
        patient_ids = list(patient_map.keys())
        current_pid = st.session_state.km.current_patient_id
        patient_index = patient_ids.index(current_pid) if current_pid in patient_ids else 0
        selected_patient = st.selectbox(
            "Paziente",
            options=patient_ids,
            index=patient_index,
            format_func=lambda pid: f"{patient_map[pid].get('name', 'Sconosciuto')} ({pid})",
        )
    else:
        selected_patient = None

    if caregiver_map:
        caregiver_ids = list(caregiver_map.keys())
        current_cid = st.session_state.km.current_caregiver_id
        caregiver_index = caregiver_ids.index(current_cid) if current_cid in caregiver_ids else 0
        selected_caregiver = st.selectbox(
            "Caregiver",
            options=caregiver_ids,
            index=caregiver_index,
            format_func=lambda cid: f"{caregiver_map[cid].get('name', 'Sconosciuto')} ({cid})",
        )
    else:
        selected_caregiver = None

    if st.button("✅ Imposta contesto"):
        if not selected_patient or not selected_caregiver:
            st.warning("Seleziona paziente e caregiver prima di continuare.")
        else:
            st.session_state.km.set_context(selected_patient, selected_caregiver)
            st.rerun()
    
    # Ricarica dati aggiornati
    if st.session_state.km.current_patient_id:
        st.session_state.km.load_data()
    therapy = st.session_state.km.therapy
    
    if therapy and therapy.activities:
        # Creiamo un DataFrame per visualizzare bene la tabella
        data = []
        for act in therapy.activities:
            for day in act.day_of_week:
                data.append({
                    "Giorno": day,
                    "Orario": act.time,
                    "Attività": act.name
                })
        
        df = pd.DataFrame(data)
        # Ordiniamo per giorno (approssimativo) e orario
        days_order = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
        df['Giorno'] = pd.Categorical(df['Giorno'], categories=days_order, ordered=True)
        df = df.sort_values(['Giorno', 'Orario'])
        
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.info("Nessuna terapia caricata.")
        
    st.markdown("---")
    if st.button("🔄 Ricarica Dati"):
        if st.session_state.km.current_patient_id:
            st.session_state.km.load_data()
            st.rerun()
        else:
            st.warning("Seleziona un paziente prima di ricaricare.")

# MAIN: Chat Interface
st.header("💬 KMChat Assistant")

if not agent:
    st.error("⚠️ Database non trovato! Esegui `python3 src/ingest_data.py` nel terminale prima di avviare l'app.")
else:
    # Mostra cronologia
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Utente
    if prompt := st.chat_input("Scrivi qui (es: 'Aggiungi visita Lunedì alle 09:00')..."):
        # 1. Mostra messaggio utente
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        logger.info("Prompt UI: %s", prompt)

        # 2. Genera risposta
        with st.chat_message("assistant"):
            with st.spinner("Ragionamento in corso..."):
                try:
                    result = run_agent_sync(agent, prompt)
                    if result and getattr(result, "response", None):
                        response_text = result.response.content or ""
                    else:
                        response_text = str(result)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    logger.info("Risposta UI: %s", response_text)
                    
                    # Forziamo il rerun per aggiornare la sidebar se i dati sono cambiati
                    if "aggiunt" in response_text.lower() or "add" in response_text.lower():
                        st.rerun()
                        
                except Exception as e:
                    logger.exception("Errore UI: %s", e)
                    st.error(f"Errore: {e}")
