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
MODEL_NAME = "llama3.1"
DB_DIR = "data"
logger = setup_logger("app", "app")

if "km" not in st.session_state:
    st.session_state.km = KnowledgeManager()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ciao! Sono KMChat. Come posso aiutarti con la gestione della terapia oggi?"}
    ]

# --- DEFINIZIONE TOOLS (Adattati per Streamlit) ---
# Nota: Li ridefiniamo qui per garantire accesso allo stato della sessione se necessario

def add_activity_tool(name: str, description: str, days: list, time: str) -> str:
    """Aggiunge una nuova attività alla terapia."""
    try:
        import time as t_lib
        act_id = f"act_{int(t_lib.time())}"
        new_activity = Activity(
            activity_id=act_id, name=name, description=description,
            day_of_week=days, time=time, dependencies=[]
        )
        result = st.session_state.km.add_activity(new_activity)
        return result
    except Exception as e:
        return f"Errore: {str(e)}"

def get_schedule_tool(day: str) -> str:
    """Restituisce le attività per un giorno specifico."""
    activities = st.session_state.km.get_activities_by_day(day)
    if not activities:
        return f"Nessuna attività prevista per {day}."
    output = f"Programma per {day}:\n"
    for act in activities:
        output += f"- [{act.time}] {act.name}\n"
    return output

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

    # 4. Agente
    logger.info("Agente inizializzato (UI) con modello %s", MODEL_NAME)
    return ReActAgent(
        tools=[rag_tool, schedule_tool, add_tool],
        llm=llm,
        verbose=True,
        streaming=False,
        system_prompt="""Sei KMChat, un assistente per caregiver.
Usa 'get_schedule_tool' prima di aggiungere attività.
Usa 'consult_guidelines' per domande mediche.
Se rilevi conflitti, avvisa l'utente.
Parla Italiano."""
    )

agent = get_agent()

# --- INTERFACCIA GRAFICA ---

# SIDEBAR: Visualizzazione Dati Strutturati
with st.sidebar:
    st.title("📅 Terapia Attuale")
    st.markdown("---")
    
    # Ricarica dati aggiornati
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
        st.session_state.km.load_data()
        st.rerun()

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
