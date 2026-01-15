
import sys
import os
import asyncio
from pathlib import Path

# Skip when collected by pytest; this is an integration script.
if __name__ != "__main__":
    try:
        import pytest

        pytest.skip("Integration script; run manually.", allow_module_level=True)
    except Exception:
        pass

# Setup path per importare i moduli
sys.path.append(os.getcwd())

# Importiamo le funzioni del bot
from src.main import run_agent_step, PENDING_ACTION, km, session, MODEL_FAST, MODEL_SMART
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

# Configurazione manuale identica al main
def setup():
    print("⚙️ Setup test environment...")
    llm_fast = Ollama(
        model=MODEL_FAST, 
        request_timeout=60.0, 
        temperature=0.1, 
        context_window=8192,
        ollama_additional_kwargs={"keep_alive": "60m", "num_predict": 100}
    )
    llm_smart = Ollama(
        model=MODEL_SMART, 
        request_timeout=120.0, 
        temperature=0.2, 
        context_window=8192,
        ollama_additional_kwargs={"keep_alive": "60m", "num_predict": 100}
    )
    Settings.llm = llm_smart
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    return {"FAST": llm_fast, "SMART": llm_smart}

async def test_flow():
    llms_dict = setup()
    
    print("\n--- STEP 1: CAMBIO PAZIENTE ---")
    # Forziamo il contesto inizialmente ad altro per vedere se cambia
    km.set_context("alessandro_01", "andrea_01") 
    
    prompt = "Passa al paziente Federico"
    print(f"User: {prompt}")
    async for response in run_agent_step(llms_dict, prompt):
        print(f"Bot: {response}")
    
    # Verifica interna
    if "federico" in km.current_patient_id.lower():
        print("✅ CONTESTO CAMBIATO CORRETTAMENTE.")
    else:
        print(f"❌ ERRORE CONTESTO: È rimasto {km.current_patient_id}")
        return

    print("\n--- STEP 2: RICHIESTA AGGIUNTA (STAGING) ---")
    prompt = "Aggiungi attività 'TestAutomato' lunedì alle 10:00"
    print(f"User: {prompt}")
    
    # Eseguiamo il passo
    last_response = ""
    async for response in run_agent_step(llms_dict, prompt):
        last_response += response
        print(f"Bot chunk: {response}")
        
    # Verifica Variabile Globale PENDING_ACTION importata dal main
    # Nota: Dobbiamo accedere alla variabile del modulo src.main
    import src.main
    pending = src.main.PENDING_ACTION
    
    if pending and pending['tool_name'] == 'add_activity':
        print(f"✅ AZIONE IN SOSPESO RILEVATA: {pending}")
    else:
        print(f"❌ NESSUNA AZIONE IN SOSPESO (Pending={pending}). Il bot ha probabilmente fallito il parsing.")
        # Se fallisce qui, stampiamo cosa ha capito il bot
        return

    print("\n--- STEP 3: CONFERMA E SCRITTURA SU DISCO ---")
    prompt = "Conferma"
    print(f"User: {prompt}")
    async for response in run_agent_step(llms_dict, prompt):
        print(f"Bot: {response}")

    # Verifica File su Disco
    target_file = Path("data/therapies/federico_01.json")
    if not target_file.exists():
        print("❌ FILE PAZIENTE NON TROVATO.")
        return
        
    content = target_file.read_text()
    if "TestAutomato" in content:
        print("🎉 VITTORIA: 'TestAutomato' TROVATO NEL FILE JSON!")
    else:
        print("❌ FALLIMENTO: L'attività non è stata scritta nel file.")
        print("Contenuto file:", content[:200], "...")

if __name__ == "__main__":
    try:
        asyncio.run(test_flow())
    except Exception as e:
        import traceback
        traceback.print_exc()
