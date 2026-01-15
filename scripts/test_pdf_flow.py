import asyncio
import sys
import os
import logging
from datetime import date

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import run_agent_step, km, session, get_rag_index
from src.models import Activity, PatientProfile, CaregiverProfile, Note, Therapy
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# Setup logger to file
log_file = "logs/pdf_demo_test.log"
if os.path.exists(log_file):
    os.remove(log_file)

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

async def run_test():
    print(f"🚀 Starting PDF Scenario Test with kmchat-14b...")
    print(f"📝 Logging to {log_file}\n")

    # 1. SETUP: Reset KM and load specific test state
    print("--- SETUP ---")
    km.current_patient_id = "test_pdf_patient"
    km.current_caregiver_id = "test_pdf_caregiver"
    
    # Create profiles manually to ensure known state
    km.patient_profile = PatientProfile(
        patient_id="test_pdf_patient",
        name="Alessandro (Test)",
        medical_conditions=["Ipertensione"],
        preferences=["Preferisce non essere svegliato prima delle 9"],
        habits=[],
        notes=[] # Will be populated dynamically
    )
    km.caregiver_profile = CaregiverProfile(
        caregiver_id="test_pdf_caregiver",
        name="Maria (Infermiere)",
        notes=[]
    )
    
    # Mock save_data to avoid file I/O errors during test
    km.save_data = lambda: None
    km.save_patient_profile = lambda: None
    km.save_caregiver_profile = lambda: None
    
    # Clear existing activities
    km.therapy = Therapy(patient_id="test_pdf_patient", activities=[])
    
    # Pre-populate specific activities for the test
    # 1. Fisioterapia for Temporal Conflict
    act_physio = Activity(
        activity_id="physio_01",
        name="Fisioterapia",
        description="Esercizi di riabilitazione",
        day_of_week=["Lunedì"],
        time="10:00", # Starts 10:00 (implicitly 1 hour or we'll see)
        dependencies=[]
    )
    km.add_activity(act_physio, force=True)
    
    # 2. Dependencies for Removal Conflict
    act_pill_a = Activity(
        activity_id="pill_a",
        name="Prendere pastiglia A",
        description="Base per pastiglia B",
        day_of_week=["Lunedì"],
        time="08:00",
        dependencies=[]
    )
    km.add_activity(act_pill_a, force=True)
    
    act_pill_b = Activity(
        activity_id="pill_b",
        name="Prendere pastiglia B",
        description="Deve essere presa dopo la A",
        day_of_week=["Lunedì"],
        time="08:30",
        dependencies=["Prendere pastiglia A"]
    )
    km.add_activity(act_pill_b, force=True)

    # 3. Constraint for Semantic Conflict
    # We add this directly to notes to simulate "Retrieval" or "Profile Knowledge"
    km.patient_profile.notes.append(Note(content="Il paziente non può assumere liquidi per le prossime 24 ore", day="Lunedì"))

    # Force LLM Settings to ensure we use the smart one
    model_name = "kmchat-14b"
    llm = Ollama(model=model_name, request_timeout=120.0, temperature=0.1)
    llms = {"FAST": llm, "SMART": llm} # Use smart for both to be safe
    Settings.llm = llm

    scenarios = [
        {
            "name": "1. Conflitto Temporale (Concurrency)",
            "prompt": "Aggiungi 'Visita infermieristica' lunedì dalle 10:00 alle 10:30.",
            "description": "Existing 'Fisioterapia' is at 10:00. Should detect overlap."
        },
        {
            "name": "2. Conflitto di Dipendenza (Removal)",
            "prompt": "Cancella l'attività 'Prendere pastiglia A' di lunedì.",
            "description": "'Pastiglia B' depends on 'A'. Should warn user."
        },
        {
            "name": "3. Conflitto Semantico (Indirect)",
            "prompt": "Aggiungi 'Assunzione Aulin con abbondante acqua' lunedì alle 12:00.",
            "description": "Patient has 'No liquids' constraint. Should trigger semantic block."
        },
        {
            "name": "4. Estrazione Conoscenza (Ambiguity)",
            "prompt": "Quando dico 'Aulin' intendo sempre la forma granulare.",
            "description": "Should call save_knowledge to update caregiver preferences/glossary."
        },
        {
            "name": "5. Cambio Routine (Eccezionale)",
            "prompt": "Per lunedì, modifica l'orario di 'Fisioterapia' alle 14:00 perché è festivo.",
            "description": "Should use modify_activity."
        }
    ]

    for i, scen in enumerate(scenarios, 1):
        print(f"\n--- SCENARIO {i}: {scen['name']} ---")
        print(f"Description: {scen['description']}")
        print(f"User: \"{scen['prompt']}\" ")
        
        logger.info(f"SCENARIO {i}: {scen['name']}")
        logger.info(f"PROMPT: {scen['prompt']}")
        
        full_response = ""
        print("KMChat: ", end="", flush=True)
        async for chunk in run_agent_step(llms, scen['prompt']):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n")
        
        logger.info(f"RESPONSE: {full_response}")
        
        # Simulate Confirmation if needed (Agent often asks for confirmation)
        if "conferma" in full_response.lower() or "sospeso" in full_response.lower():
            confirm_prompt = "Conferma"
            print(f"User: \"{confirm_prompt}\" (Auto-confirm for test)")
            logger.info(f"PROMPT: {confirm_prompt}")
            print("KMChat: ", end="", flush=True)
            async for chunk in run_agent_step(llms, confirm_prompt):
                print(chunk, end="", flush=True)
                full_confirm_response = chunk
            print("\n")
            logger.info(f"CONFIRM RESPONSE: {full_confirm_response}")

if __name__ == "__main__":
    # Ensure event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_test())
