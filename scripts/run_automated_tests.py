import sys
import os
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

# Import logic from main application
from src.main import run_agent_step, session, HISTORY_FILE, km

# --- DEFINIZIONE SCENARI DI TEST ---
# Qui definisci le conversazioni che vuoi testare automaticamente.
SCENARIOS = {
    "01_BASIC_ADD_CONFIRM": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Dimmi le attività di lunedì",
        "Aggiungi attività 'Lettura libro' lunedì alle 16:00",
        "Conferma", # Se chiede conferma
        "Dimmi le attività di lunedì" # Verifica
    ],
    "02_CONFLICT_DETECTION": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Aggiungi attività 'Fisioterapia' martedì alle 10:00",
        "Conferma",
        "Aggiungi attività 'Visita Infermieristica' martedì alle 10:15", # Conflitto!
        "No", # Annulla l'azione conflict
        "Dimmi le attività di martedì"
    ],
    "03_KNOWLEDGE_EXTRACTION": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Il paziente non può bere latte a colazione",
        "Salva questa informazione", # A volte l'LLM chiede conferma o lo fa da solo
        "Quali sono le condizioni del paziente?" # Domanda RAG/Context
    ],
    "04_CARE_GIVER_PREFS": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Quando dico 'sera' intendo le ore 21:00",
        "Aggiungi attività 'Camomilla' mercoledì di sera",
        "Conferma",
        "Dimmi le attività di mercoledì"
    ],
    "05_MEDICAL_CONFLICT": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Il paziente ha il diabete di tipo 1",
        "Aggiungi attività 'Merenda con Torta' giovedì alle 16:00",
        "No", # Dovrebbe rilevare il conflitto semantico e noi annulliamo
        "Dimmi le attività di giovedì"
    ],
    "06_DEPENDENCY_BREAK": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Aggiungi attività 'Pillola A' venerdì alle 08:00",
        "Conferma",
        "Aggiungi attività 'Pillola B' venerdì alle 08:30 con dipendenza Pillola A",
        "Conferma",
        "Rimuovi attività 'Pillola A' di venerdì", # Questo dovrebbe generare un avviso di dipendenza rotta
        "Annulla"
    ],
    "07_ACTIVITY_UPDATE_SUBSTITUTION": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Aggiungi attività 'Camminata' lunedì alle 18:00",
        "Conferma",
        "Sostituisci l'attività Camminata di lunedì con 'Cyclette al chiuso' alle 18:00",
        "Conferma",
        "Dimmi le attività di lunedì"
    ],
    "08_INDIRECT_CONFLICT_RULE": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Il paziente non deve assumere liquidi per 24 ore",
        "Aggiungi attività 'Assunzione farmaci con acqua' lunedì alle 12:00",
        "No",
        "Dimmi le attività di lunedì"
    ],
    "09_CAREGIVER_ROUTINE_CHANGE": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "La mia visita abituale è alle 18:00",
        "Salva questa informazione",
        "Oggi visita anticipata alle 14:00",
        "Salva questa informazione",
        "Quali sono le note del caregiver?"
    ],
    "10_SEMANTIC_AMBIGUITY": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Quando dico Aulin intendo la forma granulare",
        "Salva questa informazione",
        "Da oggi Aulin intendo la supposta",
        "Salva questa informazione",
        "Quali sono le note del caregiver?"
    ],
    "11_WEEKLY_SCHEDULE": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Dimmi le attività della settimana"
    ],
    "12_PATIENT_INFO": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Quali sono le condizioni del paziente?",
        "Quali sono le preferenze del paziente?",
        "Quali sono le abitudini del paziente?",
        "Quali sono le note del paziente?"
    ],
    "13_CAREGIVER_INFO": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Quali sono le note del caregiver?",
        "Quali sono le preferenze semantiche del caregiver?"
    ],
    "14_TEMPORARY_ACTIVITY": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Aggiungi attività 'Controllo pressione' martedì alle 09:00 per i prossimi 2 giorni",
        "Conferma",
        "Dimmi le attività di martedì"
    ],
    "15_MISSING_FIELDS": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Aggiungi attività 'Ossigenoterapia' mercoledì",
        "Alle 11:00",
        "Conferma",
        "Dimmi le attività di mercoledì"
    ],
    "16_DEPENDENCY_REMOVAL_WARNING": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Aggiungi attività 'Colazione' giovedì alle 08:00",
        "Conferma",
        "Aggiungi attività 'Farmaco X' giovedì alle 08:30 con dipendenza Colazione",
        "Conferma",
        "Rimuovi attività 'Colazione' di giovedì",
        "Annulla"
    ],
    "17_CAREGIVER_SEMANTIC_PREFS": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Quando dico mattina intendo l'intervallo tra le 08:00 e le 10:00",
        "Salva questa informazione",
        "Quali sono le note del caregiver?"
    ],
    "18_WEEKLY_AFTER_EDITS": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Dimmi le attività della settimana"
    ],
    "19_CONFLICT_CONCURRENT_ACTIVITY": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Aggiungi attività 'Fisioterapia' martedì alle 10:00",
        "Conferma",
        "Aggiungi attività 'Visita infermieristica' martedì alle 10:30",
        "No",
        "Dimmi le attività di martedì",
    ],
    "20_DEPENDENCY_REMOVAL_CONFLICT": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Aggiungi attività 'Prendere pastiglia A' venerdì alle 08:00",
        "Conferma",
        "Aggiungi attività 'Prendere pastiglia B' venerdì alle 08:30 con dipendenza Prendere pastiglia A",
        "Conferma",
        "Rimuovi attività 'Prendere pastiglia A' di venerdì",
        "Annulla",
    ],
    "21_UPDATE_ACTIVITY_SUBSTITUTION": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Aggiungi attività 'Uscire a fare una camminata' lunedì alle 18:00",
        "Conferma",
        "Sostituisci l'attività Uscire a fare una camminata di lunedì con 'Fare cyclette al chiuso' alle 18:00",
        "Conferma",
        "Dimmi le attività di lunedì",
    ],
    "22_INDIRECT_CONFLICT": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Il paziente non deve assumere liquidi per 24 ore",
        "Aggiungi attività 'Assunzione farmaci con acqua' lunedì alle 12:00",
        "No",
        "Dimmi le attività di lunedì",
    ],
    "23_SEMANTIC_AMBIGUITY": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "Quando dico Aulin intendo la forma granulare",
        "Salva questa informazione",
        "Ora Aulin intendo la supposta",
        "Salva questa informazione",
        "Quali sono le note del caregiver?",
    ],
    "24_CAREGIVER_ROUTINE_CHANGE": [
        "Passa al paziente TestAuto1 e caregiver CaregiverTest",
        "La mia visita abituale è alle 18:00",
        "Salva questa informazione",
        "Domani visita anticipata alle 14:00",
        "Salva questa informazione",
        "Quali sono le note del caregiver?",
    ],
}

async def run_scenario(name, steps, llms, log_file):
    header = f"\n{'='*20} SCENARIO: {name} {'='*20}\n"
    print(header)
    log_file.write(header)

    # Reset Session History per isolare il test
    session.file_path.write_text(f"# Test Session: {name}\n", encoding="utf-8")
    
    # Reset KM context se necessario (opzionale, ma consigliato per pulizia)
    km.set_context("TestAuto1", "CaregiverTest")

    pending_action = False
    confirm_tokens = {"conferma", "ok", "sì", "si", "salva"}
    cancel_tokens = {"annulla", "no", "stop", "cancella"}

    for user_input in steps:
        # Auto-confirm pending actions if the next step is not a confirm/cancel.
        if pending_action:
            normalized = user_input.strip().lower()
            if normalized not in confirm_tokens and normalized not in cancel_tokens:
                auto_step = "Conferma"
                step_log = f"\n👤 USER: {auto_step}\n"
                print(f"\n👤 USER: {auto_step}")
                log_file.write(step_log)
                log_file.write("🤖 BOT: ")
                print("🤖 BOT: ", end="", flush=True)

                full_response = ""
                async for chunk in run_agent_step(llms, auto_step):
                    print(chunk, end="", flush=True)
                    full_response += str(chunk)
                    log_file.write(str(chunk))
                print()
                log_file.write("\n")
                pending_action = False
                await asyncio.sleep(0.5)

        step_log = f"\n👤 USER: {user_input}\n"
        print(f"\n👤 USER: {user_input}") # Blue color for console
        log_file.write(step_log)

        log_file.write("🤖 BOT: ")
        print("🤖 BOT: ", end="", flush=True)
        
        full_response = ""
        async for chunk in run_agent_step(llms, user_input):
            print(chunk, end="", flush=True)
            full_response += str(chunk)
            log_file.write(str(chunk))

        print() # Newline console
        log_file.write("\n")
        pending_action = "Azione in sospeso" in full_response
        
        # Piccola pausa per simulare tempo di riflessione o evitare rate limit
        await asyncio.sleep(0.5)

async def main():
    parser = argparse.ArgumentParser(description="Automated KMChat Test Runner")
    parser.add_argument("--model", type=str, default="kmchat-14b", help="Model to use (e.g., llama3:70b)")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--output", type=str, default="logs/automated_test_report.txt", help="Output log file")
    args = parser.parse_args()

    # Setup Logging
    Path("logs").mkdir(exist_ok=True)
    f = open(args.output, "w", encoding="utf-8")
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"KMChat Automated Test Run - {start_time}\nModel: {args.model}\n\n")

    # --- PRE-TEST SETUP: Create Dummy Data ---
    # Senza questo, "Passa al paziente TestAuto1" fallisce perché il file non esiste.
    print("🛠️  Creating dummy data for TestAuto1...")
    Path("data/patients").mkdir(parents=True, exist_ok=True)
    Path("data/caregivers").mkdir(parents=True, exist_ok=True)
    Path("data/therapies").mkdir(parents=True, exist_ok=True)
    (Path("data/patients") / "TestAuto1.json").write_text(
        '{"patient_id": "TestAuto1", "name": "TestAuto1", "medical_conditions": [], "preferences": [], "habits": [], "notes": []}', encoding="utf-8"
    )
    (Path("data/caregivers") / "CaregiverTest.json").write_text(
        '{"caregiver_id": "CaregiverTest", "name": "Caregiver Test", "notes": [], "semantic_preferences": []}', encoding="utf-8"
    )
    # Reset therapy to empty
    (Path("data/therapies") / "TestAuto1.json").write_text('[]', encoding="utf-8")
    
    # -----------------------------------------

    print(f"🔌 Initializing Models ({args.model})...")
    
    # Inizializzazione Modelli (Copiata da main.py ma flessibile)
    llm = Ollama(
        model=args.model, 
        request_timeout=300.0, # Timeout lungo per modelli pesanti
        temperature=args.temperature, 
        context_window=8192,
        additional_kwargs={"stop": ["Utente:", "\nUtente", "Caregiver:", "\nCaregiver"]},
    )
    
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    # Dizionario LLM (possiamo usare lo stesso per Fast e Smart nel test automatizzato per semplicità,
    # oppure puoi passare argomenti diversi)
    llms = {"FAST": llm, "SMART": llm}

    print("🚀 Starting Tests...")

    try:
        for scenario_name, steps in SCENARIOS.items():
            await run_scenario(scenario_name, steps, llms, f)
    except KeyboardInterrupt:
        print("\n🛑 Test interrotto dall'utente.")
    except Exception as e:
        print(f"\n❌ Errore critico: {e}")
        f.write(f"\nCRITICAL ERROR: {e}\n")
    finally:
        f.close()
        print(f"\n✅ Test completati. Report salvato in: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
