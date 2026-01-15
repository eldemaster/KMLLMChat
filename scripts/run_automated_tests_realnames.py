import sys
import os
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

from src.main import run_agent_step, session, km
from src.ingest_data import ingest_data


SCENARIOS = {
    "01_MARIO_FULL_FLOW": [
        "Passa al paziente Mario Rossi",
        "Passa al caregiver Andrea Bianchi",
        "Dimmi le attività di lunedì",
        "Aggiungi attività 'Colazione zuccherata' lunedì alle 08:00",
        "No",
        "Aggiungi attività 'Visita nutrizionista' martedì alle 10:30",
        "Conferma",
        "Dimmi le attività della settimana",
        "Debug RAG: Diabete di tipo 2",
    ],
    "02_PAOL0_FULL_FLOW": [
        "Passa al paziente Paolo Verdi",
        "Passa al caregiver Maria Rossi",
        "Dimmi le attività di martedì",
        "Aggiungi attività 'Terapia calda' martedì alle 16:00",
        "No",
        "Aggiungi attività 'Stretching leggero' lunedì alle 10:00",
        "No",
        "Aggiungi attività 'Controllo fisioterapico' mercoledì alle 11:00 per i prossimi 3 giorni",
        "Conferma",
        "Dimmi le attività di mercoledì",
        "Debug RAG: Mobilizzazione articolare",
    ],
    "03_KNOWLEDGE_EXTRACTION": [
        "Passa al paziente Mario Rossi",
        "Il paziente non deve assumere FANS",
        "Salva questa informazione",
        "Quando dico mattina intendo 08:00-10:00",
        "Salva questa informazione",
        "Quali sono le condizioni del paziente?",
        "Quali sono le note del caregiver?",
        "Debug RAG: non deve assumere FANS",
    ],
}


def _write_seed_data() -> None:
    Path("data/patients").mkdir(parents=True, exist_ok=True)
    Path("data/caregivers").mkdir(parents=True, exist_ok=True)
    Path("data/therapies").mkdir(parents=True, exist_ok=True)

    (Path("data/patients") / "mario_rossi.json").write_text(
        '{\n'
        '  "patient_id": "mario_rossi",\n'
        '  "name": "Mario Rossi",\n'
        '  "medical_conditions": ["Diabete di tipo 2", "Ipertensione"],\n'
        '  "preferences": ["Preferisce colazione leggera", "Gradisce camminata al mattino"],\n'
        '  "habits": ["Beve acqua spesso durante la giornata"],\n'
        '  "notes": [\n'
        '    {"content": "Evitare zuccheri semplici", "day": null, "created_at": "2026-01-14T10:00:00"}\n'
        '  ]\n'
        '}\n',
        encoding="utf-8",
    )

    (Path("data/patients") / "paolo_verdi.json").write_text(
        '{\n'
        '  "patient_id": "paolo_verdi",\n'
        '  "name": "Paolo Verdi",\n'
        '  "medical_conditions": ["Artrite reumatoide"],\n'
        '  "preferences": ["Preferisce esercizi dolci"],\n'
        '  "habits": ["Riposa nel pomeriggio"],\n'
        '  "notes": [\n'
        '    {"content": "Evitare sforzi intensi nelle prime ore del mattino", "day": null, "created_at": "2026-01-14T10:05:00"}\n'
        '  ]\n'
        '}\n',
        encoding="utf-8",
    )

    (Path("data/caregivers") / "andrea_bianchi.json").write_text(
        '{\n'
        '  "caregiver_id": "andrea_bianchi",\n'
        '  "name": "Andrea Bianchi",\n'
        '  "role": "Infermiere",\n'
        '  "semantic_preferences": ["Quando dico \'sera\' intendo le 21:00"],\n'
        '  "notes": [\n'
        '    {"content": "Preferisce istruzioni concise", "day": null, "created_at": "2026-01-14T10:10:00"}\n'
        '  ]\n'
        '}\n',
        encoding="utf-8",
    )

    (Path("data/caregivers") / "maria_rossi.json").write_text(
        '{\n'
        '  "caregiver_id": "maria_rossi",\n'
        '  "name": "Maria Rossi",\n'
        '  "role": "Caregiver",\n'
        '  "semantic_preferences": ["Quando dico \'mattina\' intendo 08:00-10:00"],\n'
        '  "notes": [\n'
        '    {"content": "Preferisce dettaglio sugli orari", "day": null, "created_at": "2026-01-14T10:12:00"}\n'
        '  ]\n'
        '}\n',
        encoding="utf-8",
    )

    (Path("data/therapies") / "mario_rossi.json").write_text(
        '{\n'
        '  "patient_id": "mario_rossi",\n'
        '  "activities": [\n'
        '    {"activity_id": "mr_001", "name": "Colazione leggera", "description": "Colazione con yogurt e frutta", "day_of_week": ["Lunedì", "Mercoledì", "Venerdì"], "time": "08:00", "dependencies": []},\n'
        '    {"activity_id": "mr_002", "name": "Misurazione glicemia", "description": "Misurare glicemia a digiuno", "day_of_week": ["Lunedì", "Mercoledì", "Venerdì"], "time": "07:30", "dependencies": []},\n'
        '    {"activity_id": "mr_003", "name": "Camminata mattutina", "description": "Passeggiata di 20 minuti", "day_of_week": ["Martedì", "Giovedì"], "time": "09:30", "dependencies": []}\n'
        '  ]\n'
        '}\n',
        encoding="utf-8",
    )

    (Path("data/therapies") / "paolo_verdi.json").write_text(
        '{\n'
        '  "patient_id": "paolo_verdi",\n'
        '  "activities": [\n'
        '    {"activity_id": "pv_001", "name": "Mobilizzazione articolare", "description": "Esercizi leggeri per le articolazioni", "day_of_week": ["Lunedì", "Giovedì"], "time": "10:00", "dependencies": []},\n'
        '    {"activity_id": "pv_002", "name": "Terapia calda", "description": "Applicazione impacchi caldi", "day_of_week": ["Martedì", "Venerdì"], "time": "16:00", "dependencies": []}\n'
        '  ]\n'
        '}\n',
        encoding="utf-8",
    )


async def run_scenario(name, steps, llms, log_file):
    header = f"\n{'='*20} SCENARIO: {name} {'='*20}\n"
    print(header)
    log_file.write(header)

    session.file_path.write_text(f"# Test Session: {name}\n", encoding="utf-8")

    pending_action = False
    confirm_tokens = {"conferma", "ok", "sì", "si", "salva"}
    cancel_tokens = {"annulla", "no", "stop", "cancella"}

    for user_input in steps:
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
        print(f"\n👤 USER: {user_input}")
        log_file.write(step_log)

        log_file.write("🤖 BOT: ")
        print("🤖 BOT: ", end="", flush=True)

        full_response = ""
        async for chunk in run_agent_step(llms, user_input):
            print(chunk, end="", flush=True)
            full_response += str(chunk)
            log_file.write(str(chunk))

        print()
        log_file.write("\n")
        pending_action = "Azione in sospeso" in full_response
        await asyncio.sleep(0.5)


async def main():
    parser = argparse.ArgumentParser(description="Automated KMChat tests with real names")
    parser.add_argument("--model", type=str, default="kmchat-14b", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--output", type=str, default="logs/automated_test_report_realnames.txt", help="Output log file")
    parser.add_argument("--reingest", action="store_true", help="Rebuild RAG index before tests")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    f = open(args.output, "w", encoding="utf-8")
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"KMChat Automated Test Run - {start_time}\nModel: {args.model}\n\n")

    print("🛠️  Seeding data for Mario Rossi and Paolo Verdi...")
    _write_seed_data()

    if args.reingest:
        print("🔁 Re-ingesting RAG index...")
        ingest_data()

    print(f"🔌 Initializing Models ({args.model})...")
    llm = Ollama(
        model=args.model,
        request_timeout=300.0,
        temperature=args.temperature,
        context_window=8192,
        additional_kwargs={"stop": ["Utente:", "\nUtente", "Caregiver:", "\nCaregiver"]},
    )
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    llms = {"FAST": llm, "SMART": llm}

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
