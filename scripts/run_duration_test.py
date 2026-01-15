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

from src.main import run_agent_step, session, reset_rag_index
from src.ingest_data import ingest_data


SCENARIO = [
    "Passa al paziente TestAuto1",
    "Passa al caregiver Andrea",
    "Aggiungi attività 'Controllo pressione' martedì alle 09:00 per i prossimi 2 giorni",
    "Conferma",
    "Dimmi le attività di martedì",
    "Dimmi le attività di mercoledì",
    "Dimmi le attività di giovedì",
    "Dimmi le attività della settimana",
]


def _seed_data() -> None:
    Path("data/patients").mkdir(parents=True, exist_ok=True)
    Path("data/caregivers").mkdir(parents=True, exist_ok=True)
    Path("data/therapies").mkdir(parents=True, exist_ok=True)

    (Path("data/patients") / "TestAuto1.json").write_text(
        '{"patient_id": "TestAuto1", "name": "TestAuto1", "medical_conditions": [], "preferences": [], "habits": [], "notes": []}',
        encoding="utf-8",
    )
    (Path("data/caregivers") / "Andrea.json").write_text(
        '{"caregiver_id": "Andrea", "name": "Andrea", "notes": [], "semantic_preferences": []}',
        encoding="utf-8",
    )
    (Path("data/therapies") / "TestAuto1.json").write_text('[]', encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test duration-day expansion only")
    parser.add_argument("--model", type=str, default="kmchat-14b", help="Model to use")
    parser.add_argument("--output", type=str, default="logs/duration_test_report.txt", help="Output log")
    parser.add_argument("--reingest", action="store_true", help="Rebuild RAG index")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    f = open(args.output, "w", encoding="utf-8")
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"KMChat Duration Test - {start_time}\nModel: {args.model}\n\n")

    _seed_data()

    if args.reingest:
        ingest_data()
        reset_rag_index()

    llm = Ollama(
        model=args.model,
        request_timeout=300.0,
        temperature=0.1,
        context_window=8192,
        additional_kwargs={"stop": ["Utente:", "\nUtente", "Caregiver:", "\nCaregiver"]},
    )
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    llms = {"FAST": llm, "SMART": llm}

    for user_input in SCENARIO:
        step_log = f"\n👤 USER: {user_input}\n"
        print(f"\n👤 USER: {user_input}")
        f.write(step_log)

        f.write("🤖 BOT: ")
        print("🤖 BOT: ", end="", flush=True)

        full_response = ""
        async for chunk in run_agent_step(llms, user_input):
            print(chunk, end="", flush=True)
            full_response += str(chunk)
            f.write(str(chunk))

        print()
        f.write("\n")
        await asyncio.sleep(0.5)

    f.close()
    print(f"\n✅ Test completato. Report salvato in: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
