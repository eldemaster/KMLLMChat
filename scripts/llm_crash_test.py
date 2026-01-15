import argparse
import asyncio
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

from src.main import run_agent_step


DEFAULT_PROMPTS = [
    "Cambia contesto su paziente demo_giuseppe e caregiver caregiver_01.",
    "Mostrami le attività di Lunedì",
    "Aggiungi attività Passeggiata lunedì alle 12:00 con force=True",
    "Modifica l'attività Passeggiata di lunedì: spostala alle 13:00",
    "Mostrami le attività di Lunedì",
    "Consulta le linee guida della terapia del paziente",
    "Cambia contesto su paziente paziente_02 e caregiver caregiver_01.",
    "Mostrami le attività di Lunedì",
    "Aggiungi attività Riabilitazione lunedì alle 07:00 dipende da Chirurgia",
    "Aggiungi attività Controllo pressione lunedì alle 09:00",
]


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        prompts.append(stripped)
    return prompts or DEFAULT_PROMPTS


def log_block(fp, title: str, content: str):
    fp.write("\n" + "=" * 80 + "\n")
    fp.write(f"{title}\n")
    fp.write("-" * 80 + "\n")
    fp.write(content.strip() + "\n")


async def run(prompts: list[str], output_path: Path, model_fast: str, model_smart: str, prompt_timeout: int, subprocess_mode: bool):
    llms = None
    if not subprocess_mode:
        llm_fast = Ollama(model=model_fast, request_timeout=120.0, temperature=0.1)
        llm_smart = Ollama(model=model_smart, request_timeout=120.0, temperature=0.2)
        Settings.llm = llm_smart
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        llms = {"FAST": llm_fast, "SMART": llm_smart}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(f"KMChat LLM Crash Test - {datetime.now().isoformat()}\n")
        fp.write(f"Model FAST: {model_fast}\n")
        fp.write(f"Model SMART: {model_smart}\n")
        fp.write(f"Total prompts: {len(prompts)}\n")
        fp.flush()

        for i, prompt in enumerate(prompts, start=1):
            full = ""
            if subprocess_mode:
                try:
                    env = os.environ.copy()
                    env["PYTHONPATH"] = "."
                    result = subprocess.run(
                        [
                            sys.executable,
                            "src/main.py",
                            "--model-fast",
                            model_fast,
                            "--model-smart",
                            model_smart,
                            "--test-prompt",
                            prompt,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=prompt_timeout,
                        env=env,
                    )
                    full = (result.stdout + "\n" + result.stderr).strip()
                except subprocess.TimeoutExpired:
                    full = f"TIMEOUT after {prompt_timeout}s"
            else:
                try:
                    async def _collect():
                        nonlocal full
                        async for chunk in run_agent_step(llms, prompt):
                            full += str(chunk)
                    await asyncio.wait_for(_collect(), timeout=prompt_timeout)
                except asyncio.TimeoutError:
                    full = f"TIMEOUT after {prompt_timeout}s"
            log_block(fp, f"PROMPT {i}", prompt)
            log_block(fp, f"REPLY {i}", full)
            fp.flush()


def main():
    parser = argparse.ArgumentParser(description="Run LLM crash test and log output.")
    parser.add_argument("--model-fast", default="kmchat-14b")
    parser.add_argument("--model-smart", default="kmchat-14b")
    parser.add_argument("--output", default="logs/llm_crash_test.log")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--prompt-timeout", type=int, default=240)
    parser.add_argument("--subprocess", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    output_path = Path(args.output)
    asyncio.run(run(prompts, output_path, args.model_fast, args.model_smart, args.prompt_timeout, args.subprocess))
    print(f"Done. Log written to: {output_path}")


if __name__ == "__main__":
    main()
