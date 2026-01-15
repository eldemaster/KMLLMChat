import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

from src.main import run_agent_step


PROMPTS = [
    "Cambia contesto su paziente demo_giuseppe e caregiver caregiver_01.",
    "Mostrami le attività di Lunedì",
    "Consulta le linee guida della terapia del paziente",
    "Il paziente ha diabete",
    "conferma",
    "Il paziente preferisce fare il riposino alle 15",
    "conferma",
]


def log_block(fp, title: str, content: str):
    fp.write("\n" + "=" * 80 + "\n")
    fp.write(f"{title}\n")
    fp.write("-" * 80 + "\n")
    fp.write(content.strip() + "\n")


async def run(output_path: Path, model_fast: str, model_smart: str, prompt_timeout: int):
    llm_fast = Ollama(model=model_fast, request_timeout=120.0, temperature=0.1)
    llm_smart = Ollama(model=model_smart, request_timeout=120.0, temperature=0.2)
    Settings.llm = llm_smart
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    llms = {"FAST": llm_fast, "SMART": llm_smart}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(f"KMChat LLM Focus Test - {datetime.now().isoformat()}\n")
        fp.write(f"Model FAST: {model_fast}\n")
        fp.write(f"Model SMART: {model_smart}\n")
        fp.write(f"Total prompts: {len(PROMPTS)}\n")
        fp.flush()

        for i, prompt in enumerate(PROMPTS, start=1):
            full = ""
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
    parser = argparse.ArgumentParser(description="Focused LLM test with confirmations.")
    parser.add_argument("--model-fast", default="kmchat-14b")
    parser.add_argument("--model-smart", default="kmchat-14b")
    parser.add_argument("--output", default="logs/llm_focus_test.log")
    parser.add_argument("--prompt-timeout", type=int, default=180)
    args = parser.parse_args()

    os.environ.setdefault("KMCHAT_DISABLE_HISTORY", "1")
    asyncio.run(run(Path(args.output), args.model_fast, args.model_smart, args.prompt_timeout))
    print(f"Done. Log written to: {args.output}")


if __name__ == "__main__":
    main()
