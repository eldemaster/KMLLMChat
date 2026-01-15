import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _parse_size(size_str: str) -> float:
    match = re.match(r"^([0-9.]+)\\s*([KMG]B)$", size_str.strip(), re.IGNORECASE)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2).upper()
    factor = {"KB": 1e3, "MB": 1e6, "GB": 1e9}.get(unit, 1.0)
    return value * factor


def _list_ollama_models() -> list[tuple[str, float]]:
    output = subprocess.check_output(["ollama", "list"], text=True)
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    models = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0]
        size = parts[2]
        if "embed" in name.lower():
            continue
        models.append((name, _parse_size(size)))
    return models


def _pick_models(models: list[tuple[str, float]]) -> tuple[str, str]:
    if not models:
        raise RuntimeError("Nessun modello Ollama trovato.")
    models_sorted = sorted(models, key=lambda item: item[1])
    small = models_sorted[0][0]
    large = models_sorted[-1][0]
    if small == large and len(models_sorted) > 1:
        small = models_sorted[0][0]
    return large, small


def _run_tests(model: str, output_path: Path, temperature: float, strict: bool) -> None:
    cmd = [
        sys.executable,
        "scripts/run_automated_tests.py",
        "--model",
        model,
        "--temperature",
        str(temperature),
        "--output",
        str(output_path),
    ]
    env = dict(os.environ)
    if strict:
        env["KMCHAT_STRICT"] = "1"
    else:
        env.pop("KMCHAT_STRICT", None)
    subprocess.check_call(cmd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated tests with big and small Ollama models.")
    parser.add_argument("--model-large", type=str, help="Override large model name.")
    parser.add_argument("--model-small", type=str, help="Override small model name.")
    parser.add_argument("--output-dir", type=str, default="logs", help="Directory for log outputs.")
    parser.add_argument("--temperature-large", type=float, default=0.1, help="Temperature for large model.")
    parser.add_argument("--temperature-small", type=float, default=0.0, help="Temperature for small model.")
    args = parser.parse_args()

    models = _list_ollama_models()
    large, small = _pick_models(models)
    if args.model_large:
        large = args.model_large
    if args.model_small:
        small = args.model_small

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    large_log = output_dir / "automated_test_report_large.txt"
    small_log = output_dir / "automated_test_report_small.txt"

    print(f"==> Large model: {large}")
    _run_tests(large, large_log, args.temperature_large, strict=False)

    print(f"==> Small model: {small}")
    _run_tests(small, small_log, args.temperature_small, strict=True)

    print(f"Logs saved to {large_log} and {small_log}")


if __name__ == "__main__":
    main()
