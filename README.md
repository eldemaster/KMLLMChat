# KMChat - Reproducibility (macOS/Linux)

This repository contains a local LLM assistant that manages therapy schedules and knowledge for caregivers. The project runs fully offline using Ollama and a local vector store.

## Requirements
- macOS or Linux
- Python 3.11+
- Ollama installed and running

## Setup (macOS/Linux)
From the repo root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r KMChat/requirements.txt
```

## Ollama models
The custom model is built from `qwen2.5:14b` using `KMChat/Modelfile`.
Create it with:
```bash
ollama create kmchat-14b -f KMChat/Modelfile
```
Verify it is available:
```bash
ollama list
```

Embedding model used by the RAG pipeline:
```bash
ollama pull nomic-embed-text
```

Start the Ollama server:
```bash
ollama serve
```

## Data and RAG index
Sample data is stored in:
- `KMChat/data/patients`
- `KMChat/data/caregivers`
- `KMChat/data/therapies`

Build the vector index from JSON data (recommended on first run):
```bash
cd KMChat
python src/ingest_data.py
```

## Run the CLI
From the repo root:
```bash
PYTHONPATH=KMChat python KMChat/src/main.py
```
Or:
```bash
cd KMChat
python -m src.main
```

## Run the Streamlit app
From the repo root:
```bash
streamlit run KMChat/src/app.py
```

## Automated prompt tests
```bash
cd KMChat
python scripts/run_automated_tests.py
```
Log output is saved to `KMChat/logs/automated_test_report.txt`.

## Optional environment variables
- `KMCHAT_STRICT=1` enables strict routing for smaller models
- `KMCHAT_DISABLE_HISTORY=1` disables shared history file
- `KMCHAT_DISABLE_RAG_CONTEXT=1` disables RAG context injection
