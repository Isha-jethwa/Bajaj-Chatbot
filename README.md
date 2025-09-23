# Bajaj Document QA Chatbot

Ask questions about PDFs in the `docs/` folder using retrieval-augmented QA. The app builds or loads a FAISS index over your PDFs and answers with a lightweight local QA model.

## Quickstart

1. Python 3.12 recommended (Windows/macOS/Linux).
2. Create and activate a virtual environment:
   - Windows PowerShell
     ```ps1
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS/Linux
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Put your PDFs in the `docs/` directory.
5. Start the Streamlit UI:
   ```bash
   streamlit run app.py
   ```
   - Default URL: `http://localhost:8501` (if taken, Streamlit will suggest another port).

The app will automatically build `faiss_index.idx` and `chunks.pkl` if missing. You can also rebuild from the sidebar button.

## Command-line chat (optional)
If you prefer a simple CLI:
```bash
python chat_local.py
```
Type your question, or `exit`/`quit` to leave.

## How it works
- Embeddings: `all-MiniLM-L6-v2` via Sentence Transformers
- Vector store: FAISS L2 index
- Retriever: top-k chunk search over embedded PDF text
- Reader: `distilbert-base-cased-distilled-squad` QA pipeline

## Project structure
- `docs/` — place PDFs here
- `prepare_data_local.py` — manual indexing script (extract → chunk → embed → build FAISS)
- `chat_local.py` — CLI demo using the saved index
- `app.py` — Streamlit chat UI (auto-builds index if missing)
- `faiss_index.idx`, `chunks.pkl` — generated artifacts

## Manual index build (optional)
The UI builds automatically, but you can run the standalone script:
```bash
python prepare_data_local.py
```

## Deployment (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Create a new Streamlit app, select your repo.
3. Main file: `app.py`
4. Python version: 3.12
5. The platform installs from `requirements.txt` and boots automatically.

### Notes for Streamlit Cloud
- The requirements are pinned to work on both Windows (local) and Linux (Cloud):
  - FAISS uses `faiss-cpu==1.12.0` on Windows and `faiss-cpu==1.8.0.post1` on Linux via environment markers.
  - Torch is pinned to `2.4.0` for reliable CPU wheels on Cloud.
- If the installer fails, try “Restart” in Cloud. If it persists:
  - Clear the app cache in the Cloud app settings.
  - Ensure there is enough disk space (remove large artifacts).
  - If you still hit errors, open the app logs and share the error block.

## Troubleshooting
- Port already in use:
  - Change port: `streamlit run app.py --server.port 8502`
  - Windows: free 8501
    ```ps1
    netstat -ano | findstr :8501
    taskkill /PID <pid> /F
    ```
- Firewall prompt: allow Python for local access, or bind to localhost explicitly:
  ```bash
  streamlit run app.py --server.address localhost --server.port 8503
  ```
- Slow first run: models download on first use; subsequent runs are faster (cached).
- Index doesn’t update after adding PDFs: use the sidebar “Rebuild index from docs” button.
- Tokenizer warning about `clean_up_tokenization_spaces`: safe to ignore (upstream deprecation notice).
- If packages fail to install, ensure pip is recent:
  ```bash
  python -m pip install --upgrade pip
  ```

## Notes
- This project runs entirely locally with CPU-friendly models. For larger models or GPU acceleration, adjust `requirements.txt` and model names accordingly. 
