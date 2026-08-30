# Agentic Medallion Pipeline Setup Checklist

Use this checklist from the project root before running the application.

## 1. Expected Files

### Source and configuration

- [ ] `README.md`
- [ ] `requirements.txt`
- [ ] `.env.example`
- [ ] `.env` (created locally from `.env.example`; do not commit secrets)
- [ ] `test_import.py`

### Agents

- [ ] `agents/__init__.py`
- [ ] `agents/orchestrator.py`
- [ ] `agents/profiler.py`
- [ ] `agents/sttm_generator.py`
- [ ] `agents/bronze_agent.py`
- [ ] `agents/silver_agent.py`
- [ ] `agents/gold_agent.py`
- [ ] `agents/reporter.py`

### Application and core services

- [ ] `app/streamlit_app.py`
- [ ] `core/__init__.py`
- [ ] `core/config.py`
- [ ] `core/state.py`
- [ ] `core/audit.py`
- [ ] `core/observability.py`
- [ ] `core/memory.py`

### Sample data

- [ ] `data/landing/sales_data.csv`
- [ ] `data/landing/products.csv`
- [ ] `data/landing/stores.csv`

### Tests

- [ ] `tests/__init__.py`
- [ ] `tests/test_audit.py`
- [ ] `tests/test_bronze.py`
- [ ] `tests/test_silver.py`
- [ ] `tests/test_gold.py`
- [ ] `tests/test_profiler.py`
- [ ] `tests/test_reporter.py`
- [ ] `tests/test_sttm.py`

The following directories are created or populated while the application runs: `data/profiles`, `data/sttm`, `data/bronze_layer`, `data/silver_layer`, `data/gold_layer`, `data/traces`, `audit_logs`, `reports`, and `.chroma`.

## 2. Verification Commands

Run these commands in PowerShell from the project root.

### Python version

```powershell
py -V
```

Use Python 3.10 or newer.

### Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V
```

If PowerShell blocks activation, run this once for the current user, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

### Configure environment variables

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set one provider and its API key before starting an agent workflow.

### Import test

```powershell
python .\test_import.py
```

Expected output:

```text
✅ SUCCESS! All imports worked.
```

### Unit tests

```powershell
python -m pytest -q
```

All tests should pass.

### Application startup

```powershell
streamlit run .\app\streamlit_app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## 3. Common Issues and Fixes

### Missing API keys

Set `LLM_PROVIDER` and the matching credential in `.env`:

- `groq`: `GROQ_API_KEY`
- `gemini`: `GOOGLE_API_KEY`
- `openai`: `OPENAI_API_KEY`
- `github`: `GITHUB_TOKEN`

Restart Streamlit after changing `.env`. Do not put real keys in Python files or commit `.env`.

### Import errors

Confirm the virtual environment is active and install dependencies into that same interpreter:

```powershell
python -c "import pandas, pydantic, langchain_core, langgraph, chromadb, dotenv; print('imports ok')"
python -m pip install -r .\requirements.txt
```

Run commands from the project root so imports such as `core.config` and `agents.orchestrator` resolve correctly.

### Streamlit not starting

Check that the command is run from the project root and that the port is available:

```powershell
streamlit run .\app\streamlit_app.py
streamlit run .\app\streamlit_app.py --server.port 8502
```

If PowerShell cannot find Streamlit, use the active environment explicitly:

```powershell
python -m streamlit run .\app\streamlit_app.py
```

### Agent failures

Check the terminal traceback, `audit_logs\<run_id>.jsonl`, and `data\traces\`. Confirm that input files exist, STTM files are non-empty, the selected provider is configured, and the model name is valid. Restart the workflow after correcting the cause.

### Groq `model_not_found` or 404

Groq model IDs can be deprecated or renamed. Check the current model list at [Groq Models](https://console.groq.com/docs/models), then update `.env`:

```dotenv
LLM_PROVIDER="groq"
GROQ_MODEL="current-model-id"
```

`GROQ_MODEL` must be read with `os.getenv(...)` in `core/config.py`; it must not be hardcoded in an agent. Restart Streamlit after changing `.env`.

### STTM file is empty or corrupted

An error such as `Bronze/Silver/Gold STTM file is empty or corrupted` means the STTM LLM response could not be parsed as a non-empty JSON array. Check that `_make_llm()` in `agents/sttm_generator.py` passes `reasoning_format="hidden"` for Groq and that `_invoke_llm_for_sttm_rows` raises a clear error instead of silently writing an empty CSV. Inspect the terminal traceback and `data/traces/` for the raw failure context.

### The same old error appears after fixing code

A stale or duplicate Streamlit process may still be serving old code on port 8501. Check running Python and Streamlit processes:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='streamlit.exe'" | Select-Object ProcessId, CommandLine
```

Kill stray processes using their process IDs:

```powershell
Stop-Process -Id <id> -Force
```

Then start exactly one fresh process:

```powershell
streamlit run .\app\streamlit_app.py
```

## 4. Full Workflow Test

### Upload these files

From `data/landing`, upload:

1. `sales_data.csv`
2. `products.csv`
3. `stores.csv`

The sales file contains deliberate duplicates, missing stores, mixed date formats, inconsistent categories, and return transactions, so it is useful for checking the cleansing stages. The product and store catalogs provide reference data for Gold joins.

### Ask this business question

```text
Which regions and product categories drive the most revenue, and how do returns affect the results?
```

### Expected workflow

1. **Upload and intent:** The UI accepts the three CSV files and the business question, then starts profiling.
2. **Bronze STTM review:** The Profiler creates a profile and the STTM agent proposes Bronze mappings. Review the mappings and approve or deselect rows.
3. **Bronze load:** The Bronze agent writes one Parquet file per uploaded CSV and adds `_load_timestamp` and `_source_file`.
4. **Silver STTM review:** Review proposed cleansing rules for null handling, deduplication, type/date standardization, text normalization, and surrogate keys.
5. **Silver load:** The Silver agent removes duplicate records, cleans values, filters approved columns, and writes Silver Parquet files.
6. **Gold STTM review:** Review proposed joins and aggregations for regional and category analysis.
7. **Gold load:** The Gold agent creates approved analytics tables with joins, aggregations, and `pk_gold_id`.
8. **Executive report:** The Reporter queries Gold data with DuckDB, creates Plotly charts, writes HTML and JSON reports, and displays the HTML report in Streamlit.

At every STTM review, verify that the mapping CSV is non-empty and that the columns and transformations match the business question before approving it.
