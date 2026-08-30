# 🏅 Agentic Medallion Pipeline

An intent-driven data pipeline that uses AI agents to profile retail CSV data, generate transformation rules, process medallion layers, and produce an analytics report with human approval gates.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Agents](#agents)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Testing](#testing)
- [FAQ](#faq)

## Overview

Traditional data engineering usually requires separate work for data discovery, schema design, transformation code, quality checks, orchestration, and reporting. Changes often require manually updating mappings and coordinating multiple tools before business users can see a result.

With this tool, a user uploads CSV files, asks a business question in plain English, reviews and approves the generated Source-to-Target Mapping (STTM) rules, and receives an executive report. The workflow keeps people in control at each mapping gate while automating the repetitive engineering work.

The agents automatically inspect file structures, calculate profiles, generate layer-specific rules, ingest and cleanse data, join and aggregate analytics tables, run SQL queries, create charts, and record trace and audit information.

## Architecture

```text
+-------------------+
| Streamlit UI      |
| Upload / Review   |
| Report            |
+---------+---------+
          |
          v
+-------------------+
| Supervisor        |
| Orchestrator      |
| State + approvals |
+---------+---------+
          |
          v
+---------------------------------------------------+
| Specialist Agents                                 |
| Profiler | STTM Generator | Bronze | Silver      |
| Gold     | Reporter       | Memory / Observability |
+---------------------------+-----------------------+
                            |
                            v
+---------------------------------------------------+
| Storage                                           |
| Landing CSV | Profiles | STTM | Parquet layers   |
| Reports     | Audit logs | Traces | Chroma memory |
+---------------------------------------------------+
```

| Phase | Agents | Main outputs | Gate |
|---|---|---|---|
| 1. Profile and map Bronze | Profiler, STTM Generator | Profile JSON, Bronze STTM CSV | Bronze STTM approval |
| 2. Load Bronze and map Silver | Bronze, STTM Generator | Bronze Parquet, Silver STTM CSV | Silver STTM approval |
| 3. Load Silver and map Gold | Silver, STTM Generator | Silver Parquet, Gold STTM CSV | Gold STTM approval |
| 4. Load Gold and report | Gold, Reporter | Gold Parquet, HTML and JSON report | Report completion |

## How It Works

### Medallion Flow

```text
Raw CSV files
     |
     v
  Bronze  ---> raw data plus ingestion metadata
     |
     v
  Silver  ---> cleansed, typed, deduplicated data
     |
     v
   Gold   ---> joined and aggregated analytics tables
     |
     v
  Report  ---> SQL-backed answer, analysis, and charts
```

| Layer | Purpose | Typical operations | Storage |
|---|---|---|---|
| Raw / Landing | Preserve user-provided source files | Upload and file retention | `data/landing/*.csv` |
| Bronze | Keep source data with minimal changes | Renames, type casts, load metadata | `data/bronze_layer/*_bronze.parquet` |
| Silver | Make data reliable for downstream use | Null handling, deduplication, standardization, surrogate keys | `data/silver_layer/*_silver.parquet` |
| Gold | Shape data for business analysis | Joins, grouping, aggregations, approved projections | `data/gold_layer/*.parquet` |
| Report | Present business findings | DuckDB SQL, Plotly charts, narrative analysis | `reports/*.html` and `reports/*.json` |

### Agentic Pattern

Each specialist is directed to follow:

```text
THINK -> INSPECT -> PLAN -> ACT -> VERIFY
```

The agents inspect their inputs with tools, perform the requested operation through tools, verify outputs, and expose failures through traces or explicit errors. STTM files are reviewed by a human before the next layer runs.

## Project Structure

```text
medallion-pipeline/
├── agents/
│   ├── orchestrator.py       Supervisor state, phase tools, and workflow entry points
│   ├── profiler.py           CSV structure and statistics profiling agent
│   ├── sttm_generator.py     Bronze, Silver, and Gold mapping-rule generator
│   ├── bronze_agent.py       Raw CSV to Bronze Parquet ingestion
│   ├── silver_agent.py       Cleansing and standardization to Silver Parquet
│   ├── gold_agent.py         Joins and aggregations to Gold Parquet
│   └── reporter.py           DuckDB analysis, charts, and report generation
├── app/
│   └── streamlit_app.py      Human-facing upload, approval, and report UI
├── core/
│   ├── config.py             Paths, provider selection, and environment settings
│   ├── state.py              Pydantic pipeline state model
│   ├── audit.py              JSONL audit logger
│   ├── observability.py      Agent trace persistence
│   └── memory.py             Persistent ChromaDB document memory
├── data/
│   ├── landing/              Input CSV files
│   ├── profiles/             Generated profile JSON files
│   ├── sttm/                 Generated mapping CSV files
│   ├── bronze_layer/         Bronze Parquet files
│   ├── silver_layer/         Silver Parquet files
│   ├── gold_layer/           Gold Parquet files
│   └── traces/               Agent trace JSON files
├── audit_logs/               Per-run JSONL audit records
├── reports/                  Generated HTML and JSON reports
├── tests/                    Unit tests for core agents and helpers
├── requirements.txt          Python dependencies
├── .env.example              Environment variable template
└── test_import.py            Import sanity check
```

## Agents

### Supervisor and Orchestrator

`agents/orchestrator.py` coordinates the workflow, passes results through scratchpads, tracks approval flags, and updates the UI-facing pipeline state. Its phase tools dispatch the specialist agents exactly once per requested step.

### Profiler

`agents/profiler.py` previews CSV files and calculates shapes, types, nulls, unique counts, numeric statistics, and text samples.

Tools: `inspect_files_tool`, `profiler_tool`.

### STTM Generator

`agents/sttm_generator.py` creates Source-to-Target Mapping rules for each layer. It reads profile or Parquet context, validates non-empty LLM JSON arrays, and writes STTM CSV files.

Tools: `inspect_context_tool`, `generate_bronze_sttm_tool`, `generate_silver_sttm_tool`, `generate_gold_sttm_tool`.

### Bronze Agent

`agents/bronze_agent.py` applies Bronze mappings, adds `_load_timestamp` and `_source_file`, and writes Parquet outputs.

Tools: `inspect_task_tool`, `bronze_ingestion_tool`.

### Silver Agent

`agents/silver_agent.py` handles nulls, duplicate rows, type and date normalization, text cleanup, approved columns, and Silver surrogate keys.

Tools: `inspect_task_tool`, `silver_ingestion_tool`.

### Gold Agent

`agents/gold_agent.py` loads Silver tables, joins sources on matching ID columns, applies aggregations, filters approved columns, and writes Gold tables.

Tools: `inspect_task_tool`, `gold_ingestion_tool`.

### Reporter

`agents/reporter.py` registers Gold Parquet files as DuckDB views, executes read-only SQL, stores analysis in a scratchpad, renders Plotly charts, and saves HTML and JSON reports.

Tools: `inspect_gold_tables_tool`, `load_gold_data_tool`, `execute_query_tool`, `submit_analysis_tool`.

## Setup and Installation

### Prerequisites

- Python 3.10 or newer
- Git
- An API key for one supported provider: Groq, Google Gemini, OpenAI, or GitHub Models
- Windows, macOS, or Linux

### Install

```bash
git clone <repository-url>
cd medallion-pipeline
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create the environment file:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` and provide the key for the selected provider. Never commit `.env` or expose API keys in source control.

## Running the Pipeline

Start the Streamlit application from the project root:

```bash
streamlit run app/streamlit_app.py
```

Then upload one or more CSV files, enter a business question, and select **Start Workflow**. Review and approve the Bronze, Silver, and Gold STTM files as each gate appears; the final phase displays and downloads the generated report.

For a quick import sanity check:

```bash
python test_import.py
```

## Configuration

Configuration is loaded by `core/config.py` from `.env` and environment variables.

| Variable | Purpose | Example |
|---|---|---|
| `LLM_PROVIDER` | Provider selection | `groq`, `gemini`, `openai`, or `github` |
| `GROQ_API_KEY` | Groq authentication | `your_groq_api_key_here` |
| `GROQ_MODEL` | Groq model override | `llama-3.1-8b-instant` |
| `GOOGLE_API_KEY` | Gemini authentication | `your_google_api_key_here` |
| `GEMINI_MODEL` | Gemini model override | `gemini-2.5-flash` |
| `OPENAI_API_KEY` | OpenAI authentication | `your_openai_api_key_here` |
| `OPENAI_MODEL` | OpenAI model override | `gpt-4o-mini` |
| `GITHUB_TOKEN` | GitHub Models authentication | `your_github_token_here` |
| `GITHUB_MODEL` | GitHub model override | `openai/gpt-4.1-mini` |
| `GITHUB_BASE_URL` | GitHub Models endpoint | `https://models.inference.ai.azure.com` |

The application creates its data directories automatically. Generated profiles, mappings, Parquet files, reports, audit logs, traces, and Chroma data remain in the project directories configured by `core.config`.

## Testing

Run the full test suite from the project root:

```bash
python -m pytest -q
```

The tests cover audit logging, profiling, STTM context and parsing, Bronze/Silver/Gold transformations, reporting tools, and orchestrator behavior.

## FAQ

### Do I need to write transformation code for every dataset?

No. The STTM Generator proposes layer-specific rules from the available profile and table context. You review and approve those rules before the next phase runs.

### Does the application send raw data to an embedding service?

No. The profiling and transformation workflow uses local CSV, Parquet, and DuckDB operations. The memory helper is configured for basic ChromaDB document storage without embeddings.

### What happens if an agent fails?

The orchestrator records the error in pipeline state, while `AgentTrace` and `AuditLogger` preserve diagnostic history for the run. The UI can display the audit trail and restart the analysis.

### Where are generated files stored?

Inputs are stored under `data/landing`, intermediate artifacts under `data/profiles`, `data/sttm`, and the layer directories, and final reports under `reports`.

### Can I change the model provider?

Yes. Set `LLM_PROVIDER` and the corresponding credentials in `.env`, then restart Streamlit. Provider-specific model names can be overridden with the matching `*_MODEL` variable.

### Is human approval required?

Yes. The UI provides approval gates for Bronze, Silver, and Gold STTM rules so users can edit or deselect mappings before execution continues.
