import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = BASE_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
PROFILES_DIR = DATA_DIR / "profiles"
STTM_DIR = DATA_DIR / "sttm"
BRONZE_DIR = DATA_DIR / "bronze_layer"
SILVER_DIR = DATA_DIR / "silver_layer"
GOLD_DIR = DATA_DIR / "gold_layer"

# Output folders
REPORTS_DIR = BASE_DIR / "reports"
AUDIT_DIR = BASE_DIR / "audit_logs"
CHROMA_DIR = BASE_DIR / ".chroma"

# AI provider environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini")
GITHUB_BASE_URL = os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# Provider selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER")

if LLM_PROVIDER is None:
    if GITHUB_TOKEN:
        LLM_PROVIDER = "github"
    elif OPENAI_API_KEY:
        LLM_PROVIDER = "openai"
    elif GROQ_API_KEY:
        LLM_PROVIDER = "groq"
    elif GOOGLE_API_KEY:
        LLM_PROVIDER = "gemini"
    else:
        LLM_PROVIDER = "groq"

# Normalize provider values
LLM_PROVIDER = LLM_PROVIDER.lower()

# Ensure required directories exist
for directory in [
    DATA_DIR,
    LANDING_DIR,
    PROFILES_DIR,
    STTM_DIR,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    REPORTS_DIR,
    AUDIT_DIR,
    CHROMA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
