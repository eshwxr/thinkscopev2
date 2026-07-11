import os
from pathlib import Path

from dotenv import load_dotenv

# Loaded here, not in each entrypoint, so env vars are populated no matter
# which module gets imported first (import order otherwise left this None).
load_dotenv()

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHROMA_DIR = DATA_DIR / "chroma_db"
EVAL_DIR = APP_DIR / "eval"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen/qwen3-32b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

MAX_RETRIEVAL_RETRIES = 2
RETRIEVAL_TOP_K = 5

# "hybrid" (default) or "dense" -- lets the same pipeline be measured with
# either retrieval method (e.g. Phase 1 checkpoint before/after comparison).
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid")

HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
