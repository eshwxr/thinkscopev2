import os
from pathlib import Path

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

HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
