# config.py

# Paths
DOCUMENTS_DIR = "C:\\Users\\hasee\\Desktop\\NCAI\\DomainSpecificChatbotWebAppBackend\\test"
VDB_DIR = "C:\\Users\\hasee\\Desktop\\NCAI\\DomainSpecificChatbotWebAppBackend\\VectorDBs"
LOG_FILE = "C:\\Users\\hasee\\Desktop\\DomainSpecificChatbotWebAppBackend\\DomainSpecificChatbotWebAppBackend\\V42.log"

# Vector DB
COLLECTION_NAME = "rag-chroma"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 100

# LLM and Embedding Models
MODEL_NAMES = {
    "domain_check": "qwen2.5:3b",
    "context_selection": "qwen2.5:3b",
    "grade_documents": "qwen2.5:3b",
    "rewrite": "qwen2.5:3b",
    "generate": "qwen2.5:3b",
}
EMBEDDING_MODEL = "BAAI/bge-small-en"

# RAG Agent
NUM_OF_CONTEXT = 3
MAX_REWRITES = 3
VERBOSE = True

# API
ALLOWED_ORIGINS = ['*']
API_HOST = "0.0.0.0"
API_PORT = 8000

# Processing
BATCH_SIZE = 100
TEMPERATURE_DOMAIN_CHECK = 0
TEMPERATURE_CONTEXT_SELECTION = 0
TEMPERATURE_GRADE_DOCUMENTS = 1
TEMPERATURE_REWRITE = 0
TEMPERATURE_GENERATE = 0

# Keyword extraction threshold
KEYWORD_IMPORTANCE_THRESHOLD = 0.6

# Retrieval parameters
DEFAULT_RETRIEVAL_K = 10
