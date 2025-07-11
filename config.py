import os

# --- Directory Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for storing uploaded PDF files
PDF_DIRECTORY = os.path.join(BASE_DIR, "documents")

# Directory for ChromaDB persistence (unified for all knowledge)
CHROMA_DB_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")

# Path for curated knowledge JSON file
CURATED_KNOWLEDGE_FILE = os.path.join(BASE_DIR, "curated_knowledge.json")

# Directory for storing processed files (after RAG/summary generation)
DONE_DIRECTORY = os.path.join(BASE_DIR, "done_documents")


# --- Ollama Model Configurations ---
# The large language model (LLM) used for chat responses
OLLAMA_CHAT_MODEL = "goekdenizguelmez/JOSIEFIED-Qwen3:8b" # Your preferred chat model

# The embedding model used for converting text into vectors for the vector database
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" # Your preferred embedding model

# --- Application Settings ---
# Maximum number of messages to keep in the chat history
MAX_HISTORY_MESSAGES = 20 # Keep recent context, adjust as needed

# Number of most recent messages to keep unsummarized
MAX_UNSUMMARIZED_MESSAGES = 10 # Keep this many recent messages unsummarized

# ChromaDB Collection Name
CHROMA_COLLECTION_NAME = "breezeai_knowledge"

# Default system prompt for the Ollama model
DEFAULT_SYSTEM_PROMPT = """You are an great AI  Assistant."""

# --- Concurrency Settings ---
# Maximum number of worker processes for parallel tasks (e.g., document processing)
MAX_PROCESS_WORKERS = 6 # Adjust based on your CPU cores and available memory

# --- RAG (Retrieval Augmented Generation) Settings ---
# Number of top relevant results to retrieve from ChromaDB for initial context
RAG_PRE_RANK_N_RESULTS = 50 # Increased from 10: Retrieve more chunks initially for better filtering

# Number of final re-ranked relevant results to include in the context
RAG_N_RESULTS = 24 # Increased from 4: Pass more top relevant chunks to the LLM

# Similarity score threshold for RAG results (ChromaDB uses cosine distance, so lower is better)
# A value of 0.5 means documents with a cosine distance of 0.5 or less are considered relevant.
# Increased from 0.5 to 0.75: Less strict, allows for slightly less perfect matches, which can be useful
RAG_SCORE_THRESHOLD = 0.95 # Adjust based on your embedding model and desired strictness.... i think this is too strict dude


# --- Text Splitting for RAG ---
# Size of text chunks for the vector database (in characters)
TEXT_CHUNK_SIZE = 1000

# Overlap between consecutive text chunks (in characters)
TEXT_CHUNK_OVERLAP = 200