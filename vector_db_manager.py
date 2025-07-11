import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import config
from ollama_manager import get_ollama_embedding
import traceback 
import uuid 
import logging 

logger = logging.getLogger(__name__) # Get logger instance
logger.setLevel(logging.INFO) # Set level for this module

# Global variables for client and collection to avoid re-initialization IF already done
_client = None
_collection = None

# Custom embedding function wrapper for Ollama
class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input: embedding_functions.Documents) -> embedding_functions.Embeddings:
        embeddings = [get_ollama_embedding(doc) for doc in input]
        return embeddings

# Initialize the custom embedding function once at module level (it's stateless)
ollama_ef = OllamaEmbeddingFunction()

def get_chroma_collection():
    global _client, _collection
    if _collection is None:
        logger.info(f"Initializing ChromaDB client and collection '{config.CHROMA_COLLECTION_NAME}'...")
        _client = chromadb.PersistentClient(path=config.CHROMA_DB_DIRECTORY)
        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=ollama_ef, # Use the custom Ollama embedding function
            metadata={"hnsw:space": "cosine"} # Use cosine distance for similarity
        )
        logger.info(f"ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' initialized.")
    return _collection

def add_documents_to_chroma(chunks: List[str], source_filename: str):
    collection = get_chroma_collection()
    
    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Prepare metadata for each chunk
    metadatas = [{"source": source_filename} for _ in chunks]

    try:
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(chunks)} documents from '{source_filename}' to ChromaDB.")
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB from '{source_filename}': {e}")
        traceback.print_exc()
        raise

def query_chroma_for_context(query_text: str, n_results: int = 4) -> List[Dict[str, Any]]:
    collection = get_chroma_collection()
    
    if not collection.count():
        logger.warning("ChromaDB collection is empty. No context to retrieve.")
        return []

    try:
        # ChromaDB queries using the configured embedding function
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'distances', 'metadatas'] # Include metadata for debugging
        )
        
        retrieved_chunks = []
        if results and results.get('documents') and results['documents'][0]:
            logger.info(f"ChromaDB raw query results for '{query_text[:50]}...':")
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                doc_distance = results['distances'][0][i]
                doc_metadata = results['metadatas'][0][i]
                
                # Log full details of each retrieved chunk before filtering
                logger.info(f"  Chunk {i+1}: Distance={doc_distance:.4f}, Source='{doc_metadata.get('source', 'N/A')}', Content='{doc_content[:100]}...'")
                
                retrieved_chunks.append({
                    "document": doc_content,
                    "distance": doc_distance,
                    "metadata": doc_metadata
                })
        else:
            logger.info(f"ChromaDB query for '{query_text[:50]}...' returned no documents.")

        # The calling function (app.py) will handle filtering by RAG_SCORE_THRESHOLD
        # and re-ranking based on config.RAG_N_RESULTS
        return retrieved_chunks

    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        traceback.print_exc()
        return []

def clear_all_knowledge_base() -> bool:
    """
    Deletes all data from the ChromaDB collection.
    """
    global _client, _collection
    try:
        if _client:
            _client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
            logger.info(f"ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' deleted.")
            # Re-initialize the collection to ensure it's ready for new data
            _collection = _client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                embedding_function=ollama_ef,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' re-created after clearing.")
        else:
            # If client wasn't initialized, just try to get/create to ensure it's clean
            get_chroma_collection() # This will ensure a fresh collection is created if one didn't exist
            logger.info("ChromaDB client not initialized, ensured collection is clean by attempting get_or_create.")
        return True
    except Exception as e:
        logger.error(f"Error clearing ChromaDB knowledge base: {e}")
        traceback.print_exc()
        return False

def get_chunks_by_source(source_filename: str) -> List[Dict[str, Any]]:
    """
    Retrieves all document chunks that originated from a specific source file.
    """
    try:
        collection = get_chroma_collection()
        # Query for all documents where the 'source' metadata matches the filename
        results = collection.query(
            query_texts=[""], # Empty query text, we're filtering by metadata
            where={"source": source_filename},
            n_results=10000, # Retrieve a large number to get (practically) all
            include=['documents', 'metadatas']
        )

        file_chunks = []
        if results and results.get('documents') and results.get('documents')[0]: # Adjusted to access first element of documents list
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                doc_metadata = results['metadatas'][0][i]
                file_chunks.append({"content": doc_content, "metadata": doc_metadata})
        
        logger.info(f"Found {len(file_chunks)} chunks for source: {source_filename}")
        return file_chunks

    except Exception as e:
        logger.error(f"Error retrieving chunks for source '{source_filename}': {e}")
        traceback.print_exc()
        return []