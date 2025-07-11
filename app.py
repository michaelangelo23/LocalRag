# app.py

from flask import Flask, request, render_template, jsonify, Response, stream_with_context
import os
import sys
import io
from collections import deque
import json
import threading
import traceback
import subprocess
import time
import re
import logging
import queue 
import shutil 

# FIX: Set stdout and stderr encoding to UTF-8 for consistent output, especially on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import config
from document_processor import load_pdf_text, load_text_file_content, load_docx_text, load_xlsx_text, split_text_into_chunks
from vector_db_manager import add_documents_to_chroma, query_chroma_for_context, get_chroma_collection, clear_all_knowledge_base, get_chunks_by_source
from ollama_manager import get_ollama_chat_stream, get_ollama_completion
from werkzeug.utils import secure_filename

import utils

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set base level for the logger

# Create handlers (existing ones)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('app.log')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Formatter for console and file handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

# Add existing handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Queue for real-time log streaming to frontend
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    """Custom logging handler to push log records to a queue."""
    def emit(self, record):
        log_entry = self.format(record)
        log_queue.put(log_entry)

queue_handler = QueueHandler()
queue_handler.setLevel(logging.INFO) # Log levels INFO and above will be streamed
queue_handler.setFormatter(formatter) # Use the same formatter
logger.addHandler(queue_handler)


# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Used for session management (if any)

# Global flag to indicate if initial setup is complete
initial_setup_complete = False

# --- Chat History Management ---
chat_history = deque(maxlen=config.MAX_HISTORY_MESSAGES) # Stores recent messages

def add_message_to_history(role, content):
    """Adds a message to the chat history."""
    chat_history.append({"role": role, "content": content})
    logger.info(f"Added to history: {role} - {content[:50]}...") # Log message addition

def get_chat_history():
    """Returns the current chat history as a list."""
    return list(chat_history)

def clear_chat_history():
    """Clears the chat history."""
    chat_history.clear()
    logger.info("Chat history cleared.")

# --- Routes ---
@app.route('/')
def index():
    """Renders the main chat interface."""
    return render_template('index.html')

# Endpoint for streaming server logs
@app.route('/stream_logs')
def stream_logs():
    """Streams server logs in real-time using Server-Sent Events (SSE)."""
    def generate_logs():
        while True:
            if not log_queue.empty():
                log_message = log_queue.get()
                # SSE format: data: [message]\n\n
                yield f"data: {log_message}\n\n"
            time.sleep(0.05) # Small delay to prevent busy-waiting and reduce CPU usage

    return Response(stream_with_context(generate_logs()), mimetype='text/event-stream')


@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages and streams responses from Ollama."""
    if not initial_setup_complete:
        return jsonify({"error": "System still initializing. Please wait a moment."}), 503

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    add_message_to_history("user", user_message)
    logger.info(f"User message received: {user_message}")

    try:
        context_chunks = []
        
        logger.info("Attempting RAG query for context...")
        query_results = query_chroma_for_context(user_message, n_results=config.RAG_PRE_RANK_N_RESULTS)
        filtered_results = [
            res for res in query_results
            if res.get('distance', 1.0) <= config.RAG_SCORE_THRESHOLD
        ]
        
        # Simple re-ranking: take the top N after filtering
        context_chunks = [res['document'] for res in filtered_results[:config.RAG_N_RESULTS]]
        
        if context_chunks:
            logger.info(f"RAG retrieved {len(context_chunks)} relevant chunks.")
        else:
            logger.info("RAG query returned no relevant chunks for the current message.")

        # Prepare messages for Ollama
        messages = []

        # Add system prompt
        messages.append({"role": "system", "content": config.DEFAULT_SYSTEM_PROMPT})

        # Add RAG context if available
        if context_chunks:
            context_string = "\n\n".join(context_chunks)
            messages.append({"role": "system", "content": f"Here is some relevant information from the knowledge base:\n{context_string}\n\nBased on the above context, answer the user's question. If the information is not sufficient, state that you cannot answer from the provided context."})
            logger.info("RAG context added to messages.")
        else:
            # If no context was found, inform the LLM that it should answer without external knowledge
            messages.append({"role": "system", "content": "No additional context was retrieved from the knowledge base for this query. Answer based on your general knowledge."})
            logger.info("No RAG context available, informing LLM to use general knowledge.")


        # Add recent chat history (excluding summary if current history is short)
        history_to_add = list(chat_history)[-config.MAX_UNSUMMARIZED_MESSAGES:] # Get recent unsunmmarized messages
        messages.extend(history_to_add)
        logger.debug(f"Messages sent to LLM: {messages}")


        def generate_response():
            full_response_content = ""
            try:
                for chunk in get_ollama_chat_stream(messages):
                    full_response_content += chunk
                    yield chunk
            except Exception as e:
                logger.error(f"Error during streaming response: {e}")
                traceback.print_exc()
                yield f"ERROR: An error occurred during response generation: {e}"
            finally:
                add_message_to_history("assistant", full_response_content)
                logger.info("Assistant response streamed and added to history.")

        return Response(stream_with_context(generate_response()), mimetype='text/plain')

    except Exception as e:
        logger.error(f"Failed to get chat response: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to get chat response: {e}"}), 500


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Handles PDF file uploads, processes them, and adds to knowledge base."""
    if 'file' not in request.files:
        logger.warning("No file part in upload request.")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file for upload.")
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(config.PDF_DIRECTORY, filename)
        
        # Ensure the directory exists
        os.makedirs(config.PDF_DIRECTORY, exist_ok=True)

        try:
            file.save(file_path)
            logger.info(f"File saved temporarily: {file_path}")

            # Determine file type and load text
            file_extension = os.path.splitext(filename)[1].lower()
            text_content = ""
            if file_extension == '.pdf':
                text_content = load_pdf_text(file_path)
            elif file_extension == '.txt' or file_extension == '.md':
                text_content = load_text_file_content(file_path)
            elif file_extension == '.docx':
                text_content = load_docx_text(file_path)
            elif file_extension == '.xlsx':
                text_content = load_xlsx_text(file_path)
            else:
                logger.warning(f"Unsupported file type uploaded: {file_extension}")
                os.remove(file_path) # Clean up unsupported file
                return jsonify({"error": f"Unsupported file type: {file_extension}. Only PDF, TXT, MD, DOCX, XLSX are supported."}), 400

            if not text_content.strip():
                logger.warning(f"No text extracted from file: {filename}. It might be an image-based PDF or empty.")
                os.remove(file_path) # Clean up empty file
                return jsonify({"message": f"Document '{filename}' uploaded, but no text could be extracted. It might be an image-based PDF or empty."}), 200


            # Split text into chunks
            chunks = split_text_into_chunks(text_content)
            logger.info(f"Split '{filename}' into {len(chunks)} chunks.")

            # Add chunks to ChromaDB
            if chunks:
                add_documents_to_chroma(chunks, filename)
                logger.info(f"'{filename}' chunks added to ChromaDB.")
                
                # Move the processed file to 'done_documents' directory
                utils.move_file_to_directory(file_path, config.DONE_DIRECTORY)
                logger.info(f"Moved '{filename}' to done directory.")

                return jsonify({"message": f"Document '{filename}' processed and added to knowledge base."}), 200
            else:
                os.remove(file_path) # Clean up if no chunks were generated
                logger.warning(f"No chunks generated for '{filename}'.")
                return jsonify({"message": f"Document '{filename}' uploaded but no content was extracted for the knowledge base."}), 200

        except Exception as e:
            logger.error(f"Error processing uploaded file '{filename}': {e}")
            traceback.print_exc()
            # Attempt to clean up the partially processed file
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": f"Failed to process document: {e}"}), 500
    return jsonify({"error": "Unexpected error during file upload."}), 500


@app.route('/get_uploaded_documents', methods=['GET'])
def get_uploaded_documents():
    """Retrieves a list of all documents currently in the knowledge base."""
    try:
        collection = get_chroma_collection()
        # Fetch all unique source metadata values
        results = collection.get(
            where={}, # Get all documents
            ids=[], # Do not query by specific IDs
            limit=10000, # Max number of documents to return, adjust if you expect many more
            include=['metadatas']
        )
        
        unique_sources = set()
        if 'metadatas' in results and results['metadatas']:
            for metadata in results['metadatas']:
                if 'source' in metadata:
                    unique_sources.add(metadata['source'])
        
        return jsonify({"documents": sorted(list(unique_sources))}), 200
    except Exception as e:
        logger.error(f"Error fetching uploaded documents: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to retrieve uploaded documents."}), 500


@app.route('/delete_document', methods=['POST'])
def delete_document():
    """Deletes a specific document and its associated chunks from the knowledge base."""
    document_name = request.json.get('document_name')
    if not document_name:
        return jsonify({"error": "No document name provided"}), 400

    try:
        collection = get_chroma_collection()
        # Delete documents where the 'source' metadata matches the document_name
        collection.delete(where={"source": document_name})
        
        # Optionally, delete the physical file from the 'done_documents' directory
        done_file_path = os.path.join(config.DONE_DIRECTORY, document_name)
        if os.path.exists(done_file_path):
            os.remove(done_file_path)
            logger.info(f"Deleted physical file: {done_file_path}")

        logger.info(f"Document '{document_name}' and its chunks deleted from ChromaDB.")
        return jsonify({"message": f"Document '{document_name}' deleted from knowledge base."}), 200
    except Exception as e:
        logger.error(f"Error deleting document '{document_name}': {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to delete document: {e}"}), 500


@app.route('/clear_chat_history', methods=['POST'])
def clear_chat_history_route():
    """Endpoint to clear the entire chat history."""
    clear_chat_history()
    return jsonify({"message": "Chat history cleared."}), 200

@app.route('/clear_knowledge_base', methods=['POST'])
def clear_knowledge_base_route():
    """Endpoint to clear the entire knowledge base."""
    success = clear_all_knowledge_base()
    if success:
        # Also clear the 'done_documents' directory
        if os.path.exists(config.DONE_DIRECTORY):
            try:
                shutil.rmtree(config.DONE_DIRECTORY)
                os.makedirs(config.DONE_DIRECTORY) # Recreate the empty directory
                logger.info("Cleared 'done_documents' directory.")
            except Exception as e:
                logger.error(f"Error clearing 'done_documents' directory: {e}")
                return jsonify({"error": f"Knowledge base cleared, but failed to clear document directory: {e}"}), 500

        return jsonify({"message": "Knowledge base cleared successfully."}), 200
    else:
        return jsonify({"error": "Failed to clear knowledge base."}), 500


# --- Initial Setup for Application Start ---
def _initial_setup_thread():
    """
    Performs initial setup tasks in a separate thread.
    """
    global initial_setup_complete
    logger.info("Running initial setup...")

    try:
        # Ensure directories exist
        os.makedirs(config.PDF_DIRECTORY, exist_ok=True)
        os.makedirs(config.CHROMA_DB_DIRECTORY, exist_ok=True)
        os.makedirs(config.DONE_DIRECTORY, exist_ok=True)

        get_chroma_collection() # Initialize the single ChromaDB collection
        logger.info("ChromaDB initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")

    add_message_to_history("system", config.DEFAULT_SYSTEM_PROMPT)
    logger.info("Default system message added to chat history.")

    initial_setup_complete = True
    logger.info("Initial setup complete.")


if __name__ == '__main__':
    logger.info("--- Initializing BreezeAI Assistant Backend ---")

    setup_thread = threading.Thread(target=_initial_setup_thread)
    setup_thread.start()
    setup_thread.join() # Wait for the setup to complete before starting the Flask app

    logger.info("--- BreezeAI Assistant Backend Ready ---")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)