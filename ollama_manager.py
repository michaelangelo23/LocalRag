import ollama
import config # Import config to get model names
import traceback # Import for full tracebacks

def get_ollama_chat_stream(messages: list[dict]):

    try:
        # Use the chat method with streaming enabled
        stream = ollama.chat(model=config.OLLAMA_CHAT_MODEL, messages=messages, stream=True)
        for chunk in stream:
            # Check if there's content in the chunk and yield it
            if 'content' in chunk['message']:
                yield chunk['message']['content']
    except ollama.ResponseError as e:
        print(f"Ollama Response Error (Chat): {e}")
        yield f"ERROR: Ollama chat model responded with an error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during Ollama chat streaming: {e}")
        yield f"ERROR: An unexpected error occurred: {e}"

def get_ollama_completion(messages: list[dict], model: str = config.OLLAMA_CHAT_MODEL, temperature: float = 0.0) -> str:

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=False # Request a non-streaming response
        )
        return response['message']['content']
    except ollama.ResponseError as e:
        print(f"Ollama Response Error (Completion): {e}")
        traceback.print_exc()
        return f"ERROR: Ollama completion model responded with an error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during Ollama completion: {e}")
        traceback.print_exc()
        return f"ERROR: An unexpected error occurred: {e}"


def get_ollama_embedding(text: str) -> list[float]:
    try:
        response = ollama.embeddings(model=config.OLLAMA_EMBEDDING_MODEL, prompt=text)
        return response['embedding']
    except ollama.ResponseError as e:
        print(f"Ollama Response Error (Embedding): {e}")
        raise RuntimeError(f"Ollama embedding model responded with an error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during Ollama embedding: {e}")
        traceback.print_exc()
        raise RuntimeError(f"An unexpected error occurred: {e}")