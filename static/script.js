document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const pdfUpload = document.getElementById('pdf-upload');
    const uploadBtn = document.getElementById('upload-btn');
    const stopBtn = document.getElementById('stop-btn');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const clearKbBtn = document.getElementById('clear-kb-btn');
    const logContent = document.getElementById('log-content');

    let currentRequestController = null;

    marked.setOptions({
        gfm: true,
        breaks: true,
        sanitize: true,
    });

    function appendMessage(sender, message, isStreaming = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        messageDiv.classList.add('new-message');

        const contentContainer = document.createElement('div');
        contentContainer.classList.add('message-content');

        if (isStreaming) {
            contentContainer.innerHTML = marked.parse(message);
        } else {
            contentContainer.innerHTML = marked.parse(message);
        }

        messageDiv.appendChild(contentContainer);
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function appendLogMessage(message) {
        const logLine = document.createElement('div');
        logLine.classList.add('log-message');

        const levelMatch = message.match(/ - (INFO|WARNING|ERROR|CRITICAL|DEBUG) - /);
        if (levelMatch && levelMatch[1]) {
            logLine.classList.add(levelMatch[1]);
        }

        logLine.textContent = message;
        logContent.appendChild(logLine);
        logContent.scrollTop = logContent.scrollHeight;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        appendMessage('user', message);
        userInput.value = ''; 
        userInput.style.height = 'auto'; 

        sendBtn.disabled = true;
        stopBtn.disabled = false; 

        currentRequestController = new AbortController();
        const signal = currentRequestController.signal;

        try {
            const assistantMessageDiv = document.createElement('div');
            assistantMessageDiv.classList.add('message', 'assistant', 'loading'); 
            const assistantContentContainer = document.createElement('div');
            assistantContentContainer.classList.add('message-content');
            assistantMessageDiv.appendChild(assistantContentContainer);
            chatWindow.appendChild(assistantMessageDiv);
            chatWindow.scrollTop = chatWindow.scrollHeight;

            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
                signal: signal
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to get response from server.');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';

            let isLoadingIndicatorRemoved = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                fullResponse += chunk;

                if (!isLoadingIndicatorRemoved) {
                    assistantMessageDiv.classList.remove('loading');
                    assistantMessageDiv.classList.add('new-message'); 
                    isLoadingIndicatorRemoved = true;
                }
                assistantContentContainer.innerHTML = marked.parse(fullResponse); 
                chatWindow.scrollTop = chatWindow.scrollHeight;
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                appendMessage('system', 'Assistant response stopped by user.');
            } else {
                console.error('Error during chat stream:', error);
                if (!chatWindow.lastChild || !chatWindow.lastChild.textContent.includes('ERROR:')) {
                     appendMessage('system', `Error: ${error.message}`);
                }
            }
        } finally {
            sendBtn.disabled = false;
            stopBtn.disabled = true; 
            currentRequestController = null;
        }
    }

    function stopGeneration() {
        if (currentRequestController) {
            currentRequestController.abort();
            currentRequestController = null;
            stopBtn.disabled = true; 
            sendBtn.disabled = false; 
            console.log("Chat generation stopped.");
        }
    }

    async function uploadPdf() {
        const file = pdfUpload.files[0];
        if (!file) return;

        appendMessage('system', `Uploading document: ${file.name}`);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload_pdf', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (response.ok) {
                appendMessage('system', data.message);
            } else {
                throw new Error(data.error || 'Failed to upload document.');
            }
        } catch (error) {
            console.error('Error uploading document:', error);
            appendMessage('system', `Error: Could not upload document. ${error.message}`);
        } finally {
            pdfUpload.value = '';
        }
    }

    async function clearChatHistory() {
        try {
            const response = await fetch('/clear_chat_history', { method: 'POST' });
            const data = await response.json();
            if (response.ok) {
                chatWindow.innerHTML = '';
                appendMessage('system', data.message);
            } else {
                throw new Error(data.error || 'Failed to clear chat history.');
            }
        } catch (error) {
            console.error('Error clearing chat history:', error);
            appendMessage('system', `Error: Could not clear chat history. ${error.message}`);
        }
    }

    async function clearKnowledgeBase() {
        if (!confirm('Are you sure you want to clear the entire knowledge base? This action cannot be undone.')) {
            return;
        }
        try {
            const response = await fetch('/clear_knowledge_base', { method: 'POST' });
            const data = await response.json();
            if (response.ok) {
                appendMessage('system', data.message);
            } else {
                throw new Error(data.error || 'Failed to clear knowledge base.');
            }
        } catch (error) {
            console.error('Error clearing knowledge base:', error);
            appendMessage('system', `Error: Could not clear knowledge base. ${error.message}`);
        }
    }

    function startLogStream() {
        const eventSource = new EventSource('/stream_logs');

        eventSource.onmessage = function(event) {
            appendLogMessage(event.data);
        };

        eventSource.onerror = function(err) {
            console.error('EventSource failed:', err);
            eventSource.close();
            setTimeout(startLogStream, 3000);
        };

        eventSource.onclose = function() {
            console.log('EventSource closed, attempting to reconnect...');
            setTimeout(startLogStream, 3000);
        };
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    pdfUpload.addEventListener('change', uploadPdf);
    uploadBtn.addEventListener('click', () => pdfUpload.click());

    stopBtn.addEventListener('click', stopGeneration);
    clearChatBtn.addEventListener('click', clearChatHistory);
    clearKbBtn.addEventListener('click', clearKnowledgeBase);

    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    });

    startLogStream();
});