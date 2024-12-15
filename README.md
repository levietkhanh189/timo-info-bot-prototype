# Information Retrieval Chatbot with FastAPI and LangChain

This project is an information retrieval chatbot built using FastAPI and LangChain. It utilizes the OpenAI API for answering user queries based on a set of documents (PDF files). The chatbot can return context-aware answers, extract specific information from documents, and engage in friendly, human-like conversations with users.

## Features

- **Document-based Q&A**: The chatbot retrieves relevant information from uploaded documents (PDF format) and answers user queries.
- **Social Interaction**: The chatbot is able to respond to common social greetings like "Hi", "Hello", "How are you?", and more in a human-like and friendly manner.
- **FastAPI-powered API**: The project exposes an API endpoint for querying, which is built using FastAPI, a fast and modern web framework for Python.
- **Customizable Instructions**: The system allows setting an initial prompt for context that helps the AI assistant answer more effectively and in a natural tone.

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**:
  - FastAPI
  - Uvicorn
  - LangChain
  - LangChain-OpenAI
  - FAISS (for efficient vector-based search)
  - python-dotenv

## Setup Instructions

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/levietkhanh189/timo-info-bot-prototype
   cd timo-info-bot-prototype
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"  # On Windows use `set` instead of `export`
   ```

5. Ensure that your PDF documents are placed in a `data` folder inside the project directory.

6. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

7. The API will be available at `http://127.0.0.1:8000`. You can check the health status of the API at `http://127.0.0.1:8000/healthcheck`.

## API Endpoints

- **POST `/ask`**: Submit a query to the chatbot.
  - Request body: 
    ```json
    {
      "query": "Your question here"
    }
    ```
  - Response:
    ```json
    {
      "question": "Your question here",
      "answer": "Chatbot's answer",
      "source_documents": [List of documents referenced]
    }
    ```

- **GET `/healthcheck`**: Check the health status of the server.
  - Response: 
    ```json
    {
      "status": "ok"
    }
    ```

## Example Query

Example request:
```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"query": "What is the name of the author in the document?"}'
```

## Notes

- This project is designed to retrieve information from PDF files placed in the `data` folder. You can easily extend it by adding more PDF documents or adapting the document loader.
- The chatbot uses OpenAI’s language model to process the queries and provide natural-sounding responses.
- Make sure to keep your OpenAI API key secure, and do not expose it in public repositories.
