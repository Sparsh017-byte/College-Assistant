# College Assistant

A friendly, document-grounded assistant for college students. Ask questions about academic rules or fees in everyday language, and the assistant retrieves relevant information from the college handbook and fee structure before answering. Greetings and other general questions are handled conversationally.

The command-line app uses LangGraph to classify each question, LangChain to retrieve relevant PDF passages, Groq to generate the response, and Pinecone to store and search document embeddings.

## What it can help with

- **Academics:** attendance, exams, grading, credits, promotion, course structure, summer training, and degree requirements.
- **Fees:** tuition, payments, refunds, late charges, scholarships, and other money-related questions.
- **General conversation:** greetings and questions that do not need information from the college documents.

The app can tailor responses to **BCA**, **BBA**, or **B.Com (H)** students. For academic and fee questions, it uses the supplied documents and says when they do not contain enough information.

## Requirements

- Python 3.10 or newer
- A Groq API key
- A Pinecone API key and an existing Pinecone index configured for the `llama-text-embed-v2` embedding model

## Set up

1. Clone or download this project and open a terminal in its folder.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   On macOS or Linux, use:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project folder and add your credentials:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   ```

   Keep `.env` private; it contains secrets and should not be committed.

5. Make sure `academics_handbook.pdf` and `fee_structure.pdf` are present in the project folder. These are the source documents used for retrieval.

## Run

With the virtual environment active, start the assistant:

```bash
python main.py
```

Choose your programme when prompted, then enter questions at the `You:` prompt. Type `exit` or `quit` to end the session.

On startup, the application checks the `academic` and `fee` Pinecone namespaces. If a namespace has no vectors, it loads and splits the corresponding PDF and uploads its chunks. Subsequent runs reuse the existing vectors.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Runs the interactive assistant and routes questions through the LangGraph workflow. |
| `academic_agent.py` | Builds the retriever for academic questions and the college handbook. |
| `fee_agent.py` | Builds the retriever for fee questions and the fee structure. |
| `academics_handbook.pdf` | Academic source document. |
| `fee_structure.pdf` | Fee source document. |
| `requirements.txt` | Python dependencies. |


