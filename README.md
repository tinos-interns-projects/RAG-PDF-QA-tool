# RAG-PDF-QA-tool

A simple Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF document and ask questions based only on the uploaded document.

Built using Streamlit, LangChain, FAISS, Hugging Face embeddings, and Ollama.

---

## Features

- Upload any PDF document
- Ask questions about the uploaded PDF
- Answers are generated only from the PDF context
- Prevents hallucinations using a custom prompt
- Local AI inference using Ollama
- Simple and user-friendly Streamlit interface

---

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Sentence Transformers
- Ollama
- Phi / Mistral models

---

## Project Structure

```text
project/
│
├── app.py          # Streamlit frontend version
├── rag_qa.py       # Terminal-based backend version
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Move into the project folder:

```bash
cd YOUR_PROJECT_NAME
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```powershell
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```
or

```bash
python -m pip install -r requirements.txt
```

---

## Install Ollama

Download and install Ollama from:

[Ollama](https://ollama.com/?utm_source=chatgpt.com)

---

## Download a Model

### Lightweight Option (Recommended for low-memory systems)

```bash
ollama pull phi
```

### Better Quality Option (Requires more RAM)

```bash
ollama pull mistral
```

---

## Choose the Model

In `app.py`, change:

```python
llm = ChatOllama(model="phi")
```

to:

```python
llm = ChatOllama(model="mistral")
```

if your system has enough memory.

---

## Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

---
## Terminal Version

A terminal-based version of the application is also included.

Run it using:

```bash
python rag_qa.py
```

This version allows users to interact with the PDF Q&A system directly from the terminal without the Streamlit frontend.

---
## Working

1. User uploads a PDF
2. PDF text is extracted
3. Text is split into chunks
4. Embeddings are created
5. FAISS stores vector embeddings
6. Relevant chunks are retrieved for questions
7. Ollama generates answers using retrieved context only

---

## Workflow

```text
Upload PDF
    ↓
Ask Question
    ↓
Retrieve Relevant Chunks
    ↓
Generate Answer
    ↓
Display Answer
```

---

## Notes

- The application answers only from the uploaded PDF.
- If information is not found in the document, the system will say so instead of hallucinating.
- A local Ollama model is required to run the project.
- `phi` is faster and lighter.
- `mistral` provides better responses but needs more RAM.

---
