import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

print("====================================")
print("        PDF Q&A SYSTEM")
print("====================================")
pdf_path = input("Enter PDF file: ")
if not os.path.exists(pdf_path):
    print("\nPDF file not found.")
    exit()
print("\nLoading PDF...\n")
loader = PyPDFLoader(pdf_path)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
texts = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts,embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOllama(model="phi")
template = """
You are a PDF question-answering assistant.

Answer ONLY using the provided PDF context.

Rules:
- Do NOT use outside knowledge
- Do NOT hallucinate
- If the answer is not found in the context, say:
  'I could not find that information in the PDF.'
- Keep answers concise and factual

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=template,input_variables=["context", "question"])
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

print("PDF loaded successfully.")
print("\nType 'done' to exit.\n")

while True:
    question = input("Ask a question: ")
    if question.lower() == "done":
        print("\nExiting PDF Q&A System...")
        break
    response = qa.invoke({"query": question})

    print("\nAnswer:")
    print(response["result"])
    print()
