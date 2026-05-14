import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="PDF Q&A Tool")
st.title("PDF Q&A Tool")
st.write("Upload a PDF and ask questions about it.")
@st.cache_resource(show_spinner=False)
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts,embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="phi")
    template = """
You are a PDF question-answering tool.
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
    return qa
uploaded_file = st.file_uploader("Upload a PDF",type="pdf")
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        qa = process_pdf(pdf_path)
    st.success("PDF is uploaded")
    question = st.text_input("Ask a question about the PDF")
    if question:
        with st.spinner("Generating answer..."):
            response = qa.invoke({"query": question})
            st.subheader("Answer")
            st.write(response["result"])
