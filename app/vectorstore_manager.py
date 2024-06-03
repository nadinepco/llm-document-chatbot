import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# Python built-in module for handling temporary files.
import tempfile

class VectorStoreManager:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.vectorstore = None

    def load_and_chunk_docs(self, pdf_docs) -> list:
        print(pdf_docs)
        """Load PDF files, split documents into chunks, and return a list of document chunks."""
        documents = []
        for pdf_doc in pdf_docs:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(pdf_doc.read())
                pdf_path = tmp_file.name
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_chunks = splitter.split_documents(documents)
        return doc_chunks

    def create_vectorstore(self, pdf_docs):
        """Creates a vector store from the document chunks and updates the vectorstore."""
        doc_chunks = self.load_and_chunk_docs(pdf_docs)
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key,model="text-embedding-3-small")
        self.vectorstore = FAISS.from_documents(documents=doc_chunks, embedding=embeddings)
        return self.vectorstore.as_retriever()