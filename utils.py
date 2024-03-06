from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import streamlit as st

def get_pdf_text(pdf_docs):
    '''Extracts text from PDF files and returns a string
    Args:
        pdf_docs (list): list of pdf files
    Returns:
        text (str): string of text from pdf files
    '''
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    '''Splits text into chunks and returns a list of text chunks
    Args:
        raw_text (str): string of text
    Returns:
        text_chunks (list): list of text chunks
    '''
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    text_chunks = splitter.split_text(raw_text)
    return text_chunks

def get_vectorstore(text_chunks):
    '''Creates a vector store and returns it
    Args:
        text_chunks (list): list of text chunks
    Returns:
        vectorstore (FAISS): vector store
    '''
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["OPENAI_API_KEY"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    '''Creates a conversation chain and returns it
    Args:
        vectorstore (FAISS): vector store
    Returns:
        conversation (ConversationalRetrievalChain): conversation chain
    '''
    llm = ChatOpenAI(openai_api_key=st.session_state["OPENAI_API_KEY"])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain