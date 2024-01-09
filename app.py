import streamlit as st
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

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
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    '''Creates a conversation chain and returns it
    Args:
        vectorstore (FAISS): vector store
    Returns:
        conversation (ConversationalRetrievalChain): conversation chain
    '''
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def display_conversation():
    ''' Displays the chat history of the session '''
    if st.session_state.chat_history is not None:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                # display user input
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                # display the response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown(message.content)

def handle_user_input(user_question):
    ''' Displays the latest input user and gets the response from the vector database
     Args:
        user_question: string of user input
    '''
    # get response from the conversation chain by retrieving from the vector database
    response = st.session_state.conversation({'question': user_question})
    
    # update the chat history session state with the latest question and response
    st.session_state.chat_history = response['chat_history']

    # display the latest question and response. The response is being displayed by chunks that will visually look like it is
    # being typed
    for i, message in enumerate(st.session_state.chat_history[-2:]):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                if user_question != None:
                    # Simulate stream of response with milliseconds delay
                    for chunk in message.content.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

def initialize_session_state():
    ''' Initialize session states '''
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:", layout="wide")
    st.header("Chat with Multiple PDFs :books:")

    # load .env file keys
    load_dotenv()
    initialize_session_state()
    display_conversation()

    if user_question := st.chat_input(placeholder="Ask a question about your documents..."):
        handle_user_input(user_question)

    with st.sidebar:
        st.title("Document List")
        pdf_docs = st.file_uploader("Upload PDF files here and click on 'Process'", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
    

if __name__ == "__main__":
    main()