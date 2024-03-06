import streamlit as st
import time
from dotenv import load_dotenv
from sidebar import sidebar
from utils import get_conversation_chain, get_pdf_text, get_text_chunks, get_vectorstore

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
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:", layout="wide")
    st.header("Chat with your PDFs :books:")
    st.subheader("Load your PDFs and ask questions")

    # load .env file keys
    # load_dotenv()
    sidebar()
    initialize_session_state()

    if not st.session_state["OPENAI_API_KEY"]:
        st.warning(
            "To start, please enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
        st.stop()
    
    
    with st.expander("Upload PDF Files"):
        # st.title("Document List")
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
    
    display_conversation()
    if (st.session_state.conversation) is not None:
        if user_question := st.chat_input(placeholder="Ask a question about your documents..."):
            handle_user_input(user_question)

    

if __name__ == "__main__":
    main()