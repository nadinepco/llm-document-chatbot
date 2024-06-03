import streamlit as st
from sidebar import sidebar
import os
from dotenv import load_dotenv
from app.llm_manager import LLMManager
from app.vectorstore_manager import VectorStoreManager
from app.chatbot import Chatbot

def get_api_key(model_type):
    """Fetch the appropriate API key based on the model type."""
    model_api_map = {
        'llama3-8b-8192': "GROQ_API_KEY",
        'mixtral-8x7b-32768': "GROQ_API_KEY",
        'gpt-3.5-turbo-0125': "OPENAI_API_KEY"
    }
    if model_api_map[model_type] == "OPENAI_API_KEY":
        return st.session_state.get("OPENAI_API_KEY", "")
    return os.environ.get(model_api_map[model_type])

def initialize_session_state(model_type):
    """Initialize the LLMManager with the selected model, vectorstore_manager, and chatbot"""
    if 'llm_manager' not in st.session_state or st.session_state.model_type != model_type:
        api_key = get_api_key(model_type)
        st.session_state.llm_manager = LLMManager(api_key=api_key, model_type=model_type)
        st.session_state.model_type = model_type
        st.session_state.chatbot = Chatbot(llm=st.session_state.llm_manager.llm)

    if 'vectorstore_manager' not in st.session_state:
        st.session_state.vectorstore_manager = VectorStoreManager(openai_api_key=os.environ["OPENAI_API_KEY"])
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
def display_conversation():
    """ Displays the chat history of the session """
    for message in st.session_state.chatbot.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])



def main():
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:", layout="wide")
    st.header("Chat with your PDFs :books:")
    st.subheader("Load your PDFs, ask questions, and have a conversation with the chatbot")

    sidebar()

    model_type = st.selectbox('Select LLM Model', list(LLMManager.MODEL_MAPPING.keys()))
    initialize_session_state(model_type)

    if model_type == 'gpt-3.5-turbo-0125' and not st.session_state.get("OPENAI_API_KEY"):
        st.warning("To start, please enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys.")
        st.stop()

    with st.expander("Upload PDF Files", expanded=True):
        # st.title("Document List")
        pdf_docs = st.file_uploader("Upload PDF files here and click on 'Process'", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                print(pdf_docs)
                retriever = st.session_state.vectorstore_manager.create_vectorstore(pdf_docs)
                st.session_state.chatbot.create_rag_chain(retriever)
                st.success("PDF processed!")
        
    display_conversation()
    if st.session_state.chatbot.rag_chain is not None:
        # Accept user input
        if prompt := st.chat_input(placeholder="Ask a question about your documents..."):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                stream = st.session_state.chatbot.get_response(prompt)
                response = st.write_stream(stream)

            # update chat history
            st.session_state.chatbot.append_chat_history("user", prompt)
            st.session_state.chatbot.append_chat_history("assistant", response)
        

if __name__ == "__main__":
    load_dotenv()
    main()