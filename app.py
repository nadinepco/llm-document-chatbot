import streamlit as st
from sidebar import sidebar
import os
from dotenv import load_dotenv
from app.llm_manager import LLMManager
from app.vectorstore_manager import VectorStoreManager
from app.chatbot import Chatbot, SpeechProcessor
from streamlit_mic_recorder import mic_recorder
from app.chatbot_utils import METHOD_MAPPING, initialize_session_state


def display_chat_history():
    """
    Displays the chat history of the session
    """
    for message in st.session_state.chatbot.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_user_input():
    """
    Get user input based on the selected chat method, Type or Speak
    """
    if st.session_state.chatbot.rag_chain is not None:
        if st.session_state.chat_method == METHOD_MAPPING["Type"]:
            return st.chat_input(
                placeholder="Ask a question about your documents...",
            )
        elif st.session_state.chat_method == METHOD_MAPPING["Speak"]:
            return mic_recorder(
                start_prompt="Start recording",
                stop_prompt="Stop recording",
                format="webm",
                just_once=True,
                key="stButtonVoice",
            )
    return None


def process_input(prompt, audio):
    """
    Process the input and get a response.
    """
    if audio:
        # Transcribe the audio
        prompt = st.session_state.chatbot.transcribe_speech_to_text(audio["bytes"])

    get_chat_response(prompt)


def get_chat_response(prompt):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = st.session_state.chatbot.get_response(prompt)
        response = st.write_stream(stream)

    # Play response as audio
    if st.session_state.chat_method == METHOD_MAPPING["Speak"]:
        audio_data = st.session_state.chatbot.synthesize_text_to_speech(response)
        st.audio(
            data=audio_data,
            format="audio/mp3",
            autoplay=True,
        )

    # update chat history of the chatbot
    st.session_state.chatbot.append_chat_history("user", prompt)
    st.session_state.chatbot.append_chat_history("assistant", response)


def main():
    st.set_page_config(
        page_title="Conversational Chatbot", page_icon=":books:", layout="wide"
    )
    st.header("Conversational Chatbot :books:")
    st.subheader(
        "Load your PDFs, ask questions, and have a conversation with the chatbot by typing your questions in or by voice chat"
    )

    sidebar()

    ##### MODEL SELECTION #####
    model_type = st.selectbox("Select LLM Model", list(LLMManager.MODEL_MAPPING.keys()))
    initialize_session_state(model_type)

    if model_type == "gpt-3.5-turbo-0125" and not st.session_state.get(
        "OPENAI_API_KEY"
    ):
        st.warning(
            "To start, please enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
        st.stop()

    ##### PDF UPLOADER SECTION #####
    with st.expander("Upload PDF Files", expanded=True):
        # st.title("Document List")
        pdf_docs = st.file_uploader(
            "Upload PDF files here and click on 'Process'",
            type="pdf",
            accept_multiple_files=True,
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                print(pdf_docs)
                retriever = st.session_state.vectorstore_manager.create_vectorstore(
                    pdf_docs
                )
                st.session_state.chatbot.create_rag_chain(retriever)
                st.success("PDF processed!")

    ##### CHAT METHOD SELECTION #####
    st.session_state.chat_method = st.radio(
        "**Let's chat! Type it out or speak up - your choice!**",
        [METHOD_MAPPING["Type"], METHOD_MAPPING["Speak"]],
        horizontal=True,
        disabled=(st.session_state.chatbot.rag_chain is None),
    )

    user_input = get_user_input()

    # Chat history and response container
    with st.container(height=520, border=False):
        display_chat_history()

        # Process the input and get a response
        if isinstance(user_input, str):  # Text input
            process_input(user_input, None)
        elif isinstance(user_input, dict) and "bytes" in user_input:  # Audio input
            process_input(None, user_input)


if __name__ == "__main__":
    load_dotenv()
    main()
