import os
import streamlit as st
from app.llm_manager import LLMManager
from app.chatbot import Chatbot, SpeechProcessor
from app.vectorstore_manager import VectorStoreManager

METHOD_MAPPING = {
    "Type": "Type it out :keyboard:",
    "Speak": "Speak it out :microphone:",
}


def get_api_key(model_type):
    """
    Fetch the appropriate API key based on the model type.

    Args:
        model_type (str): The type of model
    Returns:
        str: The API key for the specified model type
    """
    model_api_map = {
        "llama3-8b-8192": "GROQ_API_KEY",
        "mixtral-8x7b-32768": "GROQ_API_KEY",
        "gpt-3.5-turbo-0125": "OPENAI_API_KEY",
        "deepgram": "DEEPGRAM_API_KEY",
    }
    if model_api_map[model_type] == "OPENAI_API_KEY":
        return st.session_state.get("OPENAI_API_KEY", "")
    return os.environ.get(model_api_map[model_type])


def initialize_session_state(model_type):
    """
    Initialize the LLMManager with the selected model, vectorstore_manager, and chatbot

    Args:
        model_type (str): The type of model
    """
    if (
        "llm_manager" not in st.session_state
        or st.session_state.model_type != model_type
    ):
        api_key = get_api_key(model_type)
        st.session_state.llm_manager = LLMManager(
            api_key=api_key, model_type=model_type
        )
        st.session_state.model_type = model_type

        # initialize the Deepgram Transcriber and inject to the Chatbot
        speech_processor = SpeechProcessor(get_api_key("deepgram"))
        st.session_state.chatbot = Chatbot(
            llm=st.session_state.llm_manager.llm, speech_processor=speech_processor
        )

    if "vectorstore_manager" not in st.session_state:
        st.session_state.vectorstore_manager = VectorStoreManager(
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []
