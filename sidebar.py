import streamlit as st

def sidebar():
    with st.sidebar:
        st.markdown(
            "# How to use \n"
            "1. Choose a model \n"
            "2. If chosen model is from OpenAI, enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below\n"  
            "3. Upload PDF Files \n"
            "4. Ask a question and start a conversation about the documents \n"
        )
        
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  
            value=st.session_state.get("OPENAI_API_KEY", ""),
        )
        st.session_state.OPENAI_API_KEY = api_key_input

        

        st.markdown("---")
        st.markdown("# About")
        st.markdown("""
                    This is a work in progress and is a tool to show proof of concept 
                    that you can ask questions and have a conversation about your documents. \n
                    [View the source code](https://github.com/nadinepco/llm-document-chatbot)
                    """)