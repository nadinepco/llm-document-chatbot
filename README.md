# 📚 Multi-Modal Conversational RAG Chatbot
## Introduction
This is a **Proof of Concept** of creating a Multi-Modal Conversational Retrieval Augmented Generation (RAG) chatbot that allows users to upload PDF documents and interact with the content through natural language queries. The chatbot can handle both text and voice inputs, making it versatile and user-friendly.

## Features:
- **Document Upload:** Users can upload multiple PDF files to the chatbot.
- **Question Answering:** The chatbot uses different language models to answer questions based on the uploaded documents.
- **Voice Interaction:** Users can interact with the chatbot using voice commands using Deepgram's speech recognition and synthesis.
- **Text-based Interaction:** Users can also interact with the chatbot using text-based input.
- **Remember the Conversation:** The chatbot maintains a history of the conversation for contextual understanding.
- **Model Selection:** Users can choose between different language models for the chatbot's responses. (llama3-8b-8192, mixtral-8x7b-32768, gpt-3.5-turbo-0125)
- **Fallback Response:** If the question is not related to the documents, the chatbot responds with "I don't know".

![chatbot](/images/chatbot.png)

## Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llm-document-chatbot.streamlit.app/)


## Local Setup
1. Prerequisites
    - Python 3.11+
    - OpenAI API Key
    - Deepgram API Key
    - Groq API Key 
2. Clone the repository
    `git clone https://github.com/nadinepco/llm-document-chatbot.git`
3. Install the requirements file in a new environment
    `pip install -r requirements.txt`
4. Create a .env file in the root directory and add your API key:
    `OPENAI_API_KEY=your_openai_api_key 
    DEEPGRAM_API_KEY=your_deepgram_api_key
    GROQ_API_KEY=your_groq_api_key  `
5. Start the streamlit app.
    `streamlit run app.py`

## Resources
- [Open AI](https://openai.com/)
- [LangChain](https://langchain.readthedocs.io/en/latest/index.html): For managing and chaining language models and prompts
- [Streamlit](https://streamlit.io/): For creating the web interface of the chatbot
- [FAISS](https://faiss.ai/index.html): For efficient similarity search 
- [Deepgram](https://deepgram.com/): For speech recognition and synthesis
- [Groq](https://groq.com/): For alternative language models
- [Streamlit Mic Recorder](https://pypi.org/project/streamlit-mic-recorder/): For capturing voice inputs directly from the browser.
