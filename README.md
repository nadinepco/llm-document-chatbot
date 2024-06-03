# 📚 Chat with Multiple PDFs using Streamlit, Langchain, and OpenAI
This is a **Proof of Concept** of creating a Conversational Retrieval Augmented Generation (RAG) chatbot that can answer questions and have a conversation based on the PDF files uploaded

## Current Features:
- Upload multiple PDF files and answer questions about them
- Remember the conversation
- Select different models: llama3-8b-8192, mixtral-8x7b-32768, gpt-3.5-turbo-0125
- If the question is not related to the documents, respond "I don't know"

![chatbot](/images/chatbot.png)

## Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llm-document-chatbot.streamlit.app/)


## Local Setup
1. Clone the repository
    `git clone https://github.com/nadinepco/llm-document-chatbot.git`
2. Install the requirements file
    `pip install -r requirements.txt`
3. Start the streamlit app.
    `streamlit run app.py`

## Resources
- [Open AI](https://openai.com/)
- [LangChain](https://langchain.readthedocs.io/en/latest/index.html)
- [Streamlit](https://streamlit.io/)
- FAISS
- [Groq](https://groq.com/)
