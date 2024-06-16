from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
import requests
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)


class Chatbot:
    def __init__(self, llm, speech_processor):
        self.llm = llm
        self.rag_chain = None
        self.chat_history = []
        self.speech_processor = speech_processor

    def _create_contextualize_q_prompt(self):
        """
        Creates the prompt template for contextualizing questions.

        Returns:
            ChatPromptTemplate: The prompt template for contextualizing questions.
        """
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def _create_qa_prompt(self):
        """
        Creates the prompt template for answering questions.

        Returns:
            ChatPromptTemplate: The prompt template for answering questions.
        """
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Think step by step before providing a detailed answer. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def append_chat_history(self, role, content):
        """
        Appends chat history

        Args:
            role (str): 'human' or 'assistant'
            content (str): the message content
        """
        self.chat_history.append({"role": role, "content": content})

    def create_rag_chain(self, retriever):
        """
        Creates the retrieval-augmented generation (RAG) chain.

        Args:
            retriever (Retriever): The retriever object.
        """
        contextualize_q_prompt = self._create_contextualize_q_prompt()
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        qa_prompt = self._create_qa_prompt()
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

    def get_response(self, user_input):
        """
        Generates a response from the RAG chain and updates the chat history.

        Args:
            user_input (str): The user's input.

        Returns:
            str: The generated response.

        Raises:
            ValueError: If the RAG chain is not created.
        """
        if not self.rag_chain:
            raise ValueError(
                "RAG chain has not been created. Please process a PDF first."
            )

        for chunk in self.rag_chain.stream(
            {"input": user_input, "chat_history": self.chat_history}
        ):
            if "answer" in chunk:
                yield chunk["answer"]

    def transcribe_speech_to_text(self, audio_bytes):
        """
        Transcribes speech to text using DeepgramTranscriber class.

        Args:
            audio_bytes (bytes): The audio bytes to transcribe.

        Returns:
            str: The transcribed text.
        """
        return self.speech_processor.transcribe_audio(audio_bytes)

    def synthesize_text_to_speech(self, text):
        """
        Synthesizes speech from text using SpeechProcessor.

        Args:
            text (str): The text to be converted to speech.

        Returns:
            bytes: The synthesized speech audio data.
        """
        return self.speech_processor.synthesize_speech(text)


class SpeechProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = DeepgramClient(api_key)

    def transcribe_audio(self, audio_bytes):
        """
        Transcribe audio data using the Deepgram API.

        This method takes raw audio data in bytes format, configures the
        transcription options, and sends the audio data to the Deepgram
        API for transcription. It returns the transcribed text.

        Args:
            audio_bytes (bytes): The raw audio data in bytes format to be transcribed.

        Returns:
            str: The transcribed text from the audio data.
        """
        # Create the payload for Deepgram
        payload: FileSource = {
            "buffer": audio_bytes,
        }

        # Set up transcription options
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,  # set to true to include punctuation and capitalization
        )

        # Transcribe the audio and return the transcript in text
        response = self.client.listen.prerecorded.v("1").transcribe_file(
            payload, options
        )
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]

    def synthesize_speech(self, text):
        """
        Synthesize speech from text using the Deepgram API.

        This method takes a text input, sends it to the Deepgram API's text-to-speech endpoint,
        and returns the synthesized speech as audio data in bytes format. If the request fails,
        it raises an exception with the appropriate error message.

        Args:
            text (str): The text to be converted to speech.

        Returns:
            bytes: The audio data in bytes format.

        Raises:
            Exception: If the API request fails with a non-200 status code.
        """
        # Define the API endpoint
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"

        # Define the headers
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        # Define the payload
        payload = {"text": text}

        # Make the POST request
        response = requests.post(url, headers=headers, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Return response
            return response.content
        else:
            # Raise an exception with error message
            raise Exception(f"Error: {response.status_code} - {response.text}")
