import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

class LLMManager:
    MODEL_MAPPING = {
        'llama3-8b-8192': ('groq', 'llama3-8b-8192'),
        'mixtral-8x7b-32768': ('groq', 'mixtral-8x7b-32768'),
        'gpt-3.5-turbo-0125': ('openai', 'gpt-3.5-turbo-0125')
    }

    def __init__(self, api_key, model_type="llama3-8b-8192"):
        self.api_key = api_key
        self.model_type = model_type
        self.llm = self.initialize_llm()

    def initialize_llm(self):
        api_provider, model_name = self.MODEL_MAPPING.get(self.model_type, (None, None))
        if api_provider == "groq":
            return ChatGroq(groq_api_key=self.api_key, model_name=model_name)
        elif api_provider == "openai":
            return ChatOpenAI(openai_api_key=self.api_key, model=model_name)
        else:
            raise ValueError("Invalid LLM model type.")
