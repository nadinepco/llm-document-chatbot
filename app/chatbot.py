from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever

class Chatbot:
    def __init__(self, llm):
        self.llm = llm
        self.rag_chain = None
        self.chat_history = []

    def create_contextualize_q_prompt(self):
        """Creates the prompt template for contextualizing questions."""
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

    def create_qa_prompt(self):
        """Creates the prompt template for answering questions."""
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
        """ Appends chat history """
        self.chat_history.append({"role": role, "content": content})

    def create_rag_chain(self, retriever):
        """Creates the retrieval-augmented generation (RAG) chain."""
        contextualize_q_prompt = self.create_contextualize_q_prompt()
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        qa_prompt = self.create_qa_prompt()
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_response(self, user_input):
        """Generates a response from the RAG chain and updates the chat history."""
        if not self.rag_chain:
            raise ValueError("RAG chain has not been created. Please process a PDF first.")

        for chunk in self.rag_chain.stream({"input": user_input, "chat_history": self.chat_history}):
            if "answer" in chunk:
                yield chunk["answer"]

