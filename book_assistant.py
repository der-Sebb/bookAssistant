from langchain_core.embeddings import Embeddings
from chromadb.api.types import EmbeddingFunction
from langchain_community.vectorstores import Chroma
from chromadb.utils import embedding_functions
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import chromadb
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid

class ChromaEmbeddingsAdapter(Embeddings):
    def __init__(self, ef: EmbeddingFunction):
        self.ef = ef

    def embed_documents(self, texts):
        return self.ef(texts)

    def embed_query(self, query):
        return self.ef([query])[0]

class BookAssistant:

    def __init__(self, model_name, vector_db_path, vector_db_collection_name):
        self.model = ChatOllama(model=model_name)
        self.store = {}
        self.chat_id = uuid.uuid4()
        retriever = self.create_vec_db_retriever(vector_db_path, vector_db_collection_name)
        history_aware_retriever = self.create_history_retriever(retriever)
        question_answer_chain = self.create_question_answer_chain()
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def create_vec_db_retriever(self, vector_db_path, vector_db_collection_name):
        embedding_adapter = ChromaEmbeddingsAdapter(embedding_functions.DefaultEmbeddingFunction())
        persistentClient = chromadb.PersistentClient(path=vector_db_path)
        vector_store = Chroma(client=persistentClient, collection_name=vector_db_collection_name, embedding_function=embedding_adapter)
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
        return retriever
    
    def create_history_retriever(self, retriever):
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.model, retriever, contextualize_q_prompt
        )
        return history_aware_retriever
    
    def create_question_answer_chain(self):
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)
        return question_answer_chain

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def answer(self, query):
        return self.conversational_rag_chain.invoke(
            {"input": query},
            config={
                "configurable": {"session_id": self.chat_id}
            },
        )["answer"]