from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory
import pickle


RESPONSE_TEMPLATE = """You are an expert programmer and problem-solver, tasked to answer any question about the Aeon Mecha GitHub repository. Using the provided context, answer the user's question to the best of your ability.
Generate a comprehensive and informative answer for a given question based solely on the provided context. Do not assume that the user has any knowledge of the context, or that they are able to look through the context.
Use an unbiased and journalistic tone. Do not repeat text.
If there is nothing in the context relevant to the question at hand, just say "Sorry, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
REMEMBER: If there is no relevant information within the context, just say "Sorry, I'm not sure." Don't try to make up an answer.
Answer neatly in Markdown:"""

REPHRASE_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History: {chat_history}
Follow Up Input: {question}
Standalone Question:"""


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
QA_PROMPT = PromptTemplate(template=RESPONSE_TEMPLATE, input_variables=["question","context"])


def load_retriever():
    # with open("vectorstore.pkl", "rb") as f:
    #     vectorstore = pickle.load(f)
    vectorstore = Chroma(persist_directory='chroma_vectorstore', embedding_function=OpenAIEmbeddings())
    # retriever = VectorStoreRetriever(vectorstore=vectorstore)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 50}
        )
    return retriever


def get_basic_qa_chain():
    llm = ChatOpenAI(model_name="gpt-4",temperature=0)
    retriever = load_retriever()
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True)
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True)
    return model


def get_custom_prompt_qa_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/6635
    # see: https://github.com/langchain-ai/langchain/issues/1497
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_condense_prompt_qa_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/5890
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_qa_with_sources_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = load_retriever()
    history = []
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True)

    def model_func(question):
        # bug: this doesn't work with the built-in memory
        # hacking around it for the tutorial
        # see: https://github.com/langchain-ai/langchain/issues/5630
        new_input = {"question": question['question'], "chat_history": history}
        result = model(new_input)
        history.append((question['question'], result['answer']))
        return result

    return model_func