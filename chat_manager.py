from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import faiss
from dotenv import find_dotenv, load_dotenv
import os


load_dotenv(find_dotenv('.env'))
# import os
# env_dist = os.environ
# print(env_dist.get('environment'))


class ChatManager(object):

    def __init__(self):
        self.document_dir = "./local_data/"

    def translates_chat(self, in_language, out_language, text):
        chat = ChatOpenAI(temperature=0)
        template = "You are a helpful assistant that translates {input_language} to {output_language}."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        result = chain.run(input_language=in_language, output_language=out_language, text=text)
        return result

    def chat_with_template(self, chat_scene, chat_message):
        chat = ChatOpenAI(temperature=0)
        system_message_prompt = SystemMessagePromptTemplate.from_template(chat_scene)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        result = chain.run(text=chat_message)
        return result

    def init_memory(self):
        embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query
        vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
        memory = VectorStoreRetrieverMemory(retriever=retriever)
        return memory

    def upsert_memory(self, memory, input, output):
        memory.save_context({"input": input}, {"output": output})

    def chat_with_memory(self, memory, chat_message):
        llm = OpenAI(temperature=0)  # Can be any valid LLM
        _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and AI. Now AI uses the artist data stored in the past history to talk to me in the tone of the artist, and all the data about the artist must be obtained from the historical data.
        Relevant pieces of previous conversation:
        {history}

        (You do not need to use these pieces of information if not relevant)

        Current conversation:
        Human: {input}
        AI:"""
        PROMPT = PromptTemplate(
            input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
        )
        conversation_with_summary = ConversationChain(
            llm=llm,
            prompt=PROMPT,
            memory=memory,
            verbose=True
        )
        result = conversation_with_summary.predict(input=chat_message)
        return result

    def init_docsearch(self, document_name):
        loader = DirectoryLoader(self.document_dir, glob=document_name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
        split_docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        docsearch = Chroma.from_documents(split_docs, embeddings)
        qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,
                                        return_source_documents=True)
        return qa

    def query_in_docsearch(self, qa, query):
        result = qa({"query": query})
        return result


if __name__ == '__main__':
    chat_manager = ChatManager()

    # 测试文档query功能
    # qa = chat_manager.init_docsearch("Jay.txt")
    # res = chat_manager.query_in_docsearch(qa, "周杰伦一共出过多少张专辑？")
    # print(res)

    # 测试memory
    memory = chat_manager.init_memory()
    chat_manager.upsert_memory(memory, "Profile of vegaidol community artist Name", "Maggie Smith")
    res = chat_manager.chat_with_memory(memory, "List a artist name in vegaidol community")
    print(res)

    # 测试翻译
    # res = chat_manager.translates_chat("Englist", "Chinese", "Hello World!")
    # print(res)

    # 测试情景对话
    # res = chat_manager.chat_with_template(
    #     "In this conversation, you're a rock star's assistant, and all the questions and technical vocabulary revolve around music",
    #     "Explain the classification of rock")
    # print(res)
