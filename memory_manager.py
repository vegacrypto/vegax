from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
import faiss
from dotenv import find_dotenv, load_dotenv
import params
import os

load_dotenv(find_dotenv('.env'))


# import os
# env_dist = os.environ
# print(env_dist.get('environment'))


class MemoryManager(object):

    def __init__(self):
        self.document_dir = params.LOCAL_DATA
        self.total_memory = dict()

    def save_memory_to_local(self, chat_id):
        if chat_id in self.total_memory:
            memory_manager.total_memory[chat_id].retriever.vectorstore.save_local(self.document_dir, str(chat_id))
            return True
        else:
            return False

    def load_local_memory(self, chat_id):
        if chat_id not in self.total_memory:
            self.init_new_memory(chat_id)
        memory_manager.total_memory[chat_id].retriever.vectorstore.load_local(self.document_dir,
                                                                              embeddings=OpenAIEmbeddings(),
                                                                              index_name=str(chat_id))
        return True

    def init_new_memory(self, chat_id):
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query
        vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
        memory = VectorStoreRetrieverMemory(retriever=retriever)
        self.total_memory[chat_id] = memory

    def view_memory(self, chat_id):
        try:
            return self.total_memory[chat_id]
        except:
            return {}

    def remove_memory(self, chat_id):
        try:
            del self.total_memory[chat_id]
        except:
            pass

    def upsert_memory(self, chat_id, query, answer):
        if chat_id in self.total_memory:
            self.total_memory[chat_id].save_context({"input": query}, {"output": answer})
            return True
        else:
            return False


if __name__ == '__main__':
    memory_manager = MemoryManager()
    memory_manager.init_new_memory(1)
    memory_manager.upsert_memory(1, "Profile of vegaidol community artist Name", "周杰伦")
    memory_manager.upsert_memory(1, "周杰伦对人类的回复会是什么样的风格？", "诙谐幽默喜欢开玩笑")
    print(memory_manager.total_memory[1])
    print("*" * 50)
    print(memory_manager.total_memory[1].memory_variables)
    print("*" * 50)
    print(memory_manager.total_memory[1].retriever)
    print("*" * 50)
