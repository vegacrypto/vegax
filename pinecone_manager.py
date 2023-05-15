import openai
import pinecone
from tqdm.auto import tqdm
from datasets import load_dataset
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import find_dotenv, load_dotenv
import params
import os

# load env
load_dotenv(find_dotenv('.env'))
env_dist = os.environ
# print(env_dist.get('environment'))


class VectorManager(object):
    def __init__(self):
        self.local_data = params.LOCAL_DATA
        self.embeddings = OpenAIEmbeddings()
        self.embedding_model = params.EMBEDDING_MODEL
        pinecone.init(
            api_key=env_dist.get("PINECONE_API_KEY"),
            environment=params.PINECONE_ENVIRONMENT
        )

    def create_index(self, index_name, demension=params.OPENAI_DEMENSION):
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=demension)

    def upsert_index(self, txt_name, index_name=params.PINECONE_INDEX_NAME):
        index = pinecone.Index(index_name)
        # TODO:查看trec的数据形式
        trec = load_dataset('trec', split='train[:1000]')
        # process everything in batches of 32
        batch_size = 32
        for i in tqdm(range(0, len(trec['text']), batch_size)):
            i_end = min(i + batch_size, len(trec['text']))

            # create embeddings
            lines_batch = trec['text'][i: i + batch_size]
            res = openai.Embedding.create(input=lines_batch, engine=self.embedding_model)

            # fix vector data
            ids_batch = [str(n) for n in range(i, i_end)]
            embeds = [record['embedding'] for record in res['data']]
            meta = [{'text': line} for line in lines_batch]
            to_upsert = zip(ids_batch, embeds, meta)

            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert))

    def query_index(self, query, index_name=params.PINECONE_INDEX_NAME):
        index = pinecone.Index(index_name)
        xq = openai.Embedding.create(input=query, engine=self.embedding_model)['data'][0]['embedding']
        res = index.query([xq], top_k=5, include_metadata=True)
        return res
