import json
import pandas as pd
import numpy as np
import redis
from langchain.vectorstores.redis import Redis
import os
import textwrap
import openai,azure
from langchain.llms import AzureOpenAI, OpenAI
from langchain.embeddings import OpenAIEmbeddings
import sys
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from redis.commands.search.query import Query

# load the .env file in the parent directory into the current environment
from dotenv import load_dotenv
load_dotenv('./.env')

# setup Llama Index to use Azure OpenAI
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

def create_embeddings(df):
    vectors= []
    for text in df['text']:
        response = openai.Embedding.create(
            input=text,
            engine="text-embedding-ada-002"
        )
        # print the embedding (length = 1536)
        embedding= response["data"][0]["embedding"]
        vectors.append(embedding)
    return vectors




# Function to create indexes in Redis schema
def create_indexes_in_redis(vectors,redis_conn,text_data):
    #### Index creation #####
    pipe = redis_conn.pipeline(transaction=False)
    schema = (
      TextField("text"),
      VectorField("vector", "HNSW", {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"}),
      )
    prefix="doc:abc"
    redis_conn.ft("safe_abc").create_index(fields=schema,
                                        definition=IndexDefinition(prefix=[prefix],index_type=IndexType.HASH)
                                      )

    for i in text_data.keys():
        key=prefix+':' + str(text_data[i]['pagenumber'])
        record = text_data[i]
        record['vector']=np.array(vectors[i]).astype(np.float32).tobytes()
        pipe.hset(key,mapping=record)
    pipe.execute()

# Function to query records from Redis
def query_records_from_redis(redis_conn):
    
    # Perform the similarity search
    query_syntax = "*=>[KNN 5 @vector $vec_param AS vector_score]"

    vss_query=Query(query_syntax).return_fields("text", "vector_score").sort_by("vector_score").dialect(2)

    query_string="Very uncomfortable"
    response=openai.Embedding.create(input=query_string,engine="text-embedding-ada-002")
    embedded_query=np.array([response["data"][0]["embedding"]]).astype(np.float32).tobytes()

    params_dict = {"vec_param": embedded_query}

    vss_results = redis_conn.ft('safe_contractz').search(vss_query, query_params = params_dict)
    
    return vss_results