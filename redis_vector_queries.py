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
from redis.commands.search.field import NumericField

from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from redis.commands.search.query import Query


# load the .env file in the parent directory into the current environment
from dotenv import load_dotenv
load_dotenv('./.env')


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