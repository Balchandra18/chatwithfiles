import pandas as pd
import numpy as np
import redis
from langchain.vectorstores.redis import Redis
import os
import openai,azure
from langchain.llms import AzureOpenAI, OpenAI
from langchain.embeddings import OpenAIEmbeddings
import sys
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.base import Document
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from redis.commands.search.query import Query
from embedding_index import create_embeddings

chat = ChatOpenAI(model_name="gpt-3.5",temperature=0.2)


# load the .env file in the parent directory into the current environment
from dotenv import load_dotenv
load_dotenv('./.env')

# Redis connection details
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_password = os.getenv('REDIS_PASSWORD')

r=redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password,ssl=True)
#r.ping()
url='redis_url'
vstore = redis.StrictRedis.from_existing_index(index_name='safe_contractz', embedding=vectors,redis_url=url)
review_chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=vstore.as_retriever())
q="""
The reviews you see are for a product called 'Powerstep Pinnacle Orthotic Shoe Insoles'.
What is the overall impression of these reviews? Give most prevalent examples in bullets.
What do you suggest we focus on improving?
"""

result=review_chain.run(q)
print(result)

# Perform the similarity search
query_syntax = "*=>[KNN 5 @vector $vec_param AS vector_score]"

vss_query=Query(query_syntax).return_fields("text", "vector_score").sort_by("vector_score").dialect(2)

query_string="Very uncomfortable"
response=openai.Embedding.create(input=query_string,engine="YOUR_DEPLOYMENT_NAME")
embedded_query=np.array([response["data"][0]["embedding"]]).astype(np.float32).tobytes()

params_dict = {"vec_param": embedded_query}

vss_results = r.ft('safe_contractz').search(vss_query, query_params = params_dict)

print(vss_results)

docs=[]
for data in vss_results.docs:
    result_string = ''
    result_string += "  Text:" + data.text
    docs.append(Document(page_content=result_string))
    

prompt_template_summary = """
Write a summary of the text:

{text}

The summary should be about five lines long
"""
PROMPT = PromptTemplate(template=prompt_template_summary, input_variables=["text"])
chain = load_summarize_chain(chat, chain_type="stuff", prompt=PROMPT)
summary=chain.run(docs)
print(summary)