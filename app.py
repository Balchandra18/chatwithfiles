from flask import Flask, request, jsonify
import pandas as pd
import openai
import redis
import logging
import os
from embedding_index import create_embeddings,create_indexes_in_redis
from redis_vector_queries import query_records_from_redis

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load the .env file in the parent directory into the current environment
from dotenv import load_dotenv
load_dotenv('./.env')

# Redis connection details
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_password = os.getenv('REDIS_PASSWORD')

# Set up Redis connection
redis_conn=redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password,ssl=True)
print(redis_conn.ping())

# Route to process Form Recognizer JSON and apply OpenAI embeddings
@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        # Get JSON input data
        json_data = request.json

        df = pd.DataFrame.from_dict(json_data)
        text_data=df.head(100).to_dict(orient='index')
        
        
        # Create OpenAI embeddings
        vectors=create_embeddings(df)
        logger.info("OpenAI embeddings applied")
        
        # Create indexes in Redis schema
        create_indexes_in_redis(vectors, redis_conn,text_data)
        logger.info("Indexes created in Redis")        

        return jsonify({"message": "Data processing successful"})
    except Exception as e:
        logger.error(f"Error in processing data: {str(e)}")
        return jsonify({"error": "Data processing failed"})


# Route to query results from Redis
@app.route('/query_results', methods=['GET'])
def query_results():
    try:
        # Example: Query all records from Redis
        results = query_records_from_redis(redis_conn)
        logger.info("Results queried from Redis")

        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in querying results: {str(e)}")
        return jsonify({"error": "Querying results failed"})

# Function to query records from Redis?



if __name__ == '__main__':
    app.run(debug=True)
