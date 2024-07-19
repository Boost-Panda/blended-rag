import os

# load the environment variables
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile

from data_loader import DataLoader
from data_retriever import DataRetriever
from models import TextData
from response_generator import ResponseGenerator

load_dotenv()

es_index_name = os.getenv("ELASTICSEARCH_INDEX_NAME")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# create a FastAPI app
app = FastAPI()

# initialize the components
# data loader is used to load data into Elasticsearch and Pinecone
data_loader = DataLoader(
    es_index_name=es_index_name,
    pinecone_index_name=pinecone_index_name,
    pinecone_api_key=pinecone_api_key,
)

# data retriever is used to retrieve data from Elasticsearch and Pinecone
data_retriever = DataRetriever(
    es_index_name=es_index_name,
    pinecone_index_name=pinecone_index_name,
    pinecone_api_key=pinecone_api_key,
)


# response generator is used to generate responses using OpenAI
response_generator = ResponseGenerator(openai_api_key)


# endpoint to upload a text file and create embeddings for the text
@app.post("/create_embeddings_via_text_file/")
async def upload_text_file(file: UploadFile):
    text = await file.read()  # Read the file content
    text = text.decode("utf-8")

    # add data to the knowledge base
    data_loader.save_embeddings_and_documents(text)

    # return success message along with the status code
    return {"message": "Embeddings created successfully", "success": True}


# endpoint to create embeddings for the text
@app.post("/create_embeddings/")
async def create_embeddings_from_text(data: TextData):
    text = data.text
    # add data to the knowledge base
    data_loader.save_embeddings_and_documents(text)

    return {"message": "Embeddings created successfully", "success": True}


# endpoint to query the data
@app.post("/query_data/")
async def query_data(data: TextData):
    query = data.text
    context = data_retriever.blended_retrieval(query)
    results = response_generator.generate_response(query, context)
    return {"results": results, "success": True}


# delete all the indexes
@app.get("/delete_indexes/")
async def delete_indexes():
    data_loader.delete_indexes()
    return {"message": "Indexes deleted successfully", "success": True}


# create all the indexes
@app.get("/create_indexes/")
async def create_indexes():
    data_loader.create_indexes()
    return {"message": "Indexes created successfully", "success": True}
