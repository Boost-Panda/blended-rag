import os

# load the environment variables
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile

from data_loader import DataLoader
from data_retriever import DataRetriever
from models import TextData
from models import URLData
from models import ImageDocument
from response_generator import ResponseGenerator

from baygata import (
    get_matched_embeddings,
    save_embeddings_and_documents,
)

from crawler import parse_url_and_get_text

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
    pinecone_index_name = data.pinecone_index_name
    elastic_index_name = data.elastic_index_name

    # add data to the knowledge base
    data_loader.save_embeddings_and_documents(
        text, pinecone_index_name, elastic_index_name
    )

    return {"message": "Embeddings created successfully", "success": True}


# endpoint to create embeddings from the URL
@app.post("/create_embeddings_from_url/")
async def create_embeddings_from_url(data: URLData):
    url = data.url
    text = parse_url_and_get_text(url)
    # add data to the knowledge base
    data_loader.save_embeddings_and_documents(text)

    return {"message": "Embeddings created successfully", "success": True}


# endpoint to query the data
@app.post("/query_data/")
async def query_data(data: TextData):
    query = data.text
    pinecone_index_name = data.pinecone_index_name
    elastic_index_name = data.elastic_index_name

    context = data_retriever.blended_retrieval(
        query, pinecone_index_name, elastic_index_name
    )
    results = response_generator.generate_response(query, context)
    return {"results": results, "success": True}


# create an endpoint for creating embedding for the image document
@app.post("/create_embeddings_from_image/")
async def create_embeddings_from_image(data: ImageDocument):
    print("Receive Image Document")
    print("Data", data)
    image_doc = data.image_doc
    print("Image Document", image_doc)
    chunk_id = save_embeddings_and_documents(image_doc, image_doc.get("_id"))
    return {
        "message": "Embeddings created successfully",
        "chunk_id": chunk_id,
        "success": True,
    }


# query image embeddings
@app.post("/query_embeddings_for_images/")
async def query_embeddings(data: TextData):
    query = data.text
    top_k = 5
    dense_results = get_matched_embeddings(query, top_k)
    return {"results": dense_results, "success": True}


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
