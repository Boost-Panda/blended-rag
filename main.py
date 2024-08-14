import os

# load the environment variables
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from gotrue.errors import AuthApiError 
from supabase import create_client, Client


from data_loader import DataLoader
from data_retriever import DataRetriever
from models import TextData
from models import URLData
from response_generator import ResponseGenerator

from crawler import parse_url_and_get_text

load_dotenv()

es_index_name = os.getenv("ELASTICSEARCH_INDEX_NAME")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
# create a FastAPI app
app = FastAPI()
client = create_client(supabase_url, supabase_key)

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


# Function to authenticate the access token
async def authenticate_request(access_token: str):
    try:
        # Attempt to retrieve the user with the provided access token
        response = client.auth.get_user(access_token)
        return response.user
    except AuthApiError as e:
        # If there's an AuthApiError, raise a 401 Unauthorized exception
        raise HTTPException(status_code=401, detail="Invalid or expired access token")

# Function to fetch index names from Supabase
async def get_index_names(user_id: str, project_id: str):
    response = client.table('projects').select('pinecone_index_name', 'elastic_index_name').eq('id', project_id).eq('user_id', user_id).execute()
    data = response.data
    # Check if the data array is empty
    if not data or len(data) == 0:
        raise HTTPException(status_code=404, detail="Project not found or unauthorized")
    # Extract the first dictionary from the data list
    project_data = data[0]
    return project_data['pinecone_index_name'], project_data['elastic_index_name']

# endpoint to create embeddings for the text
@app.post("/create_embeddings/")
async def create_embeddings_from_text(data: TextData, access_token: str = Header(...)):
     # Authenticate the user using the access token
    print(data)
    text = data.text
    project_id= data.project_id
    user = await authenticate_request(access_token)
    user_id = user.id
    pinecone_index_name, elastic_index_name = await get_index_names(user_id, project_id)
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
