# Blended RAG App
================

## Introduction
------------

The Blended RAG App is a powerful application that combines the capabilities of Elasticsearch, Pinecone, and OpenAI to provide a seamless and efficient way to create, store, and query text embeddings. This app allows users to upload text files or input text directly, which is then processed to create embeddings and stored in Elasticsearch and Pinecone indexes. Users can then query the stored data using natural language queries, and the app will generate relevant responses using OpenAI's language models.

## Features
--------

* Upload text files or input text directly to create embeddings
* Store text embeddings in Elasticsearch and Pinecone indexes for efficient retrieval
* Query stored data using natural language queries
* Generate relevant responses using OpenAI's language models
* Delete and create indexes as needed

## Requirements
------------

* Python 3.7 or higher
* Elasticsearch 7.x or higher
* Pinecone 1.x or higher
* OpenAI API key

## Installation
------------

1. Clone the repository:
```bash
git clone https://github.com/Boost-Panda/blended-rag.git
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Set up environment variables:
```makefile
PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_INDEX_NAME=<your-pinecone-index-name>
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_ORG=<your-openai-organization>
ELASTICSEARCH_URL=<your-elasticsearch-url>
OPENAI_MODEL_NAME=<your-openai-model-name>
ELASTICSEARCH_INDEX_NAME=<your-elasticsearch-index-name>
```
4. Run the app:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
## Usage
-----

### Create Embeddings

To create embeddings, users can either upload a text file or input text directly.

#### Upload Text File

To upload a text file, use the `/create_embeddings_via_text_file/` endpoint:
```bash
curl -X POST -H "Content-Type: multipart/form-data" -F "file=@path/to/text/file.txt" http://localhost:8000/create_embeddings_via_text_file/
```
#### Input Text Directly

To input text directly, use the `/create_embeddings/` endpoint:
```json
{
    "text": "This is some sample text."
}
```
### Query Data

To query the stored data, use the `/query_data/` endpoint:
```json
{
    "text": "What is this text about?"
}
```
### Delete Indexes

To delete all Elasticsearch and Pinecone indexes, use the `/delete_indexes/` endpoint:
```bash
curl -X GET http://localhost:8000/delete_indexes/
```
### Create Indexes

To create Elasticsearch and Pinecone indexes, use the `/create_indexes/` endpoint:
```bash
curl -X GET http://localhost:8000/create_indexes/
```
Contributing
------------

Contributions are always welcome! If you have any suggestions, improvements, or bug fixes, please open an issue or submit a pull request.
