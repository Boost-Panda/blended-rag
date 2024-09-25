import os
import uuid

import openai
import pinecone
from elasticsearch import Elasticsearch, helpers
from pinecone import Pinecone, ServerlessSpec


class DataLoader:
    def __init__(
        self,
        es_index_name,
        pinecone_index_name,
        pinecone_api_key,
    ):
        print("asdasdsad",es_index_name,
        pinecone_index_name,)
        self.es = Elasticsearch(
            os.getenv("ELASTICSEARCH_URL"), timeout=30, max_retries=10
        )
        self.es_index_name = es_index_name

        # check if Elasticsearch index exists, if not create it
        if not self.es.indices.exists(index=self.es_index_name):
            print(f"Creating Elasticsearch index {self.es_index_name}")
            self.es.indices.create(index=self.es_index_name, ignore=400)
        else:
            print(f"Elasticsearch index {self.es_index_name} already exists")

        # check if Pinecone index exists, if not create it
        pc = Pinecone(api_key=pinecone_api_key, environment="us-west1-gcp")
        self.pc = pc
        self.pinecone_index_name = pinecone_index_name
        if not self.pinecone_index_name in [index.name for index in pc.list_indexes()]:
            print(f"Pinecone index {self.pinecone_index_name} already exists")
            pc.create_index(
                name=pinecone_index_name,
                dimension=1536,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            print(f"Pinecone index {self.pinecone_index_name} already exists")

        # fetch the Pinecone index
        self.pinecone_index = pc.Index(self.pinecone_index_name)

    # Function to create embeddings for a given text
    def create_embeddings(self, text):
        response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
        embeddings = response.data[0].embedding
        return embeddings

    # Function to chunk the document
    def chunk_text(self, text, max_chunk_size=512):
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # Function to read a text file, chunk it, and create embeddings for each chunk
    def save_embeddings_and_documents(self, text, pinecone_index_name, es_index_name):
        
        if pinecone_index_name != "" and es_index_name != "":
            pinecone_index = self.pc.Index(pinecone_index_name)
        else:
            es_index_name = self.es_index_name
            pinecone_index = self.pinecone_index
        
        chunks = self.chunk_text(text)
        for i, chunk in enumerate(chunks):

            # Pinecone Index
            embeddings = self.create_embeddings(chunk)
            chunk_id = f"chunk-{uuid.uuid4()}"
            pinecone_index.upsert([(chunk_id, embeddings)])
            print(f"Stored embeddings and documents for chunk {i+1}/{len(chunks)}")

            # Elasticsearch index
            actions = [
                {
                    "_index": es_index_name,
                    "_id": chunk_id,
                    "_source": {
                        "content": chunk,
                    },
                }
            ]
            helpers.bulk(self.es, actions)

    def delete_indexes(self):
        """
        delete all the elastic search and pinecone indexes
        """
        self.es.indices.delete(index=self.es_index_name, ignore=[400, 404])
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
        pc.delete_index(self.pinecone_index_name)

    def create_indexes(self):
        """
        create elastic search and pinecone indexes
        """
        self.pc.create_index(
            name=self.pinecone_index_name,
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        self.es.indices.create(index=self.es_index_name, ignore=400)
