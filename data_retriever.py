# load environment variables
import os

import openai
import pinecone
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


class DataRetriever:
    def __init__(self, es_index_name, pinecone_index_name, pinecone_api_key):
        self.es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))
        self.es_index_name = es_index_name

        self.pc = Pinecone(api_key=pinecone_api_key, environment="us-west1-gcp")
        self.pinecone_index_name = pinecone_index_name

        self.pinecone_index = self.pc.Index(self.pinecone_index_name)

        # initialize OpenAI API
        openai.api_key = os.getenv("OPENAI_API_KEY")

    # Function to create embeddings for a given text
    def create_embeddings(self, text):
        response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
        embeddings = response.data[0].embedding
        return embeddings

    def blended_retrieval(self, query, pinecone_index_name, es_index_name, top_k=3):
        
        if pinecone_index_name != "" and es_index_name != "":
            pinecone_index = self.pc.Index(pinecone_index_name)
        else:
            pinecone_index = self.pinecone_index
            es_index_name = self.es_index_name
        
        # Step 1: BM25 keyword search in Elasticsearch
        bm25_results = self.es.search(
            index=es_index_name,
            body={"query": {"match": {"content": query}}, "size": top_k},
        )
        bm25_docs = [hit["_source"]["content"] for hit in bm25_results["hits"]["hits"]]
        print(f"BM25 Results: Found {len(bm25_docs)} documents")

        # Step 2: Dense vector search in Pinecone
        query_embeddings = self.create_embeddings(query)
        dense_results = pinecone_index.query(
            vector=query_embeddings, top_k=top_k, include_values=True
        )
        matched_chunks = [match["id"] for match in dense_results["matches"]]

        # based on the id, fetch the records from the Elasticsearch
        dense_docs = []
        for id in matched_chunks:
            result = self.es.search(
                index=self.es_index_name,
                body={"size": 1, "query": {"match": {"_id": id}}},
            )
            if len(result["hits"]["hits"]) > 0:
                dense_docs.append(result["hits"]["hits"][0]["_source"]["content"])

        # Combine results from BM25 and Dense Vector search
        combined_docs = list(set(bm25_docs + dense_docs))

        return combined_docs
