import os
import uuid

import openai
import pinecone
from elasticsearch import Elasticsearch, helpers
from pinecone import Pinecone, ServerlessSpec

es_index_name_baygata = os.getenv("ES_INDEX_BAYGATA")
pc_index_name_baygata = os.getenv("PINECONE_INDEX_BAYGATA")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

es = Elasticsearch([os.getenv("ELASTICSEARCH_URL")])
pc = Pinecone(api_key=pinecone_api_key, environment="us-west1-gcp")

pinecone_index = pc.Index(pc_index_name_baygata)

# keynames that need to be extracted from the image document
keynames = [
    "setting_and_context",
    "colors_and_lighting",
    "mood_and_tone",
    "technical_aspects",
]


def create_embeddings(text):
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    embeddings = response.data[0].embedding
    return embeddings


def conv_image_doc_to_text(image_doc):
    text = ""
    for keyname in keynames:
        # save key and value
        text += keyname + ": " + image_doc.get("analysis", {}).get(keyname, "") + "\n"
    return text


def save_embeddings_and_documents(_doc, chunk_id):
    try:
        text = conv_image_doc_to_text(_doc)
        print("Indexing text into the pinecone", text)
        embeddings = create_embeddings(str(text))
        chunk_id = f"chunk-{chunk_id}"

        pinecone_index.upsert([(chunk_id, embeddings)])
        print(f"Stored embeddings and documents for chunk {chunk_id}")

        # Elasticsearch index
        print(f"Indexing document in Elasticsearch")
        print(_doc)

        # deleting _id from the document
        if "_id" in _doc:
            del _doc["_id"]

        actions = [{"_index": es_index_name_baygata, "_id": chunk_id, "_source": _doc}]
        helpers.bulk(es, actions)
        return chunk_id
    except Exception as e:
        print(f"Error in saving embeddings and documents: {e}")
        return None


def get_matched_embeddings(query, top_k=20):

    # Step 2: Dense vector search in Pinecone
    query_embeddings = create_embeddings(query)
    dense_results = pinecone_index.query(
        vector=query_embeddings, top_k=top_k, include_values=True
    )
    matched_chunks = [match["id"] for match in dense_results["matches"]]

    # based on the id, fetch the records from the Elasticsearch
    dense_docs = []
    for id in matched_chunks:
        result = es.search(
            index=es_index_name_baygata,
            body={"size": 1, "query": {"match": {"_id": id}}},
        )
        if len(result["hits"]["hits"]) > 0:
            dense_docs.append(result["hits"]["hits"][0]["_source"])

    return dense_docs
