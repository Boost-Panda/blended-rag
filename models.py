from pydantic import BaseModel

class TextData(BaseModel):
    text: str
    project_id:str
    # optional pinecone index name, default is empty string
    pinecone_index_name: str = ""
    # optional elastic index name
    elastic_index_name: str = ""
    

class URLData(BaseModel):
    url: str


# model for receiving the image document
class ImageDocument(BaseModel):
    # dict
    image_doc: dict
