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

class CreateData(BaseModel):
    uuid: str