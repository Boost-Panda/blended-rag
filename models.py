from pydantic import BaseModel

class TextData(BaseModel):
    text: str
    

class URLData(BaseModel):
    url: str
