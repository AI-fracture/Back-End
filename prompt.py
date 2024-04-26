from pydantic import BaseModel


class Prompt(BaseModel):
    role: str
    content: str
