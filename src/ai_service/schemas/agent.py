from pydantic import BaseModel


class ShoppingAnswer(BaseModel):
    answer: str
