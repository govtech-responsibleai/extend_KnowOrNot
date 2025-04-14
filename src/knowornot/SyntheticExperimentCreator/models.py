from pydantic import BaseModel


class CanBeAnswered(BaseModel):
    can_be_answered: bool
