from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator

class MCQ(BaseModel):
    question: str = Field(min_length=8)
    options: List[str] = Field(min_length=4, max_length=4)
    correct_answer: str = Field(min_length=1)

    @field_validator("question", "correct_answer")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: List[str]) -> List[str]:
        cleaned = [x.strip() for x in value]
        if any(not x for x in cleaned):
            raise ValueError("Options must not be empty.")
        if len({x.casefold() for x in cleaned}) != 4:
            raise ValueError("Each MCQ must have four distinct options.")
        return cleaned

    @model_validator(mode="after")
    def correct_answer_must_be_option(self):
        if self.correct_answer not in self.options:
            raise ValueError("correct_answer must match one option exactly.")
        return self

class MCQSet(BaseModel):
    questions: List[MCQ] = Field(min_length=1, max_length=10)
