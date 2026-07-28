from pydantic import BaseModel, Field, field_validator

from chat_schemas import ChatHistoryItem


class PetChatRequest(BaseModel):
    petId: int
    petName: str = Field(min_length=1)
    petType: str = Field(min_length=1)
    avoidIngredients: list[str] = Field(default_factory=list)
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatHistoryItem] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question은 공백일 수 없습니다.")
        return stripped
