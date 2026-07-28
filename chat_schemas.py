from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


MatchStatus = Literal["MATCHED", "NOT_MATCHED", "UNKNOWN"]
HistoryRole = Literal["user", "assistant"]


class ChatHistoryItem(BaseModel):
    role: HistoryRole
    content: str = Field(min_length=1)


class ChatIngredientResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ocrIngredient: str
    ingredientName: Optional[str] = None
    matchStatus: MatchStatus
    matchedAvoidIngredientId: Optional[int] = None
    matchedAvoidIngredientName: Optional[str] = None
    description: Optional[str] = None
    similarityScore: Optional[float] = None


class ChatSource(BaseModel):
    ocrIngredient: str
    ingredientName: Optional[str] = None
    matchStatus: MatchStatus
    matchedAvoidIngredientName: Optional[str] = None
    description: Optional[str] = None
    similarityScore: Optional[float] = None


class RagChatRequest(BaseModel):
    analysisId: int
    productName: str = Field(min_length=1)
    petName: str = Field(min_length=1)
    petType: str = Field(min_length=1)
    avoidIngredients: list[str]
    ocrText: str
    ingredientResults: list[ChatIngredientResult]
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatHistoryItem] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question은 공백일 수 없습니다.")
        return stripped


class ChatUsage(BaseModel):
    promptTokens: int = Field(ge=0)
    completionTokens: int = Field(ge=0)
    totalTokens: int = Field(ge=0)


class RagChatResponse(BaseModel):
    answer: str
    model: str
    finishReason: str
    usage: ChatUsage
    sources: list[ChatSource]
