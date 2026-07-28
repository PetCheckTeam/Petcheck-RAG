import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from clova_client import get_clova_embedding
from database import IngredientKnowledge, get_db
from ingredient_extractor import extract_ingredients

# .env 파일 로드
load_dotenv()

# 필수 환경변수 존재 여부 체크
REQUIRED_ENV_VARS = [
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "CLOVA_STUDIO_API_KEY",
    "DATABASE_URL",
]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    print(f"⚠️ 경고: 다음 환경변수가 설정되지 않았습니다: {', '.join(missing_vars)}")


app = FastAPI(
    title="PetCheck RAG Engine API",
    description="OCR 원료를 추출하고 Clova Embedding과 pgvector로 검색합니다.",
    version="1.0.0",
)


# --- DTO (Pydantic 스키마) ---
class RagSearchRequest(BaseModel):
    analysisId: int
    ocrText: str = Field(min_length=1)
    topK: int = Field(default=1, ge=1, le=10)
    petType: Optional[str] = "DOG"


class ContextItem(BaseModel):
    ocrIngredient: str
    ingredientName: str
    description: Optional[str] = None
    similarityScore: float


class RagSearchResponse(BaseModel):
    analysisId: int
    extractedIngredients: list[str]
    totalCount: int
    contexts: list[ContextItem]


# --- 보조 함수: 중복 성분 제거 ---
def deduplicate_ingredients(ingredients: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in ingredients:
        cleaned = item.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


# --- API 엔드포인트 ---
@app.post("/api/v1/rag/search", response_model=RagSearchResponse)
def search_rag_context(
    request: RagSearchRequest,
    db: Session = Depends(get_db),
) -> RagSearchResponse:
    # 1. OCR 텍스트에서 성분 추출 및 중복 제거
    raw_ingredients = extract_ingredients(request.ocrText)
    ingredients = deduplicate_ingredients(raw_ingredients)

    if not ingredients:
        raise HTTPException(
            status_code=422,
            detail="OCR 텍스트에서 원료명을 찾지 못했습니다.",
        )

    contexts: list[ContextItem] = []

    # 2. 키워드 정확 일치 검색 후, 일치하지 않으면 벡터 유사도 검색
    for ingredient in ingredients:
        keyword_match = (
            db.query(IngredientKnowledge)
            .filter(
                (IngredientKnowledge.alias_name == ingredient)
                | (IngredientKnowledge.standard_name == ingredient)
            )
            .first()
        )

        if keyword_match:
            contexts.append(
                ContextItem(
                    ocrIngredient=ingredient,
                    ingredientName=keyword_match.standard_name,
                    description=keyword_match.description,
                    similarityScore=1.0,
                )
            )
            continue

        try:
            query_vector = get_clova_embedding(ingredient)
            sql = text(
                """
                SELECT
                    id,
                    ingredient_id,
                    standard_name,
                    alias_name,
                    description,
                    1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM ingredient_knowledge
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :top_k
                """
            )
            results = db.execute(
                sql,
                {"vec": str(query_vector), "top_k": request.topK},
            ).fetchall()

            for row in results:
                contexts.append(
                    ContextItem(
                        ocrIngredient=ingredient,
                        ingredientName=row.standard_name,
                        description=row.description,
                        similarityScore=round(float(row.similarity), 4),
                    )
                )
        except Exception as error:
            print(f"'{ingredient}' 임베딩 검색 중 오류: {error}")

    return RagSearchResponse(
        analysisId=request.analysisId,
        extractedIngredients=ingredients,
        totalCount=len(contexts),
        contexts=contexts,
    )