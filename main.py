import os
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import get_db, IngredientKnowledge
from clova_client import get_clova_embedding
from ingredient_extractor import extract_ingredients

load_dotenv()

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
    safetyLevel: Optional[str] = None
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
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


# --- API 엔드포인트 (명세서 규격) ---
@app.post("/api/v1/rag/search", response_model=RagSearchResponse)
def search_rag_context(request: RagSearchRequest, db: Session = Depends(get_db)):
    # 1. OCR 텍스트에서 성분 추출 및 중복 제거
    raw_ingredients = extract_ingredients(request.ocrText)
    ingredients = deduplicate_ingredients(raw_ingredients)

    if not ingredients:
        raise HTTPException(
            status_code=422,
            detail="OCR 텍스트에서 원료명을 찾지 못했습니다.",
        )

    contexts: list[ContextItem] = []

    # 2. 각 추출 성분에 대해 하이브리드 검색 (키워드 매칭 -> 벡터 유사도 검색)
    for ingredient in ingredients:
        matched_item = None
        score = 0.0

        # Step A: Exact / Keyword Match
        keyword_match = (
            db.query(IngredientKnowledge)
            .filter(
                (IngredientKnowledge.raw_name == ingredient)
                | (IngredientKnowledge.canonical_name == ingredient)
            )
            .first()
        )

        if keyword_match:
            matched_item = keyword_match
            score = 1.0
        else:
            # Step B: Vector Similarity Match
            try:
                query_vector = get_clova_embedding(ingredient)
                sql = text(
                    """
                    SELECT id, raw_name, canonical_name, category, description, caution,
                           1 - (embedding <=> :vec::vector) AS similarity
                    FROM ingredient_knowledge
                    ORDER BY embedding <=> :vec::vector
                    LIMIT :top_k
                    """
                )
                results = db.execute(
                    sql, {"vec": str(query_vector), "top_k": request.topK}
                ).fetchall()

                for row in results:
                    contexts.append(
                        ContextItem(
                            ocrIngredient=ingredient,
                            ingredientName=row.canonical_name or row.raw_name,
                            safetyLevel=row.category,
                            description=row.description,
                            similarityScore=round(float(row.similarity), 4),
                        )
                    )
                continue
            except Exception as e:
                print(f"'{ingredient}' 임베딩 검색 중 오류: {e}")

        # 키워드 매칭 결과 처리
        if matched_item:
            contexts.append(
                ContextItem(
                    ocrIngredient=ingredient,
                    ingredientName=matched_item.canonical_name or matched_item.raw_name,
                    safetyLevel=matched_item.category,
                    description=matched_item.description,
                    similarityScore=round(score, 4),
                )
            )

    return RagSearchResponse(
        analysisId=request.analysisId,
        extractedIngredients=ingredients,
        totalCount=len(contexts),
        contexts=contexts,
    )