import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

load_dotenv()

app = FastAPI(
    title="PetCheck RAG Engine",
    description="Clova Embedding v2 기반 성분 지식 RAG 서버",
    version="1.0.0"
)

# ==========================================
# 1. 환경 변수 및 설정 값
# ==========================================
# Docker PostgreSQL (계정: postgres / 비번: postgres)
DB_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/petcheck_db"
)

# Clova Studio API 설정
CLOVA_EMBEDDING_URL = "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2"
# 발급받으신 API Key를 넣으세요 ('Bearer ' 붙여서)
API_KEY = os.getenv("CLOVA_API_KEY", "Bearer nv-YOUR_ACTUAL_API_KEY_HERE")
REQUEST_ID = os.getenv("CLOVA_REQUEST_ID", "126186fac4564c9b8d9d4054f08186aa")

# DB 세션 생성
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# 2. PostgreSQL DB 테이블 매핑 (ingredient_knowledge)
# ==========================================
class IngredientKnowledge(Base):
    __tablename__ = "ingredient_knowledge"

    id = Column(Integer, primary_key=True, index=True)
    ingredient_name = Column(String(100), nullable=False)
    safety_level = Column(String(20))
    description = Column(Text, nullable=False)
    # Clova Embedding v2 (1024차원)
    embedding = Column(Vector(1024))

# ==========================================
# 3. Clova Studio Embedding API 호출 함수
# ==========================================
def get_clova_embedding(text: str) -> List[float]:
    """Clova Studio Embedding v2 API를 호출해 텍스트를 1024차원 벡터로 변환합니다."""
    headers = {
        "Authorization": CLOVA_API_KEY,
        "X-NCP-CLOVASTUDIO-REQUEST-ID": CLOVA_REQUEST_ID,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text
    }

    try:
        response = requests.post(CLOVA_EMBEDDING_URL, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result_data = response.json()
            if result_data.get("status", {}).get("code") == "20000":
                return result_data["result"]["embedding"]
            else:
                raise Exception(f"API Internal Error: {result_data}")
        else:
            raise Exception(f"HTTP Error ({response.status_code}): {response.text}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clova Embedding API Call Failed: {str(e)}")

# ==========================================
# 4. Request / Response DTO
# ==========================================
class RagSearchRequest(BaseModel):
    analysisId: int
    ocrText: str
    topK: Optional[int] = 3
    petType: Optional[str] = "DOG"

class ContextItem(BaseModel):
    ingredientName: str
    safetyLevel: str
    description: str
    similarityScore: float

class RagSearchResponse(BaseModel):
    analysisId: int
    totalCount: int
    contexts: List[ContextItem]

# ==========================================
# 5. RAG 검색 API 엔드포인트
# ==========================================
@app.post("/api/v1/rag/search", response_model=RagSearchResponse)
async def search_rag_context(request: RagSearchRequest):
    db = SessionLocal()
    try:
        # ① OCR 텍스트 1024차원 벡터 변환
        query_vector = get_clova_embedding(request.ocrText)

        # ② pgvector 코사인 거리(Cosine Distance) 기반 유사도 검색
        results = db.query(
            IngredientKnowledge,
            IngredientKnowledge.embedding.cosine_distance(query_vector).label("distance")
        ).order_by("distance").limit(request.topK).all()

        contexts = []
        for item, distance in results:
            # Cosine Distance -> Similarity Score (0~1)
            similarity_score = round(1.0 - float(distance), 4)
            
            contexts.append(
                ContextItem(
                    ingredientName=item.ingredient_name,
                    safetyLevel=item.safety_level,
                    description=item.description,
                    similarityScore=similarity_score
                )
            )

        return RagSearchResponse(
            analysisId=request.analysisId,
            totalCount=len(contexts),
            contexts=contexts
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Search Execution Error: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)