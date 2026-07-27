# -*- coding: utf-8 -*-
import os
import http.client
import json
import time
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. DB 및 API 설정 정보
# ==========================================
DB_CONFIG = {
    "dbname": "petcheck_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432"
}

HOST = "clovastudio.stream.ntruss.com"
API_KEY = os.getenv("CLOVA_API_KEY", "Bearer nv-YOUR_ACTUAL_API_KEY_HERE")
REQUEST_ID = os.getenv("CLOVA_REQUEST_ID", "petcheck-seed-001")

# ==========================================
# 2. 샘플 데이터
# ==========================================
SAMPLE_INGREDIENTS = [
    {
        "name": "닭고기",
        "safety": "주의",
        "description": "고단백 영양원이지만 일부 강아지/고양이에게 식이 알레르기 반응(가려움증, 눈물 등)을 유발할 수 있습니다."
    },
    {
        "name": "BHA",
        "safety": "위험",
        "description": "합성 보존제(방부제)로, 장기 복용 시 독성 논란 및 발암 가능성이 제기되는 성분이므로 주의가 필요합니다."
    },
    {
        "name": "소고기",
        "safety": "안전",
        "description": "필수 아미노산과 철분이 풍부한 우수한 단백질원입니다. 단, 우육 알레르기가 있는 경우 제한해야 합니다."
    },
    {
        "name": "연어",
        "safety": "안전",
        "description": "오메가-3 지방산이 풍부하여 피모 건강과 염증 완화에 매우 도움을 주는 단백질원입니다."
    },
    {
        "name": "콩 (대두)",
        "safety": "주의",
        "description": "식물성 단백질원이나, 일부 반려동물에게 소화 불량이나 알레르기를 유발할 수 있습니다."
    }
]

# ==========================================
# 3. Clova Embedding API 호출 함수
# ==========================================
def get_clova_embedding(text: str):
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': API_KEY,
        'X-NCP-CLOVASTUDIO-REQUEST-ID': REQUEST_ID
    }

    payload = {"text": text}

    try:
        conn = http.client.HTTPSConnection(HOST)
        conn.request('POST', '/v1/api-tools/embedding/v2', json.dumps(payload), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()

        if result.get('status', {}).get('code') == '20000':
            return result['result']['embedding']
        else:
            print(f"❌ API 오류 발생: {result}")
            return None
    except Exception as e:
        print(f"❌ 통신 예외 발생: {e}")
        return None

# ==========================================
# 4. DB 적재 실행 로직
# ==========================================
def run_seed():
    print("🚀 PostgreSQL DB 연결 중...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # 1) pgvector 확장 설치 및 테이블 자동 생성
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingredient_knowledge (
                id SERIAL PRIMARY KEY,
                ingredient_name VARCHAR(100) NOT NULL,
                safety_level VARCHAR(20),
                description TEXT NOT NULL,
                embedding VECTOR(1024)
            );
        """)
        conn.commit()
        print("✅ 테이블 및 pgvector 익스텐션 확인 완료!")

    except Exception as e:
        print(f"❌ DB 연결/설정 실패: {e}")
        return

    print("🌱 데이터 임베딩 생성 및 DB 적재 시작...\n")

    for idx, item in enumerate(SAMPLE_INGREDIENTS, 1):
        name = item["name"]
        safety = item["safety"]
        desc = item["description"]

        text_to_embed = f"{name}: {desc}"
        
        print(f"[{idx}/{len(SAMPLE_INGREDIENTS)}] '{name}' 임베딩 변환 중...")
        embedding_vector = get_clova_embedding(text_to_embed)

        if embedding_vector:
            vector_str = str(embedding_vector)
            
            insert_query = """
                INSERT INTO ingredient_knowledge (ingredient_name, safety_level, description, embedding)
                VALUES (%s, %s, %s, %s::vector);
            """
            cur.execute(insert_query, (name, safety, desc, vector_str))
            conn.commit()
            print(f"  └ ✅ DB 저장 완료!")
        else:
            print(f"  └ ⚠️ 저장 실패 (임베딩 생성 오류)")

        time.sleep(0.2)

    cur.close()
    conn.close()
    print("\n🎉 모든 데이터가 성공적으로 DB에 적재되었습니다!")

if __name__ == "__main__":
    run_seed()