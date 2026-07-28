import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

CLOVA_EMBEDDING_API_KEY = os.getenv("CLOVA_STUDIO_API_KEY")
CLOVA_APIGW_API_KEY = os.getenv("CLOVA_APIGW_API_KEY")
CLOVA_REQUEST_ID = os.getenv("CLOVA_REQUEST_ID", "default-request-id")
CLOVA_EMBEDDING_URL = os.getenv(
    "CLOVA_EMBEDDING_URL",
    "https://clovastudio.stream.ntruss.com/testapp/v1/api-tools/embedding/v2",
)


def get_clova_embedding(text: str, max_retries: int = 3) -> list[float]:
    """Clova Studio Embedding API를 호출하여 1024차원 벡터를 반환합니다."""
    headers = {
        "Authorization": f"Bearer {CLOVA_EMBEDDING_API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": CLOVA_REQUEST_ID,
        "Content-Type": "application/json",
    }
    if CLOVA_APIGW_API_KEY:
        headers["X-NCP-APIGW-API-KEY"] = CLOVA_APIGW_API_KEY

    payload = {"text": text}

    # 테스트 API Key 제한(분당 60회 = 1초당 1회)을 맞추기 위해 1.1초 간격 유지
    time.sleep(1.1)

    for attempt in range(max_retries):
        response = requests.post(CLOVA_EMBEDDING_URL, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if "result" in data and "embedding" in data["result"]:
                return data["result"]["embedding"]
            elif "embedding" in data:
                return data["embedding"]
            else:
                raise KeyError(f"Unexpected response format from Clova API: {data}")

        elif response.status_code == 429:
            # QPM 초기화(1분)를 기다리기 위해 60초 대기
            print(f"API Rate Limit(429) 발생. 1분(60초) 대기 후 재시도합니다... ({attempt + 1}/{max_retries})")
            time.sleep(60)

        else:
            raise RuntimeError(
                f"Clova Embedding API error ({response.status_code}): {response.text}"
            )

    raise RuntimeError(f"Clova Embedding API 최대 재시도 횟수({max_retries}회)를 초과했습니다.")