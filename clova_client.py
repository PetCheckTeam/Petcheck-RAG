import os
import requests
from dotenv import load_dotenv

load_dotenv()

CLOVA_EMBEDDING_API_KEY = os.getenv("CLOVA_EMBEDDING_API_KEY")
CLOVA_APIGW_API_KEY = os.getenv("CLOVA_APIGW_API_KEY")
CLOVA_REQUEST_ID = os.getenv("CLOVA_REQUEST_ID", "default-request-id")
CLOVA_EMBEDDING_URL = os.getenv(
    "CLOVA_EMBEDDING_URL",
    "https://clovastudio.stream.ntruss.com/testapp/v1/api-tools/embedding/v2",
)


def get_clova_embedding(text: str) -> list[float]:
    """Clova Studio Embedding API를 호출하여 1024차원 벡터를 반환합니다."""
    headers = {
        "Authorization": f"Bearer {CLOVA_EMBEDDING_API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": CLOVA_REQUEST_ID,
        "Content-Type": "application/json",
    }
    if CLOVA_APIGW_API_KEY:
        headers["X-NCP-APIGW-API-KEY"] = CLOVA_APIGW_API_KEY

    payload = {"text": text}

    response = requests.post(CLOVA_EMBEDDING_URL, json=payload, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(
            f"Clova Embedding API error ({response.status_code}): {response.text}"
        )

    data = response.json()

    if "result" in data and "embedding" in data["result"]:
        return data["result"]["embedding"]
    elif "embedding" in data:
        return data["embedding"]
    else:
        raise KeyError(f"Unexpected response format from Clova API: {data}")