import json

from chat_schemas import RagChatRequest


SYSTEM_PROMPT = """당신은 PetCheck 반려동물 사료 성분 분석 챗봇입니다.

규칙:
1. 제공된 분석 결과와 성분 근거만 사용합니다.
2. 서버가 전달한 matchStatus를 변경하거나 재판단하지 않습니다.
3. MATCHED는 등록된 회피 성분과 일치함을 뜻합니다.
4. NOT_MATCHED는 회피 성분과 일치하지 않았다는 뜻이지 절대적으로 안전하다는 뜻이 아닙니다.
5. UNKNOWN은 현재 정보로 비교하기 어렵다는 뜻입니다.
6. similarityScore는 검색 유사도이며 위험도나 안전도가 아닙니다.
7. 제공되지 않은 성분, 효능, 위험성, 진단 내용을 만들어내지 않습니다.
8. 정보가 부족하면 제공된 정보만으로 확인하기 어렵다고 답합니다.
9. 반려동물의 질병을 진단하거나 약 또는 치료 방법을 지시하지 않습니다.
10. 호흡 곤란, 반복 구토, 의식 저하 등 심각한 증상이 언급되면 즉시 동물병원 상담을 안내합니다.
11. 보호자가 이해하기 쉬운 한국어로 답합니다.
12. 답변은 결론, 근거, 주의사항 순서로 작성합니다.
13. 지나치게 긴 답변을 만들지 않습니다.
14. 분석과 무관한 질문에는 PetCheck가 사료 성분 분석을 돕는 챗봇임을 설명하고 관련 질문을 요청합니다."""


def build_user_prompt(request: RagChatRequest) -> str:
    ingredient_results = [
        item.model_dump(mode="json") for item in request.ingredientResults
    ]
    context = {
        "pet": {
            "name": request.petName,
            "type": request.petType,
            "avoidIngredients": request.avoidIngredients,
        },
        "product": {
            "name": request.productName,
            "ocrIngredients": request.ocrText,
        },
        "ingredientResults": ingredient_results,
        "question": request.question,
    }

    evidence_instruction = (
        "ingredientResults가 비어 있습니다. 원료나 일치 여부를 추측하지 말고 "
        "제공된 정보만으로는 확인하기 어렵다고 답하세요."
        if not ingredient_results
        else (
            "ingredientResults의 matchStatus를 서버가 확정한 사실로 유지하고, "
            "각 원료의 표준 성분명·설명·검색 유사도를 근거로 설명하세요."
        )
    )

    return (
        "다음은 Spring Boot 서버가 저장하고 확정한 분석 문맥입니다.\n"
        f"{evidence_instruction}\n\n"
        "[분석 문맥]\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
