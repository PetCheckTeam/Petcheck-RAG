from __future__ import annotations

from fastapi import HTTPException

from chat_schemas import ChatSource, ChatUsage, RagChatRequest, RagChatResponse
from clova_chat_client import (
    ClovaChatClient,
    ClovaChatClientError,
    ClovaChatConnectionError,
    ClovaChatTimeoutError,
)
from prompts import SYSTEM_PROMPT, build_user_prompt


MAX_HISTORY_ITEMS = 10


def build_chat_messages(request: RagChatRequest) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        {"role": item.role, "content": item.content}
        for item in request.history[-MAX_HISTORY_ITEMS:]
    )
    messages.append({"role": "user", "content": build_user_prompt(request)})
    return messages


def generate_chat_response(
    request: RagChatRequest,
    client: ClovaChatClient | None = None,
) -> RagChatResponse:
    try:
        chat_client = client or ClovaChatClient()
        result = chat_client.chat(build_chat_messages(request))
    except ClovaChatTimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="HyperCLOVA X 응답 시간이 초과되었습니다.",
        ) from error
    except ClovaChatConnectionError as error:
        raise HTTPException(
            status_code=502,
            detail="HyperCLOVA X 서비스에 연결할 수 없습니다.",
        ) from error
    except ClovaChatClientError as error:
        raise HTTPException(
            status_code=502,
            detail="HyperCLOVA X 응답을 생성하지 못했습니다.",
        ) from error

    return RagChatResponse(
        answer=result.answer,
        model=result.model,
        finishReason=result.finish_reason,
        usage=ChatUsage(
            promptTokens=result.prompt_tokens,
            completionTokens=result.completion_tokens,
            totalTokens=result.total_tokens,
        ),
        sources=[
            ChatSource(
                ocrIngredient=item.ocrIngredient,
                ingredientName=item.ingredientName,
                matchStatus=item.matchStatus,
                matchedAvoidIngredientName=item.matchedAvoidIngredientName,
                description=item.description,
                similarityScore=item.similarityScore,
            )
            for item in request.ingredientResults
        ],
    )
