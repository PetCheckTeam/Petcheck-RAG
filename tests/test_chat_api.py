import os
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ["CLOVA_STUDIO_API_KEY"] = "test-chat-key"
os.environ["CLOVA_CHAT_BASE_URL"] = "https://chat.example.com"
os.environ["CLOVA_CHAT_MODEL"] = "HCX-DASH-002"
os.environ["CLOVA_CHAT_TIMEOUT_SECONDS"] = "60"

import requests
from fastapi import HTTPException
from pydantic import ValidationError

from chat_schemas import RagChatRequest
from main import app, chat_rag_context


class FakeResponse:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        return self.payload


def success_response(answer: str = "분석 결과에 따른 답변입니다.") -> FakeResponse:
    return FakeResponse(
        200,
        {
            "result": {
                "message": {"role": "assistant", "content": answer},
                "finishReason": "stop",
                "usage": {
                    "promptTokens": 120,
                    "completionTokens": 30,
                    "totalTokens": 150,
                },
            }
        },
    )


def request_payload(**overrides):
    payload = {
        "analysisId": 1,
        "productName": "테스트 사료",
        "petName": "초코",
        "petType": "DOG",
        "avoidIngredients": ["닭고기", "밀"],
        "ocrText": "계육분, 돼지지방, 밀글루텐, 어유, 연어",
        "ingredientResults": [
            {
                "ocrIngredient": "계육분",
                "ingredientName": "닭고기",
                "matchStatus": "MATCHED",
                "matchedAvoidIngredientName": "닭고기",
                "description": "닭·가금류에서 유래한 육류, 분말 및 부산물 원료",
                "similarityScore": 1.0,
            }
        ],
        "question": "이 사료를 먹여도 괜찮아?",
    }
    payload.update(overrides)
    return payload


class RagChatApiTest(unittest.TestCase):
    def test_normal_response_and_recent_history(self):
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(12)
        ]

        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response(
                "이 제품에는 초코의 회피 성분인 닭고기 관련 원료가 있습니다."
            ),
        ) as post:
            response = chat_rag_context(
                RagChatRequest(**request_payload(history=history))
            )

        self.assertEqual(response.model, "HCX-DASH-002")
        self.assertEqual(response.finishReason, "stop")
        self.assertEqual(response.usage.totalTokens, 150)
        self.assertEqual(response.sources[0].matchStatus, "MATCHED")
        self.assertEqual(response.sources[0].ocrIngredient, "계육분")

        call = post.call_args
        self.assertEqual(
            call.args[0],
            "https://chat.example.com/v3/chat-completions/HCX-DASH-002",
        )
        self.assertEqual(call.kwargs["timeout"], 60.0)
        self.assertEqual(call.kwargs["json"]["topP"], 0.8)
        self.assertEqual(call.kwargs["json"]["maxTokens"], 700)
        messages = call.kwargs["json"]["messages"]
        self.assertEqual(len(messages), 12)
        self.assertEqual(messages[1]["content"], "2")
        self.assertIn("계육분", messages[-1]["content"])
        self.assertIn("MATCHED", messages[-1]["content"])
        self.assertIn("similarityScore", messages[-1]["content"])
        UUID(call.kwargs["headers"]["X-NCP-CLOVASTUDIO-REQUEST-ID"])

    def test_history_defaults_to_empty_list(self):
        first = RagChatRequest(**request_payload())
        second = RagChatRequest(**request_payload(analysisId=2))

        self.assertEqual(first.history, [])
        self.assertEqual(second.history, [])
        self.assertIsNot(first.history, second.history)

        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response(),
        ) as post:
            chat_rag_context(first)

        self.assertEqual(len(post.call_args.kwargs["json"]["messages"]), 2)

    def test_unknown_result_accepts_null_ingredient_name(self):
        request = RagChatRequest(
            **request_payload(
                ingredientResults=[
                    {
                        "ocrIngredient": "알 수 없는 원료",
                        "ingredientName": None,
                        "matchStatus": "UNKNOWN",
                        "matchedAvoidIngredientName": None,
                        "description": None,
                        "similarityScore": 0.3,
                    }
                ]
            )
        )

        self.assertEqual(request.ingredientResults[0].matchStatus, "UNKNOWN")
        self.assertIsNone(request.ingredientResults[0].ingredientName)

    def test_result_accepts_null_similarity_score(self):
        ingredient_result = dict(request_payload()["ingredientResults"][0])
        ingredient_result["similarityScore"] = None

        request = RagChatRequest(
            **request_payload(ingredientResults=[ingredient_result])
        )

        self.assertIsNone(request.ingredientResults[0].similarityScore)

    def test_matched_avoid_ingredient_id_is_not_exposed_in_sources(self):
        ingredient_result = dict(request_payload()["ingredientResults"][0])
        ingredient_result["matchedAvoidIngredientId"] = 42
        request = RagChatRequest(
            **request_payload(ingredientResults=[ingredient_result])
        )

        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response(),
        ):
            response = chat_rag_context(request)

        self.assertEqual(
            request.ingredientResults[0].matchedAvoidIngredientId,
            42,
        )
        source = response.model_dump(mode="json")["sources"][0]
        self.assertNotIn("matchedAvoidIngredientId", source)

    def test_invalid_history_role_is_rejected(self):
        with self.assertRaises(ValidationError):
            RagChatRequest(
                **request_payload(
                    history=[{"role": "system", "content": "허용되지 않음"}]
                )
            )

    def test_blank_question_is_rejected(self):
        for question in ("", "   ", "\t\n"):
            with self.subTest(question=repr(question)):
                with self.assertRaises(ValidationError):
                    RagChatRequest(**request_payload(question=question))

    def test_timeout_is_converted_to_504(self):
        with patch(
            "clova_chat_client.requests.post",
            side_effect=requests.Timeout("sensitive request context"),
        ):
            with self.assertRaises(HTTPException) as raised:
                chat_rag_context(RagChatRequest(**request_payload()))

        self.assertEqual(raised.exception.status_code, 504)
        self.assertNotIn("sensitive", raised.exception.detail)

    def test_connection_error_is_converted_to_502(self):
        with patch(
            "clova_chat_client.requests.post",
            side_effect=requests.ConnectionError("sensitive connection detail"),
        ):
            with self.assertRaises(HTTPException) as raised:
                chat_rag_context(RagChatRequest(**request_payload()))

        self.assertEqual(raised.exception.status_code, 502)
        self.assertNotIn("sensitive", raised.exception.detail)

    def test_429_is_retried_with_exponential_backoff(self):
        responses = [
            FakeResponse(429, {"message": "rate limited"}),
            FakeResponse(429, {"message": "rate limited"}),
            success_response(),
        ]

        with patch(
            "clova_chat_client.requests.post",
            side_effect=responses,
        ) as post, patch("clova_chat_client.time.sleep") as sleep:
            response = chat_rag_context(RagChatRequest(**request_payload()))

        self.assertEqual(response.finishReason, "stop")
        self.assertEqual(post.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [1.0, 2.0],
        )

    def test_500_stops_after_three_retries(self):
        with patch(
            "clova_chat_client.requests.post",
            return_value=FakeResponse(500, {"message": "internal detail"}),
        ) as post, patch("clova_chat_client.time.sleep") as sleep:
            with self.assertRaises(HTTPException) as raised:
                chat_rag_context(RagChatRequest(**request_payload()))

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(post.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [1.0, 2.0, 4.0],
        )

    def test_abnormal_or_empty_response_is_converted_to_502(self):
        invalid_responses = [
            FakeResponse(200, {"unexpected": {}}),
            FakeResponse(
                200,
                {
                    "result": {
                        "message": {"content": "   "},
                        "finishReason": "stop",
                        "usage": {},
                    }
                },
            ),
        ]

        for response in invalid_responses:
            with self.subTest(payload=response.payload), patch(
                "clova_chat_client.requests.post",
                return_value=response,
            ):
                with self.assertRaises(HTTPException) as raised:
                    chat_rag_context(RagChatRequest(**request_payload()))
                self.assertEqual(raised.exception.status_code, 502)

    def test_upstream_4xx_does_not_expose_response(self):
        with patch(
            "clova_chat_client.requests.post",
            return_value=FakeResponse(400, {"apiKey": "secret-value"}),
        ):
            with self.assertRaises(HTTPException) as raised:
                chat_rag_context(RagChatRequest(**request_payload()))

        self.assertEqual(raised.exception.status_code, 502)
        self.assertNotIn("secret-value", raised.exception.detail)

    def test_empty_results_return_no_sources_and_no_guess_instruction(self):
        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response("제공된 정보만으로 확인하기 어렵습니다."),
        ) as post:
            response = chat_rag_context(
                RagChatRequest(**request_payload(ingredientResults=[]))
            )

        self.assertEqual(response.sources, [])
        user_prompt = post.call_args.kwargs["json"]["messages"][-1]["content"]
        self.assertIn("ingredientResults가 비어 있습니다", user_prompt)
        self.assertIn("추측하지 말고", user_prompt)

    def test_question_over_2000_characters_is_rejected(self):
        with self.assertRaises(ValidationError):
            RagChatRequest(**request_payload(question="가" * 2001))

    def test_chat_route_exists_without_changing_search_route(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/v1/rag/chat", paths)
        self.assertIn("/api/v1/rag/search", paths)


if __name__ == "__main__":
    unittest.main()
