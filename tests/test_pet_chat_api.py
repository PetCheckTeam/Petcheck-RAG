import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ["CLOVA_STUDIO_API_KEY"] = "test-pet-chat-key"
os.environ["CLOVA_CHAT_BASE_URL"] = "https://chat.example.com"
os.environ["CLOVA_CHAT_MODEL"] = "HCX-DASH-002"
os.environ["CLOVA_CHAT_TIMEOUT_SECONDS"] = "60"

import requests
from fastapi import HTTPException
from pydantic import ValidationError

from main import app, pet_chat_rag_context
from pet_chat_schemas import PetChatRequest


class FakeResponse:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        return self.payload


def success_response(answer: str = "반려동물 상담 답변입니다.") -> FakeResponse:
    return FakeResponse(
        200,
        {
            "result": {
                "message": {"role": "assistant", "content": answer},
                "finishReason": "stop",
                "usage": {
                    "promptTokens": 80,
                    "completionTokens": 20,
                    "totalTokens": 100,
                },
            }
        },
    )


def request_payload(**overrides):
    payload = {
        "petId": 1,
        "petName": "보리",
        "petType": "DOG",
        "avoidIngredients": ["닭고기", "돼지고기", "밀", "생선"],
        "question": "오늘 저녁에는 어떤 간식을 주는 게 좋을까?",
    }
    payload.update(overrides)
    return payload


class PetChatApiTest(unittest.TestCase):
    def test_consultation_with_avoid_ingredients_succeeds(self):
        answer = (
            "보리의 회피 성분을 확인하면서 간식 원재료명을 살펴보세요."
        )
        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response(answer),
        ) as post:
            response = pet_chat_rag_context(
                PetChatRequest(**request_payload())
            )

        self.assertEqual(response.answer, answer)
        self.assertEqual(response.model, "HCX-DASH-002")
        self.assertEqual(response.finishReason, "stop")
        self.assertEqual(response.usage.totalTokens, 100)
        self.assertEqual(response.sources, [])
        user_prompt = post.call_args.kwargs["json"]["messages"][-1]["content"]
        self.assertIn("닭고기", user_prompt)
        self.assertIn("돼지고기", user_prompt)
        self.assertIn("밀", user_prompt)
        self.assertIn("생선", user_prompt)

    def test_consultation_without_avoid_ingredients_succeeds(self):
        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response(),
        ) as post:
            response = pet_chat_rag_context(
                PetChatRequest(**request_payload(avoidIngredients=[]))
            )

        self.assertEqual(response.sources, [])
        user_prompt = post.call_args.kwargs["json"]["messages"][-1]["content"]
        self.assertIn("등록된 회피 성분이 없습니다", user_prompt)
        self.assertIn("모든 음식의 안전을 보장하지는 않는다", user_prompt)

    def test_general_free_question_succeeds(self):
        question = "오늘 서울 날씨에 어울리는 인사말을 알려줘"
        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response("따뜻한 저녁 보내세요."),
        ) as post:
            response = pet_chat_rag_context(
                PetChatRequest(**request_payload(question=question))
            )

        self.assertEqual(response.answer, "따뜻한 저녁 보내세요.")
        messages = post.call_args.kwargs["json"]["messages"]
        self.assertIn("일반적인 일상 질문", messages[0]["content"])
        self.assertIn(question, messages[-1]["content"])

    def test_pet_food_question_succeeds(self):
        question = "보리에게 줄 간식을 고를 때 무엇을 확인해야 해?"
        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response(),
        ) as post:
            pet_chat_rag_context(
                PetChatRequest(**request_payload(question=question))
            )

        messages = post.call_args.kwargs["json"]["messages"]
        self.assertIn("식품이나 간식", messages[0]["content"])
        self.assertIn("등록된 회피 성분만", messages[-1]["content"])

    def test_prompt_does_not_confirm_product_safety_without_analysis(self):
        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response(),
        ) as post:
            pet_chat_rag_context(
                PetChatRequest(
                    **request_payload(question="이 사료 먹여도 돼?")
                )
            )

        messages = post.call_args.kwargs["json"]["messages"]
        self.assertIn("분석 결과 없이", messages[0]["content"])
        self.assertIn("특정 제품이 안전하다고 확정하지 말고", messages[-1]["content"])
        self.assertIn("성분표 확인 또는 PetCheck 사료 분석", messages[-1]["content"])

    def test_history_defaults_to_empty_list(self):
        first = PetChatRequest(**request_payload())
        second = PetChatRequest(**request_payload(petId=2))

        self.assertEqual(first.history, [])
        self.assertEqual(second.history, [])
        self.assertIsNot(first.history, second.history)

        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response(),
        ) as post:
            pet_chat_rag_context(first)

        self.assertEqual(len(post.call_args.kwargs["json"]["messages"]), 2)

    def test_only_ten_most_recent_history_items_are_sent(self):
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(12)
        ]

        with patch(
            "clova_chat_client.requests.post",
            return_value=success_response(),
        ) as post:
            pet_chat_rag_context(
                PetChatRequest(**request_payload(history=history))
            )

        messages = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(len(messages), 12)
        self.assertEqual(messages[1]["content"], "2")
        self.assertEqual(messages[-2]["content"], "11")

    def test_blank_question_is_rejected(self):
        for question in ("", "   ", "\t\n"):
            with self.subTest(question=repr(question)):
                with self.assertRaises(ValidationError):
                    PetChatRequest(**request_payload(question=question))

    def test_question_length_is_limited_to_2000_characters(self):
        request = PetChatRequest(**request_payload(question="가" * 2000))
        self.assertEqual(len(request.question), 2000)

        with self.assertRaises(ValidationError):
            PetChatRequest(**request_payload(question="가" * 2001))

    def test_invalid_history_role_is_rejected(self):
        with self.assertRaises(ValidationError):
            PetChatRequest(
                **request_payload(
                    history=[{"role": "system", "content": "허용되지 않음"}]
                )
            )

    def test_timeout_is_converted_to_504(self):
        with patch(
            "clova_chat_client.requests.post",
            side_effect=requests.Timeout("sensitive question context"),
        ):
            with self.assertRaises(HTTPException) as raised:
                pet_chat_rag_context(PetChatRequest(**request_payload()))

        self.assertEqual(raised.exception.status_code, 504)
        self.assertNotIn("sensitive", raised.exception.detail)

    def test_connection_error_is_converted_to_502(self):
        with patch(
            "clova_chat_client.requests.post",
            side_effect=requests.ConnectionError("sensitive connection detail"),
        ):
            with self.assertRaises(HTTPException) as raised:
                pet_chat_rag_context(PetChatRequest(**request_payload()))

        self.assertEqual(raised.exception.status_code, 502)
        self.assertNotIn("sensitive", raised.exception.detail)

    def test_invalid_upstream_response_is_converted_to_502(self):
        with patch(
            "clova_chat_client.requests.post",
            return_value=FakeResponse(200, {"unexpected": {}}),
        ):
            with self.assertRaises(HTTPException) as raised:
                pet_chat_rag_context(PetChatRequest(**request_payload()))

        self.assertEqual(raised.exception.status_code, 502)

    def test_pet_chat_route_is_added_without_removing_existing_routes(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/v1/rag/pet-chat", paths)
        self.assertIn("/api/v1/rag/chat", paths)
        self.assertIn("/api/v1/rag/search", paths)


if __name__ == "__main__":
    unittest.main()
