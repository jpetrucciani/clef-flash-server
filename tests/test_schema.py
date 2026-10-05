import unittest

from pydantic import ValidationError

from clef_flash_server.schema import DecisionRequest


class DecisionRequestTests(unittest.TestCase):
    def test_all_native_question_types(self) -> None:
        request = DecisionRequest.model_validate(
            {
                "model": "clef-flash",
                "state": {"invoice": {"total": 1250, "status": "overdue"}},
                "questions": {
                    "status": {
                        "type": "choice",
                        "criteria": {"paid": "Paid", "overdue": "Past due"},
                    },
                    "large": {
                        "type": "noul",
                        "instructions": "Is the total above 1000?",
                    },
                    "priority": {
                        "type": "score",
                        "criteria": ["low", "medium", "high"],
                    },
                },
            }
        )
        self.assertEqual(request.model_dump()["state"]["invoice"]["total"], 1250)
        self.assertEqual(len(request.questions), 3)

    def test_invalid_schema_is_rejected(self) -> None:
        invalid_questions = [
            {},
            {"q": {"type": "choice", "criteria": {}}},
            {"q": {"type": "choice", "criteria": ["a", "b"]}},
            {"q": {"type": "score", "criteria": {"a": "b"}}},
            {"q": {"type": "noul", "criteria": {"maybe": "Perhaps"}}},
            {"q": {"type": "text"}},
            {"": {"type": "noul"}},
        ]
        for questions in invalid_questions:
            with self.subTest(questions=questions), self.assertRaises(ValidationError):
                DecisionRequest.model_validate(
                    {"model": "clef-flash", "state": "state", "questions": questions}
                )

    def test_null_state_is_preserved(self) -> None:
        request = DecisionRequest.model_validate(
            {"model": "clef-flash", "state": None, "questions": {"q": {"type": "noul"}}}
        )
        self.assertIn("state", request.model_dump())
        self.assertIsNone(request.state)

    def test_wrong_model_and_unsupported_media_are_rejected(self) -> None:
        for extra in [{"model": "qwen"}, {"images": ["https://example.com/a.jpg"]}]:
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                DecisionRequest.model_validate(
                    {
                        "model": "clef-flash",
                        "state": "state",
                        "questions": {"q": {"type": "noul"}},
                        **extra,
                    }
                )


if __name__ == "__main__":
    unittest.main()
