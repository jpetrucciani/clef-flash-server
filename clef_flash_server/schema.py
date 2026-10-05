"""Validate the native SystemOne request before GPU admission."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: JsonValue = None


class NoulQuestion(Question):
    type: Literal["noul"]
    criteria: dict[Literal["true", "false"], JsonValue] | None = None


class ChoiceQuestion(Question):
    type: Literal["choice"]
    criteria: dict[str, JsonValue] = Field(min_length=1, max_length=512)


class ScoreQuestion(Question):
    type: Literal["score"]
    criteria: list[JsonValue] = Field(min_length=1, max_length=512)


DecisionQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["clef-flash", "Cloudflare/clef-flash"]
    state: JsonValue
    questions: dict[str, DecisionQuestion] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def bound_schema(self) -> "DecisionRequest":
        if any(not question_id for question_id in self.questions):
            raise ValueError("question IDs must not be empty")
        options = sum(
            2 if isinstance(question, NoulQuestion) else len(question.criteria)
            for question in self.questions.values()
        )
        if options > 2048:
            raise ValueError("at most 2048 total options are supported")
        return self
