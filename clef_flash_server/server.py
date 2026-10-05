"""Token-bounded CUDA batching with no silent state truncation."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from cloudflare_clef_release import EncodedRecord

import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import JsonValue
from starlette.responses import JSONResponse, Response
from transformers import BitsAndBytesConfig

from clef_flash_server.schema import DecisionRequest

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    model_path: Path
    quantization: str = "nf4"
    max_input_tokens: int = 16384
    max_queued_requests: int = 4
    max_body_bytes: int = 8 * 1024 * 1024
    max_batch_size: int = 4
    max_batch_tokens: int = 16384
    batch_wait_ms: float = 5

    def __post_init__(self) -> None:
        if (
            min(
                self.max_input_tokens,
                self.max_queued_requests,
                self.max_body_bytes,
                self.max_batch_size,
                self.max_batch_tokens,
            )
            < 1
        ):
            raise ValueError("token, batch, body and queue limits must be positive")
        if self.max_batch_tokens < self.max_input_tokens:
            raise ValueError("batch token budget must cover one maximum-length input")
        if not 0 <= self.batch_wait_ms <= 1000:
            raise ValueError("batch wait must be between 0 and 1000 milliseconds")
        if self.quantization not in ("nf4", "bf16"):
            raise ValueError("quantization must be nf4 or bf16")


@dataclass(frozen=True)
class PreparedRequest:
    request: DecisionRequest
    encoded: EncodedRecord

    @property
    def input_tokens(self) -> int:
        return len(self.encoded.input_ids)


def select_batch(lengths: list[int], settings: Settings) -> list[int]:
    """Serve the oldest input, then compatible followers within the padding budget."""
    if not lengths:
        return []
    selected = [0]
    shortest = longest = lengths[0]
    for index, length in enumerate(lengths[1:], start=1):
        if len(selected) == settings.max_batch_size:
            break
        low, high = min(shortest, length), max(longest, length)
        if high <= 2 * low and high * (len(selected) + 1) <= settings.max_batch_tokens:
            selected.append(index)
            shortest, longest = low, high
    return selected


class Engine:
    def __init__(self, settings: Settings) -> None:
        import cloudflare_clef_release as joint_schema_model

        if not settings.model_path.is_dir():
            raise ValueError(
                f"provision the model directory first: {settings.model_path}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("Clef requires a working CUDA GPU")
        from transformers.models.qwen3_5 import modeling_qwen3_5

        self.fast_kernels = modeling_qwen3_5.is_fast_path_available
        if not self.fast_kernels:
            raise RuntimeError(
                "Qwen3.5 fast kernels require causal-conv1d and flash-linear-attention"
            )
        quantization = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                # The joint head indexes this full vocabulary matrix directly.
                llm_int8_skip_modules=["lm_head"],
            )
            if settings.quantization == "nf4"
            else None
        )
        self.model, self.processor = joint_schema_model.load_release_model(
            settings.model_path,
            device="cuda:0",
            dtype=torch.bfloat16,
            quantization_config=quantization,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.settings = settings
        self.upstream = joint_schema_model
        context = self.model.language_model.config.text_config.max_position_embeddings
        if settings.max_input_tokens > context:
            raise ValueError(
                f"configured input limit exceeds the backbone's {context} tokens"
            )
        self.completed_batches = 0
        self.completed_requests = 0
        self.largest_batch = 0
        torch.cuda.synchronize()

    def prepare(self, request: DecisionRequest) -> PreparedRequest:
        payload = request.model_dump()
        # Cloudflare's encoder truncates state at max_length. Encode completely,
        # then reject over-limit requests so decisive evidence is never discarded.
        encoded = self.upstream.encode_record(
            self.processor.tokenizer,
            payload,
            max_length=2**31 - 1,
            processor=self.processor,
        )
        if len(encoded.input_ids) > self.settings.max_input_tokens:
            raise HTTPException(
                413,
                f"state and schema require {len(encoded.input_ids)} tokens; "
                f"maximum is {self.settings.max_input_tokens}",
            )
        return PreparedRequest(request, encoded)

    @torch.inference_mode()
    def predict_batch(
        self, requests: list[PreparedRequest]
    ) -> list[dict[str, JsonValue]]:
        if not requests:
            raise ValueError("a batch must contain at least one request")
        padded_tokens = max(item.input_tokens for item in requests) * len(requests)
        if (
            len(requests) > self.settings.max_batch_size
            or padded_tokens > self.settings.max_batch_tokens
        ):
            raise ValueError("batch exceeds configured size or padded token budget")
        batch = self.upstream.collate_records(
            [item.encoded for item in requests],
            self.processor.tokenizer.pad_token_id,
            torch.device("cuda:0"),
        )
        all_logits = self.model(batch)
        responses: list[dict[str, JsonValue]] = []
        for item, logits in zip(requests, all_logits, strict=True):
            payload = item.request.model_dump()
            answers: dict[str, JsonValue] = {
                question.question_id: self.upstream.systemone_answer(
                    payload["questions"][question.question_id],
                    dict(
                        zip(
                            question.option_ids,
                            scores.float().softmax(-1).tolist(),
                            strict=True,
                        )
                    ),
                )
                for question, scores in zip(item.encoded.questions, logits, strict=True)
            }
            responses.append(
                {
                    "model": item.request.model,
                    "answers": answers,
                    "usage": {"input_tokens": item.input_tokens, "output_tokens": 0},
                }
            )
        torch.cuda.synchronize()
        self.completed_batches += 1
        self.completed_requests += len(requests)
        self.largest_batch = max(self.largest_batch, len(requests))
        return responses

    def predict(self, request: DecisionRequest) -> dict[str, JsonValue]:
        return self.predict_batch([self.prepare(request)])[0]


@dataclass
class Job:
    request: DecisionRequest
    result: asyncio.Future[dict[str, JsonValue]]


def create_app(settings: Settings) -> FastAPI:
    queue: asyncio.Queue[Job] = asyncio.Queue()
    engine: Engine | None = None
    waiting = 0

    def fail(job: Job, error: Exception) -> None:
        if not isinstance(error, HTTPException):
            error = HTTPException(500, "Clef inference failed")
        if not job.result.done():
            job.result.set_exception(error)

    async def consume() -> None:
        nonlocal waiting
        pending: list[tuple[Job, PreparedRequest]] = []
        while True:
            if engine is None:
                raise RuntimeError("model is not ready")
            incoming: list[Job] = []
            if not pending:
                incoming.append(await queue.get())
                # A bounded window lets simultaneous clients share a forward pass.
                if settings.max_batch_size > 1 and settings.batch_wait_ms:
                    await asyncio.sleep(settings.batch_wait_ms / 1000)
            while not queue.empty():
                incoming.append(queue.get_nowait())
            for job in incoming:
                try:
                    if job.result.cancelled():
                        waiting -= 1
                        queue.task_done()
                        continue
                    prepared = await asyncio.to_thread(engine.prepare, job.request)
                    pending.append((job, prepared))
                except Exception as error:
                    if not isinstance(error, HTTPException):
                        LOGGER.exception("Clef input preparation failed")
                    fail(job, error)
                    waiting -= 1
                    queue.task_done()
            live: list[tuple[Job, PreparedRequest]] = []
            for job, prepared in pending:
                if job.result.cancelled():
                    waiting -= 1
                    queue.task_done()
                else:
                    live.append((job, prepared))
            pending = live
            if not pending:
                continue
            selected = select_batch(
                [item.input_tokens for _, item in pending], settings
            )
            batch = [pending[index] for index in selected]
            pending = [
                item for index, item in enumerate(pending) if index not in selected
            ]
            waiting -= len(batch)
            try:
                responses = await asyncio.to_thread(
                    engine.predict_batch, [item for _, item in batch]
                )
                for (job, _), response in zip(batch, responses, strict=True):
                    if not job.result.done():
                        job.result.set_result(response)
            except torch.cuda.OutOfMemoryError:
                LOGGER.exception("CUDA ran out of memory during Clef inference")
                for job, _ in batch:
                    fail(
                        job,
                        HTTPException(
                            503, "CUDA memory exhausted; reduce input or batch size"
                        ),
                    )
                torch.cuda.empty_cache()
            except Exception as error:
                LOGGER.exception("Clef batch inference failed")
                for job, _ in batch:
                    fail(job, error)
            finally:
                for _ in batch:
                    queue.task_done()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal engine
        engine = await asyncio.to_thread(Engine, settings)
        worker = asyncio.create_task(consume())
        try:
            yield
        finally:
            # Finish admitted jobs before releasing the model or stopping CUDA work.
            await queue.join()
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="Clef-Flash", lifespan=lifespan)

    @app.middleware("http")
    async def bound_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method == "POST":
            size = 0
            chunks: list[bytes] = []
            async for chunk in request.stream():
                size += len(chunk)
                if size > settings.max_body_bytes:
                    return JSONResponse(
                        {"detail": "request body exceeds the byte limit"},
                        status_code=413,
                    )
                chunks.append(chunk)
            # Starlette caches the bounded body for downstream JSON validation.
            request._body = b"".join(chunks)
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, JsonValue]:
        return {
            "status": "ready" if engine is not None else "loading",
            "model": "clef-flash",
            "quantization": settings.quantization,
            "max_input_tokens": settings.max_input_tokens,
            "queued_requests": waiting,
            "max_queued_requests": settings.max_queued_requests,
            "max_batch_size": settings.max_batch_size,
            "max_batch_tokens": settings.max_batch_tokens,
            "batch_wait_ms": settings.batch_wait_ms,
            "fast_kernels": engine.fast_kernels if engine else False,
            "completed_batches": engine.completed_batches if engine else 0,
            "completed_requests": engine.completed_requests if engine else 0,
            "largest_batch": engine.largest_batch if engine else 0,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(0),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }

    def admit(
        requests: list[DecisionRequest],
    ) -> list[asyncio.Future[dict[str, JsonValue]]]:
        nonlocal waiting
        if waiting + len(requests) > settings.max_queued_requests:
            raise HTTPException(
                429, "Clef request queue is full", headers={"Retry-After": "1"}
            )
        results: list[asyncio.Future[dict[str, JsonValue]]] = []
        for request in requests:
            result: asyncio.Future[dict[str, JsonValue]] = (
                asyncio.get_running_loop().create_future()
            )
            waiting += 1
            queue.put_nowait(Job(request, result))
            results.append(result)
        return results

    @app.post("/v1/systemone")
    async def decide(request: DecisionRequest) -> dict[str, JsonValue]:
        return await admit([request])[0]

    @app.post("/v1/systemone/batch")
    async def decide_batch(
        requests: Annotated[list[DecisionRequest], Body(min_length=1, max_length=128)],
    ) -> list[dict[str, JsonValue]]:
        if len(requests) > settings.max_queued_requests:
            raise HTTPException(
                422, "batch contains more requests than the configured queue limit"
            )
        # Admission is atomic; collect every result even if one input fails.
        results = await asyncio.gather(*admit(requests), return_exceptions=True)
        responses: list[dict[str, JsonValue]] = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            responses.append(result)
        return responses

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-queued-requests", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--batch-wait-ms", type=float, default=5)
    arguments = parser.parse_args()
    try:
        settings = Settings(
            model_path=arguments.model_path,
            quantization=arguments.quantization,
            max_input_tokens=arguments.max_input_tokens,
            max_queued_requests=arguments.max_queued_requests,
            max_batch_size=arguments.max_batch_size,
            max_batch_tokens=arguments.max_batch_tokens,
            batch_wait_ms=arguments.batch_wait_ms,
        )
    except ValueError as error:
        parser.error(str(error))
    uvicorn.run(
        create_app(settings), host=arguments.host, port=arguments.port, workers=1
    )


if __name__ == "__main__":
    main()
