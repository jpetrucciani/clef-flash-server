"""Benchmark real HTTP inference with local release weights and one CUDA GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cloudflare_clef_release
import torch
import uvicorn
from transformers import AutoTokenizer

from clef_flash_server import server as server_module
from clef_flash_server.schema import DecisionRequest
from clef_flash_server.server import Settings, create_app


@dataclass(frozen=True)
class Sample:
    latency_seconds: float
    rejections: int
    input_tokens: int


@dataclass(frozen=True)
class LengthMeasurement:
    input_tokens: int
    successful_requests: int
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class Measurement:
    concurrency: int
    successful_requests: int
    queue_rejections: int
    wall_seconds: float
    requests_per_second: float
    p50_ms: float
    p95_ms: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    by_input_tokens: list[LengthMeasurement]
    gpu_batches: int
    largest_batch: int


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def make_payload(model_path: Path, tokens: int, questions: int) -> bytes:
    native_questions = [
        {
            "type": "choice",
            "criteria": {
                "billing": "Invoices and payments",
                "technical": "Bugs and outages",
            },
        },
        {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        {"type": "noul", "instructions": "Is a service down?"},
    ]
    request = DecisionRequest.model_validate(
        {
            "model": "clef-flash",
            "state": "",
            "questions": {
                f"q{index}": native_questions[index % len(native_questions)]
                for index in range(questions)
            },
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    fixed = len(
        cloudflare_clef_release.encode_record(
            tokenizer, request.model_dump(), max_length=2**31 - 1
        ).input_ids
    )
    if tokens < fixed:
        raise ValueError(
            f"schema requires {fixed} tokens, above the requested {tokens}"
        )
    request.state = " neutral" * (tokens - fixed)
    actual = len(
        cloudflare_clef_release.encode_record(
            tokenizer, request.model_dump(), max_length=2**31 - 1
        ).input_ids
    )
    if actual != tokens:
        raise ValueError(f"generated {actual} input tokens instead of {tokens}")
    return request.model_dump_json().encode()


def send_request(base_url: str, payload: bytes, tokens: int, timeout: float) -> Sample:
    start = time.perf_counter()
    deadline = start + timeout
    rejections = 0
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("request exhausted its timeout while retrying the queue")
        request = Request(
            base_url + "/v1/systemone",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=remaining) as response:
                result = json.load(response)
            if result["usage"]["input_tokens"] != tokens:
                raise ValueError("server returned a different input token count")
            return Sample(time.perf_counter() - start, rejections, tokens)
        except HTTPError as error:
            if error.code != 429:
                detail = error.read().decode(errors="replace")
                raise RuntimeError(f"HTTP {error.code}: {detail}") from error
            delay = float(error.headers.get("Retry-After", "1"))
            error.close()
            rejections += 1
            time.sleep(min(delay, max(0, deadline - time.perf_counter())))


def benchmark(arguments: argparse.Namespace) -> None:
    if arguments.torch_threads is not None:
        torch.set_num_threads(arguments.torch_threads)
    lengths = arguments.mixed_input_tokens or [arguments.input_tokens]
    payloads = {
        tokens: make_payload(arguments.model_path, tokens, arguments.questions)
        for tokens in set(lengths)
    }
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    settings = Settings(
        arguments.model_path,
        arguments.quantization,
        max_queued_requests=arguments.max_queued_requests,
        max_batch_size=arguments.max_batch_size,
        max_batch_tokens=arguments.max_batch_tokens,
        batch_wait_ms=arguments.batch_wait_ms,
    )
    server = uvicorn.Server(uvicorn.Config(create_app(settings), log_level="warning"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    rows: list[Measurement] = []
    try:
        deadline = time.monotonic() + 90
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("server failed to start")
            if time.monotonic() > deadline:
                raise TimeoutError("model startup exceeded 90 seconds")
            time.sleep(0.1)
        with urlopen(base_url + "/health", timeout=5) as response:
            health = json.load(response)
        print(
            json.dumps(
                {
                    "event": "ready",
                    "gpu": health["cuda_device"],
                    "quantization": arguments.quantization,
                    "tokens": lengths,
                    "questions": arguments.questions,
                    "torch_threads": torch.get_num_threads(),
                }
            ),
            flush=True,
        )
        for tokens, payload in payloads.items():
            for _ in range(arguments.warmup):
                send_request(base_url, payload, tokens, arguments.request_timeout)
        for concurrency in arguments.concurrency:
            # Warm concurrent shapes too, including any Triton specializations
            # reached only when multiple records share a GPU pass.
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                list(
                    executor.map(
                        lambda index: send_request(
                            base_url,
                            payloads[lengths[index % len(lengths)]],
                            lengths[index % len(lengths)],
                            arguments.request_timeout,
                        ),
                        range(concurrency * arguments.warmup),
                    )
                )
            with urlopen(base_url + "/health", timeout=5) as response:
                before = json.load(response)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                samples = list(
                    executor.map(
                        lambda index: send_request(
                            base_url,
                            payloads[lengths[index % len(lengths)]],
                            lengths[index % len(lengths)],
                            arguments.request_timeout,
                        ),
                        range(arguments.requests),
                    )
                )
            wall = time.perf_counter() - start
            torch.cuda.synchronize()
            latencies = [sample.latency_seconds for sample in samples]
            with urlopen(base_url + "/health", timeout=5) as response:
                after = json.load(response)
            row = Measurement(
                concurrency,
                len(samples),
                sum(sample.rejections for sample in samples),
                wall,
                len(samples) / wall,
                1000 * percentile(latencies, 0.5),
                1000 * percentile(latencies, 0.95),
                torch.cuda.max_memory_allocated() / 2**20,
                torch.cuda.max_memory_reserved() / 2**20,
                [
                    LengthMeasurement(
                        tokens,
                        len(selected),
                        1000 * percentile(selected, 0.5),
                        1000 * percentile(selected, 0.95),
                    )
                    for tokens in sorted(payloads)
                    if (
                        selected := [
                            sample.latency_seconds
                            for sample in samples
                            if sample.input_tokens == tokens
                        ]
                    )
                ],
                after["completed_batches"] - before["completed_batches"],
                after["largest_batch"],
            )
            rows.append(row)
            print(json.dumps(asdict(row)), flush=True)
    finally:
        server.should_exit = True
        thread.join(timeout=30)
        listener.close()
        if thread.is_alive():
            raise TimeoutError("benchmark server did not stop within 30 seconds")
    report = {
        "date": datetime.now(UTC).isoformat(),
        "gpu": health["cuda_device"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "model_directory": arguments.model_path.name,
        "server_sha256": hashlib.sha256(
            Path(server_module.__file__).read_bytes()
        ).hexdigest(),
        "quantization": arguments.quantization,
        "input_tokens": lengths[0] if len(lengths) == 1 else lengths,
        "questions": arguments.questions,
        "warmup_requests": arguments.warmup,
        "concurrent_warmup_requests_per_client": arguments.warmup,
        "max_queued_requests": arguments.max_queued_requests,
        "max_batch_size": arguments.max_batch_size,
        "max_batch_tokens": arguments.max_batch_tokens,
        "batch_wait_ms": arguments.batch_wait_ms,
        "fast_kernels": health["fast_kernels"],
        "workload": "Synthetic repeated neutral text with mixed choice/score/noul questions; closed-loop HTTP clients; 429 retries honor Retry-After and count toward latency.",
        "client_location": "HTTP clients and the real Uvicorn server run in separate threads in one process on localhost.",
        "measurements": [asdict(row) for row in rows],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {arguments.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    shape = parser.add_mutually_exclusive_group()
    shape.add_argument("--input-tokens", type=int, default=512)
    shape.add_argument("--mixed-input-tokens", type=int, nargs="+")
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-queued-requests", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--batch-wait-ms", type=float, default=5)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    lengths = arguments.mixed_input_tokens or [arguments.input_tokens]
    if (
        any(not 1 <= length <= 16384 for length in lengths)
        or not 1 <= arguments.questions <= 128
    ):
        parser.error("input tokens must be 1–16384 and questions must be 1–128")
    if min(arguments.concurrency) < 1 or arguments.requests < max(
        arguments.concurrency
    ):
        parser.error(
            "concurrency must be positive and requests must cover every client"
        )
    if (
        arguments.warmup < 1
        or arguments.max_queued_requests < 1
        or arguments.request_timeout <= 0
    ):
        parser.error("warmup, queue size, and timeout must be positive")
    if arguments.torch_threads is not None and arguments.torch_threads < 1:
        parser.error("torch threads must be positive")
    logging.getLogger("transformers").setLevel(logging.WARNING)
    benchmark(arguments)


if __name__ == "__main__":
    main()
