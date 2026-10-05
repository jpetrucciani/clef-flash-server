"""Measure the native GPU batch path independently of the HTTP scheduler."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmarks.benchmark_http import make_payload
from clef_flash_server.schema import DecisionRequest
from clef_flash_server.server import Engine, Settings


@dataclass(frozen=True)
class BatchMeasurement:
    batch_size: int
    iterations: int
    median_ms: float
    requests_per_second: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    max_probability_difference: float


@torch.inference_mode()
def evaluate(
    engine: Engine, request: DecisionRequest, batch_size: int
) -> list[list[list[float]]]:
    records = [
        engine.upstream.encode_record(
            engine.processor.tokenizer,
            request.model_dump(),
            max_length=2**31 - 1,
            processor=engine.processor,
        )
        for _ in range(batch_size)
    ]
    batch = engine.upstream.collate_records(
        records, engine.processor.tokenizer.pad_token_id, torch.device("cuda:0")
    )
    logits = engine.model(batch)
    probabilities = [
        [scores.float().softmax(-1).tolist() for scores in record] for record in logits
    ]
    torch.cuda.synchronize()
    return probabilities


def benchmark(arguments: argparse.Namespace) -> None:
    request = DecisionRequest.model_validate_json(
        make_payload(arguments.model_path, arguments.input_tokens, arguments.questions)
    )
    engine = Engine(Settings(arguments.model_path, arguments.quantization))
    baseline_allocated = torch.cuda.memory_allocated()
    free, _ = torch.cuda.mem_get_info()
    # Keep 15% of the currently available device capacity outside the experiment.
    budget = 0.85 * (free + torch.cuda.memory_reserved())
    rows: list[BatchMeasurement] = []
    skipped: list[int] = []
    report = {
        "date": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model_directory": arguments.model_path.name,
        "quantization": arguments.quantization,
        "fast_kernels": engine.fast_kernels,
        "input_tokens": arguments.input_tokens,
        "questions": arguments.questions,
        "workload": "Identical synthetic records; measures encoding, collation, backbone and joint head plus probability readback. This is an upstream batching experiment, not an HTTP throughput measurement or accuracy evaluation.",
        "memory_budget_mib": budget / 2**20,
        "max_batch_tokens": arguments.max_batch_tokens,
        "measurements": [],
        "skipped_batch_sizes": skipped,
        "completed": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    baseline_probabilities: list[list[float]] | None = None
    baseline_activation = 0.0
    for batch_size in sorted({1, *arguments.batch_sizes}):
        if batch_size * arguments.input_tokens > arguments.max_batch_tokens:
            skipped.append(batch_size)
            print(
                json.dumps(
                    {
                        "batch_size": batch_size,
                        "skipped": "padded token budget exceeded",
                    }
                ),
                flush=True,
            )
            continue
        if baseline_probabilities is not None:
            estimated = baseline_allocated + batch_size * baseline_activation
            if estimated > budget:
                skipped.append(batch_size)
                print(
                    json.dumps(
                        {
                            "batch_size": batch_size,
                            "skipped": "estimated peak exceeds the memory budget",
                            "estimated_mib": estimated / 2**20,
                            "budget_mib": budget / 2**20,
                        }
                    ),
                    flush=True,
                )
                continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        evaluate(engine, request, batch_size)
        timings: list[float] = []
        maximum_difference = 0.0
        for _ in range(arguments.iterations):
            start = time.perf_counter()
            probabilities = evaluate(engine, request, batch_size)
            timings.append(time.perf_counter() - start)
            if baseline_probabilities is None:
                baseline_probabilities = probabilities[0]
            for record in probabilities:
                for scores, reference in zip(
                    record, baseline_probabilities, strict=True
                ):
                    maximum_difference = max(
                        maximum_difference,
                        *(
                            abs(score - expected)
                            for score, expected in zip(scores, reference, strict=True)
                        ),
                    )
        if batch_size == 1:
            baseline_activation = max(
                0, torch.cuda.max_memory_reserved() - baseline_allocated
            )
        median = statistics.median(timings)
        row = BatchMeasurement(
            batch_size,
            len(timings),
            1000 * median,
            batch_size / median,
            torch.cuda.max_memory_allocated() / 2**20,
            torch.cuda.max_memory_reserved() / 2**20,
            maximum_difference,
        )
        rows.append(row)
        print(json.dumps(asdict(row)), flush=True)
        report["measurements"] = [asdict(row) for row in rows]
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    report["completed"] = True
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {arguments.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not 1 <= arguments.input_tokens <= 16384 or not 1 <= arguments.questions <= 128:
        parser.error("input tokens must be 1–16384 and questions must be 1–128")
    if min(arguments.batch_sizes) < 1 or arguments.iterations < 1:
        parser.error("batch sizes and iterations must be positive")
    if arguments.max_batch_tokens < arguments.input_tokens:
        parser.error("batch token budget must admit at least one record")
    benchmark(arguments)


if __name__ == "__main__":
    main()
