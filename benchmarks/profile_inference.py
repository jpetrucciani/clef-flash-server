"""Profile real CUDA inference stages and operators with local release weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.benchmark_http import make_payload
from clef_flash_server.schema import DecisionRequest
from clef_flash_server.server import Engine, Settings


@dataclass(frozen=True)
class Stages:
    encode_ms: float
    collate_ms: float
    backbone_ms: float
    head_ms: float
    readback_ms: float


@dataclass(frozen=True)
class Operator:
    name: str
    calls: int
    cpu_total_ms: float
    cuda_total_ms: float


@dataclass(frozen=True)
class Measurement:
    input_tokens: int
    batch_size: int
    median_stages: Stages
    peak_reserved_mib: float
    operators: list[Operator]


@torch.inference_mode()
def measure_stages(engine: Engine, request: DecisionRequest, batch_size: int) -> Stages:
    torch.cuda.synchronize()
    start = time.perf_counter()
    prepared = [engine.prepare(request) for _ in range(batch_size)]
    encoded = time.perf_counter()
    batch = engine.upstream.collate_records(
        [item.encoded for item in prepared],
        engine.processor.tokenizer.pad_token_id,
        torch.device("cuda:0"),
    )
    torch.cuda.synchronize()
    collated = time.perf_counter()
    backbone = engine.model.language_model.model.language_model
    hidden = backbone(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    torch.cuda.synchronize()
    inferred = time.perf_counter()
    logits = engine.model.head(
        hidden,
        batch["input_ids"],
        batch["attention_mask"],
        batch["records"],
        engine.model.language_model.get_output_embeddings().weight,
    )
    torch.cuda.synchronize()
    scored = time.perf_counter()
    for record in logits:
        for scores in record:
            scores.float().softmax(-1).tolist()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return Stages(
        1000 * (encoded - start),
        1000 * (collated - encoded),
        1000 * (inferred - collated),
        1000 * (scored - inferred),
        1000 * (end - scored),
    )


def benchmark(arguments: argparse.Namespace) -> None:
    settings = Settings(
        arguments.model_path,
        arguments.quantization,
        max_batch_size=arguments.batch_size,
    )
    engine = Engine(settings)
    rows: list[Measurement] = []
    report = {
        "date": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "quantization": arguments.quantization,
        "model_directory": arguments.model_path.name,
        "server_sha256": hashlib.sha256(
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("clef_flash_server/server.py")
            .read_bytes()
        ).hexdigest(),
        "workload": "Synthetic repeated neutral text with mixed choice/score/noul questions. Stage boundaries synchronize CUDA, so these timings isolate costs and do not measure HTTP throughput. Operator totals include child operations and must not be added together.",
        "completed": False,
        "measurements": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in arguments.input_tokens:
        if tokens * arguments.batch_size > settings.max_batch_tokens:
            raise ValueError("profile batch exceeds the server's padded token budget")
        request = DecisionRequest.model_validate_json(
            make_payload(arguments.model_path, tokens, arguments.questions)
        )
        prepared = [engine.prepare(request) for _ in range(arguments.batch_size)]
        for _ in range(arguments.warmup):
            engine.predict_batch(prepared)
        torch.cuda.reset_peak_memory_stats()
        samples = [
            measure_stages(engine, request, arguments.batch_size)
            for _ in range(arguments.iterations)
        ]
        stages = Stages(
            **{
                field: statistics.median(getattr(sample, field) for sample in samples)
                for field in Stages.__dataclass_fields__
            }
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as profiler:
            engine.predict_batch(prepared)
        operators = [
            Operator(
                event.key,
                event.count,
                event.cpu_time_total / 1000,
                event.device_time_total / 1000,
            )
            for event in profiler.key_averages()
            if event.key.startswith(("aten::", "bitsandbytes::", "DaoAILab::"))
        ]
        operators.sort(key=lambda operator: operator.cuda_total_ms, reverse=True)
        row = Measurement(
            tokens,
            arguments.batch_size,
            stages,
            torch.cuda.max_memory_reserved() / 2**20,
            operators[:20],
        )
        rows.append(row)
        print(json.dumps(asdict(row)), flush=True)
        report["measurements"] = [asdict(item) for item in rows]
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    report["completed"] = True
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {arguments.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument(
        "--input-tokens", type=int, nargs="+", default=[512, 4096, 16384]
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if min(arguments.input_tokens) < 1 or max(arguments.input_tokens) > 16384:
        parser.error("input tokens must be between 1 and 16384")
    if min(arguments.batch_size, arguments.warmup, arguments.iterations) < 1:
        parser.error("batch size, warmup and iterations must be positive")
    if max(arguments.input_tokens) * arguments.batch_size > 16384:
        parser.error("profile batch exceeds the 16384 padded-token budget")
    if not 1 <= arguments.questions <= 128:
        parser.error("questions must be between 1 and 128")
    benchmark(arguments)


if __name__ == "__main__":
    main()
