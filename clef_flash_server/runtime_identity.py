"""Observe the loaded CUDA model before publishing an authoritative identity."""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import blake3
import torch
from bitsandbytes.nn.modules import Linear4bit, Params4bit

from clef_flash_server import identity, release_files, schema
from clef_flash_server.identity import (
    MODEL_REVISION,
    REFERENCE_SHA256,
    TOKENIZER_SHA256,
    IdentityUnavailable,
    ModelIdentity,
    PackageIdentity,
    Quantization,
    Release,
    packages,
    regular_digest,
    store_root,
)

if TYPE_CHECKING:
    from clef_flash_server.server import Engine


@dataclass(frozen=True)
class Precision:
    quantization: Quantization
    compute: str
    double_quantization: bool
    quantized_layers: int
    head_dtype: str
    vocabulary_dtype: str
    attention: str


@dataclass(frozen=True)
class Library:
    path: str
    sha256: str
    immutable_root: str | None


@dataclass(frozen=True)
class Build:
    python: str
    python_store: str
    packages: tuple[PackageIdentity, ...]
    source_files: dict[str, str]
    settings: dict[str, int | float]
    precision: Precision
    cuda_version: str | None
    torch_config: str
    gpu_name: str
    gpu_capability: tuple[int, int]
    gpu_uuid: str
    libraries: tuple[Library, ...]
    kernel_bindings: dict[str, str]
    environment: dict[str, str]
    arithmetic: dict[str, str | bool]


def loaded_libraries() -> tuple[Library, ...]:
    paths: set[Path] = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/") or ".so" not in fields[5]:
            continue
        name = fields[5]
        if name.endswith(" (deleted)"):
            raise IdentityUnavailable("a loaded runtime library was replaced")
        paths.add(Path(name))
    if not any("libcuda.so" in path.name for path in paths):
        raise IdentityUnavailable("the loaded CUDA driver is not identifiable")
    result = []
    for path in sorted(paths):
        file = regular_digest(path)
        try:
            root = store_root(path)
        except IdentityUnavailable:
            root = None
        result.append(Library(str(path.resolve()), file.sha256, root))
    return tuple(result)


def precision(engine: Engine) -> Precision:
    model = engine.model
    backbone = model.language_model
    quantized = [
        module for module in backbone.modules() if isinstance(module, Linear4bit)
    ]
    actual: Quantization = "nf4" if quantized else "none"
    expected = "nf4" if engine.settings.quantization == "nf4" else "none"
    if actual != expected or bool(
        getattr(backbone, "is_loaded_in_4bit", False)
    ) != bool(quantized):
        raise IdentityUnavailable(
            "loaded quantization differs from the configured backend"
        )
    for module in quantized:
        if not isinstance(module.weight, Params4bit):
            raise IdentityUnavailable("loaded 4-bit layer lacks quantized parameters")
        state = module.weight.quant_state
        if state is None or state.quant_type != "nf4" or state.state2 is None:
            raise IdentityUnavailable("loaded 4-bit layers are not nested NF4")
        if module.compute_dtype != torch.bfloat16:
            raise IdentityUnavailable("loaded quantized compute is not BF16")
    parameters = list(model.parameters())
    if not parameters or any(
        parameter.device.type != "cuda" for parameter in parameters
    ):
        raise IdentityUnavailable("loaded parameters are not entirely on CUDA")
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.bfloat16
        for parameter in parameters
    ):
        raise IdentityUnavailable("loaded floating parameters are not entirely BF16")
    head = next(model.head.parameters(), None)
    vocabulary = backbone.get_output_embeddings().weight
    if (
        head is None
        or head.dtype != torch.bfloat16
        or vocabulary.dtype != torch.bfloat16
    ):
        raise IdentityUnavailable("loaded head/vocabulary precision differs from BF16")
    if model.training or any(module.training for module in model.modules()):
        raise IdentityUnavailable("loaded model has a training-mode module")
    attention = backbone.config.text_config._attn_implementation
    if attention != "sdpa":
        raise IdentityUnavailable("loaded attention implementation differs from SDPA")
    return Precision(
        actual,
        "bf16",
        bool(quantized),
        len(quantized),
        str(head.dtype),
        str(vocabulary.dtype),
        attention,
    )


def encoder_identity(engine: Engine) -> str:
    upstream = Path(engine.upstream.__file__)
    if regular_digest(upstream).sha256 != REFERENCE_SHA256:
        raise IdentityUnavailable(
            "loaded encoder source differs from the pinned reference"
        )
    tokenizer_path = engine.settings.model_path / "tokenizer.json"
    raw = tokenizer_path.read_bytes()
    if regular_digest(tokenizer_path).sha256 != TOKENIZER_SHA256:
        raise IdentityUnavailable(
            "loaded tokenizer content differs from the pinned reference"
        )
    backend = engine.processor.tokenizer.backend_tokenizer
    if json.loads(backend.to_str()) != json.loads(raw):
        raise IdentityUnavailable(
            "loaded tokenizer semantics differ from the pinned JSON"
        )
    content = blake3.blake3(raw).hexdigest()
    return f"clef-v1@{MODEL_REVISION};tokenizer={MODEL_REVISION};blake3={content}"


def capture(engine: Engine, release: Release) -> ModelIdentity:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from clef_flash_server import server

    code = {
        "server": Path(server.__file__),
        "schema": Path(schema.__file__),
        "identity": Path(identity.__file__),
        "runtime_identity": Path(__file__),
        "release_manifest": Path(release_files.__file__),
        "upstream": Path(engine.upstream.__file__),
    }
    for path in code.values():
        store_root(path)
    sources = {name: regular_digest(path).sha256 for name, path in code.items()}
    kernels: dict[str, str] = {}
    for module in engine.model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            for name in ("causal_conv1d_fn", "chunk_gated_delta_rule"):
                function = getattr(module, name, None)
                if function is None:
                    raise IdentityUnavailable("a loaded fast-kernel binding is missing")
                binding = f"{function.__module__}.{function.__qualname__}"
                if name in kernels and kernels[name] != binding:
                    raise IdentityUnavailable(
                        "loaded layers disagree on fast-kernel bindings"
                    )
                kernels[name] = binding
    if not kernels or not engine.fast_kernels:
        raise IdentityUnavailable("loaded fast kernels are unavailable")
    relevant = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("TRITON_", "TORCHINDUCTOR_", "PYTORCH_"))
        or key
        in {
            "CC",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_MODULE_LOADING",
            "CUDA_CACHE_DISABLE",
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "LD_LIBRARY_PATH",
            "LIBRARY_PATH",
        }
    }
    observed = precision(engine)
    gpu = torch.cuda.get_device_properties(0)
    build = Build(
        python=platform.python_version(),
        python_store=store_root(Path(sys.executable)),
        packages=packages(),
        source_files=sources,
        settings={
            "max_input_tokens": engine.settings.max_input_tokens,
            "max_queued_requests": engine.settings.max_queued_requests,
            "max_body_bytes": engine.settings.max_body_bytes,
            "max_batch_size": engine.settings.max_batch_size,
            "max_batch_tokens": engine.settings.max_batch_tokens,
            "batch_wait_ms": float(engine.settings.batch_wait_ms),
        },
        precision=observed,
        cuda_version=torch.version.cuda,
        torch_config=torch.__config__.show(),
        gpu_name=gpu.name,
        gpu_capability=torch.cuda.get_device_capability(0),
        gpu_uuid=str(getattr(gpu, "uuid", "")),
        libraries=loaded_libraries(),
        kernel_bindings=kernels,
        environment=relevant,
        arithmetic={
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
    )
    return ModelIdentity.create(
        release, observed.quantization, encoder_identity(engine), asdict(build)
    )
