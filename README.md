# Clef-Flash server

Serve [Cloudflare Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) through its
trained joint schema head on one CUDA GPU. The native `POST /v1/systemone` API accepts
text or JSON state and `choice`, `score`, and `noul` questions.

The default backbone uses bitsandbytes NF4 with double quantization and BF16 computation.
The vocabulary output matrix and joint head remain BF16 because the decision head reads
that matrix directly. The server batches up to four requests in one GPU pass, with
at most four additional requests waiting.
Image and video HTTP inputs are not implemented.

## Standalone development

Install Nix and direnv, then enter the environment:

```sh
direnv allow
```

The environment imports an exact, hashed `jpetrucciani/nix` revision and builds the locked
Python environment, using 3.13 by default. It includes the server, Python, uv, Ruff, Pyright, jfmt, and the
CUDA JIT compiler. WSL is detected automatically and uses `/usr/lib/wsl/lib`; native NixOS
uses `/run/opengl-driver/lib`. Both direnv and the packaged executables configure the
CUDA loader and Triton compiler paths.

An optional, ignored `.env` can select a GPU or model directory:

```sh
CUDA_VISIBLE_DEVICES=0
CLEF_MODEL_PATH=/path/to/clef-flash
CLEF_PYTHON_VERSION=3.13
```

Download the pinned weights outside the Nix store, then start the server:

```sh
clef-flash-download --local-dir "$CLEF_MODEL_PATH"
clef-flash-server --model-path "$CLEF_MODEL_PATH"
```

Without a custom path, weights go under
`${XDG_CACHE_HOME:-$HOME/.cache}/clef-flash/models/<model-revision>`. The loader and
download command pin revision `17f0b0ad64efb65d273590632833508766b2aae6`. Inference reads
local weights and does not download missing files.

Direnv adds this checkout to `PYTHONPATH`, so Python changes take effect when you restart
the server. Dependency or Nix changes rebuild the environment. Without direnv, use
`nix-shell` and `python -m clef_flash_server.server --model-path "$CLEF_MODEL_PATH"`.

Python 3.12, 3.13, and 3.14 are selectable. Set `CLEF_PYTHON_VERSION` in `.env` for direnv,
or use `nix-shell --argstr pythonVersion 3.14`. Python 3.14.1 is excluded because
[TorchVision 0.28.0](https://pypi.org/project/torchvision/0.28.0/) excludes that patch version.
These are standard CPython interpreters.

## GPU compatibility

NF4 remains the default across GPU generations. Its four-bit weight storage uses
BF16 computation and does not require native FP4 hardware. The current CUDA/Triton
stack targets NVIDIA Ampere or newer, with
[compute capability 8.0 or above](https://github.com/triton-lang/triton#compatibility).
The locked environment uses CUDA 13 and requires a
[compatible NVIDIA driver](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

| GPU                            | Compute capability | NF4 validation                              |
| ------------------------------ | ------------------ | ------------------------------------------- |
| RTX 3090, NVIDIA A2, RTX A6000 | 8.6                | Architecture target; runtime tests pending  |
| RTX 4090                       | 8.9                | Architecture target; runtime tests pending  |
| RTX 5090                       | 12.0               | Local CUDA, HTTP and benchmark tests passed |

Compute capabilities are from [NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus).
Only the RTX 5090 has been tested here. Its mixed 512–16,384-token NF4 run reserved
11.47 GiB, which does not establish memory usage on another GPU. For the
[16 GB A2](https://www.nvidia.com/content/dam/en-zz/solutions/data-center/a2/pdf/a2-datasheet.pdf),
start with `--max-batch-size 1` and measure the intended input and question mix
before increasing batch size.

An accelerated NVFP4 backend is not implemented. That candidate would require
[Blackwell hardware](https://docs.pytorch.org/ao/stable/workflows/inference.html)
and remain optional; it would not replace NF4 on Ampere or Ada GPUs.

## API

The server listens on `127.0.0.1:8015` by default. `GET /health` reports readiness,
quantization, GPU identity, queue length, fast-kernel availability, batch counts, and
PyTorch memory counters.

GET /v1/metadata?model=clef-flash provides fresh, non-cacheable identity discovery.
It accepts the same two model aliases as inference. Verified Nix packages report
the full loaded model identity in metadata and every inference response's model
field. The identity binds the pinned release files, actual tokenizer and encoder,
loaded BF16/NF4 precision, immutable dependency roots, server code, batching
settings, CUDA driver/library content, GPU and kernel bindings.

At startup, the server checks all 13 runtime files against the pinned Hub manifest
before and after loading. It also compares the loaded tokenizer with tokenizer.json
and computes its BLAKE3 digest. This adds startup file reads. Edited releases,
mutable development sources or an unverifiable runtime retain alias responses,
return 503 from metadata and expose the reason in health's identity_error field.
Clients must disable caching in that case. This metadata implementation has CPU
and package validation; live CUDA identity qualification is still pending.

```sh
curl http://127.0.0.1:8015/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "clef-flash",
    "state": "Checkout is returning errors and customers cannot place orders.",
    "questions": {
      "department": {
        "type": "choice",
        "criteria": {"billing": "Invoices and payments", "technical": "Bugs and outages"}
      },
      "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
      "outage": {"type": "noul", "instructions": "Is a service down?"}
    }
  }'
```

To submit several decisions together, put ordinary request objects in a JSON file:

```sh
curl http://127.0.0.1:8015/v1/systemone/batch \
  -H 'Content-Type: application/json' \
  --data-binary @decisions.json
```

`decisions.json` contains an array such as
`[{"model":"clef-flash","state":"Checkout is down","questions":{"outage":{"type":"noul"}}},
{"model":"clef-flash","state":"Checkout has recovered","questions":{"outage":{"type":"noul"}}}]`.
The response is an array in the same order. See the limits below before raising batch size.

Input includes both state and schema and is limited to 16,384 tokens. Over-limit requests
return HTTP 413 without truncation; a full queue returns HTTP 429; CUDA memory exhaustion
returns HTTP 503. A short-input BF16 comparison can use `--quantization bf16` after stopping
the NF4 process. Raising the context limit needs separate memory and latency testing.

This is a decision API. To use LiteLLM, configure an authenticated pass-through endpoint
to `/v1/systemone` rather than a chat-completions model alias.

## Checks

From this checkout:

```sh
test-clef-flash
ruff check clef_flash_server tests
ruff format --check clef_flash_server tests
uv lock --check
jfmt --ci
```

Schema tests run without a GPU. To also run the real CUDA tests, stop any existing Clef
process and set the model directory:

```sh
CLEF_TEST_MODEL_PATH="$CLEF_MODEL_PATH" test-clef-flash
```

To run only the scheduler and CUDA/HTTP tests changed for batching:

```sh
test-clef-flash -- -p test_batching.py
CLEF_TEST_MODEL_PATH="$CLEF_MODEL_PATH" test-clef-flash -- -p 'test_*cuda.py'
```

The CPU tests cover schemas and batch selection. CUDA tests verify actual NF4 layers,
BF16 head/output embeddings, active fast kernels, convolution agreement with PyTorch,
and batched versus individual probabilities on varied-length inputs (absolute tolerance
0.02). Real localhost HTTP tests cover automatic and explicit batches, ordered responses,
queue saturation, cancelled waiters, body/schema limits, full 16,384-token inputs and
rejection at 16,385 tokens. They do not establish NF4 accuracy against a BF16 evaluation
dataset.

## Performance and concurrency

One process owns one model on one GPU and runs one GPU batch at a time. Separate
`POST /v1/systemone` requests are batched automatically. Defaults are four records
per batch, 16,384 padded tokens, a 5 ms collection window, and four waiting
requests. Padded tokens are `longest_input × batch_size`, including padding.
The scheduler serves the oldest request and selects followers whose lengths stay
within a factor of two. Full 16,384-token inputs therefore run alone by default.
The waiting limit includes inputs being encoded and inputs held for another batch;
it excludes the active GPU batch. A full queue returns HTTP 429 and `Retry-After: 1`.

Use `--max-batch-size`, `--max-batch-tokens`, `--batch-wait-ms` and
`--max-queued-requests` to tune these limits. The token budget must cover the
configured maximum input length. `--max-batch-size 1 --batch-wait-ms 0` disables
batching. Additional Uvicorn workers would load additional model copies on the
same GPU; scale across GPUs with one process per GPU and a load balancer.

For an explicit batch, send a JSON array of ordinary decision requests to
`POST /v1/systemone/batch`. The result is an array in input order, preserving each
request's model alias, question IDs, and token usage. A call cannot contain more
records than `--max-queued-requests` (four by default). Admission is atomic: all
records enter the queue together, or the call returns 429. They may run in separate
GPU batches when lengths or the token budget require it. Empty or invalid arrays
return 422. The 8 MiB body limit applies to the whole array. Over-limit inputs
return 413 without truncation; other valid records in an admitted call may still
execute before its error response is returned.

The package includes pinned
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) and
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) kernels.
Nix builds the CUDA convolution extension against the locked PyTorch version.
The server checks fast-kernel availability at startup. First inference compiles
Triton kernels; the benchmarks warm each measured concurrency before timing.
`/health` reports
`fast_kernels`, batch limits, completed batch/request counts, and the largest
completed batch, so actual batching can be checked during a benchmark.

On the RTX 5090, warmed NF4 HTTP runs with three questions and four clients gave:

| Workload                | Original serial server | Fast kernels, serial | Fast kernels and batching |
| ----------------------- | ---------------------- | -------------------- | ------------------------- |
| 512 tokens              | 4.99 requests/sec      | 12.21 requests/sec   | 17.98 requests/sec        |
| Mixed 512–16,384 tokens | 0.95 requests/sec      | 1.92 requests/sec    | 1.97 requests/sec         |

The mixed sequence was 50% 512 tokens, 25% 4,096 tokens, and 25% 16,384 tokens.
Batching adds substantial short-input throughput; long inputs dominate the mixed
workload and run alone under the default budget. For 24 short requests, the server
completed six GPU batches instead of 24. Short-request median latency at four
clients fell from 800 ms to 221 ms. The mixed run's peak PyTorch reserved memory
fell from 14.09 GiB to 11.47 GiB.

Start with four HTTP clients. Eight clients barely changed mixed throughput but
triggered 30 queue rejections and raised its 95th-percentile latency from 2.42 to
7.14 seconds. These are warmed synthetic measurements, with 24 requests per
concurrency setting, rather than production capacity guarantees. The corresponding
`rtx5090-nf4-fast-*.json` reports retain batch counts and configuration.

Before these changes, an RTX 5090 produced these warmed NF4 HTTP measurements
with three questions:

| Input tokens | Requests/second, one client | Median latency, one client | Median latency, four clients |
| ------------ | --------------------------- | -------------------------- | ---------------------------- |
| 512          | 4.96                        | 197 ms                     | 800 ms                       |
| 4,096        | 1.32                        | 758 ms                     | 3,020 ms                     |
| 16,384       | 0.33                        | 3,022 ms                   | 11,969 ms                    |

In those baseline runs, throughput stayed approximately constant at client concurrency 1, 2, 4, and 8.
Eight clients triggered queue rejections. BF16 improved short-request throughput
by about 7% while increasing peak reserved memory from 7.84 GiB to 17.91 GiB.
NF4 reached 13.90 GiB peak reserved memory at 16,384 tokens.

The baseline mixed workload (50% short, 25% medium, 25% long) sustained 0.94–0.96
requests/second. With four clients, even the 512-token requests had a 4.18-second
median latency because they shared the queue with longer inputs.

The earlier upstream batching experiment reached 11.21 decisions/second for four
512-token records, compared with 5.31 for one, using the PyTorch fallback.
Two 16,384-token records completed with 18.84 GiB peak reserved memory; a batch of
four approached the device memory limit and exceeded the 280-second experiment
budget. This is why the server's default padded-token budget keeps long inputs
separate. The standalone batch benchmark limits each batch to 32,768 padded tokens
and estimates capacity from reserved-memory growth.

Repeat a mixed-input HTTP benchmark from this checkout, with the weights already
available and capacity for one additional model on the selected GPU:

```sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  timeout 280 clef-flash-python -m benchmarks.benchmark_http \
  --model-path "$CLEF_MODEL_PATH" \
  --mixed-input-tokens 512 4096 512 16384 \
  --requests 24 --concurrency 1 2 4 8 \
  --output benchmarks/measurements/local-http.json
```

The benchmark starts and stops a real localhost server. The mixed sequence above
uses 50% short, 25% medium, and 25% long inputs. Use `--input-tokens 4096` for one
length or `--quantization bf16` for a precision comparison. Pass
`--max-batch-size 1 --batch-wait-ms 0` to measure fast kernels without batching.
Reports include completed GPU batch counts and all batching settings.

To locate the next bottleneck, profile the real inference stages and CUDA operators:

```sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  timeout 280 clef-flash-python -m benchmarks.profile_inference \
  --model-path "$CLEF_MODEL_PATH" \
  --input-tokens 512 4096 16384 \
  --output benchmarks/measurements/local-profile.json
```

The 5090 profile puts a 16,384-token request at about 1,469 ms in the backbone,
16 ms encoding, 2 ms collation, 8 ms in the joint head and less than 1 ms reading
probabilities back. Matrix multiplications dominate the remaining cost. The profiler
synchronizes at stage boundaries and its operator totals overlap, so these numbers
isolate costs rather than establish HTTP throughput.

Additional local experiments did not establish a better mixed-workload default.
BF16 reached 2.03 requests/sec with four clients but reserved 21.78 GiB, compared
with NF4's 1.97 requests/sec and 11.47 GiB. cuBLASLt, MLP projection fusion and
one compiled gate projection produced no useful gain. Backbone CUDA graph replay
reduced a 512-token single-request prototype from 76 to 64 ms, but increased a
16,384-token request from 1,504 to 1,910 ms. Those prototype paths are not enabled
in the server; their measurements are retained in
[rtx5090-tuning-experiments.json](benchmarks/measurements/rtx5090-tuning-experiments.json).

For short-input throughput, the existing flags can admit eight-record batches:

```sh
clef-flash-server --model-path "$CLEF_MODEL_PATH" \
  --max-batch-size 8 --max-queued-requests 8
```

A 64-request, 512-token run reached 19.83 requests/sec with eight clients, versus
18.45 with four clients in that sweep. Median latency rose from 216 to 401 ms,
and peak reserved memory rose from 8.08 to 8.33 GiB. Keep four clients as the
starting point for the mixed workload; use larger batches when throughput matters
more than per-request latency. The 16,384 padded-token budget still applies.

To measure the upstream batch path separately:

```sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  timeout 280 clef-flash-python -m benchmarks.benchmark_batch \
  --model-path "$CLEF_MODEL_PATH" --input-tokens 512 \
  --batch-sizes 1 2 4 8 --iterations 8 \
  --output benchmarks/measurements/local-batches.json
```

Raw measurements are in [benchmarks/measurements](benchmarks/measurements). They use
synthetic repeated text, warmed inference, and a small number of samples. The HTTP
clients share the server's process and run in separate threads. HTTP latency includes
queue waiting and retries that honor `Retry-After`; batch timings exclude HTTP. These
are local performance measurements, not accuracy or production latency guarantees.

### Titan RTX 3090, October 5, 2026

Titan's installed `0.1.0` package was measured on GPU 1 with the same model revision
and three typed questions. The ordinary service was paused during isolated trials;
TTS and Whisper remained on GPU 0. The mixed sequence was 50% 512 tokens, 25% 4,096
tokens, and 25% 16,384 tokens. These trials completed 2,016 measured HTTP requests,
plus warmups, across batch limits, concurrency levels, token budgets and precisions.

| Workload                 | Clients | Batch/queue limits | Requests/sec | p95 latency | Peak reserved VRAM |
| ------------------------ | ------- | ------------------ | ------------ | ----------- | ------------------ |
| 512 tokens               | 4       | 4/4                | 7.36         | 0.55 s      | 7.89 GiB           |
| 512 tokens               | 8       | 8/8                | 7.66         | 1.05 s      | 8.19 GiB           |
| 512 tokens               | 16      | 16/16              | 8.00         | 2.01 s      | 8.75 GiB           |
| 16,384 tokens            | 1       | 8/8                | 0.249        | 4.05 s      | 9.90 GiB           |
| Mixed, 64-request repeat | 4       | 4/4                | 0.745        | 6.41 s      | 9.90 GiB           |
| Mixed, 64-request repeat | 2       | 8/8                | 0.738        | 4.30 s      | 9.90 GiB           |
| Mixed, 64-request repeat | 4       | 8/8                | 0.729        | 6.55 s      | 9.90 GiB           |

These warmed text workloads approach 4,000 input tokens/sec. For mixed traffic,
start with two concurrent requests: the longer repeat retained almost the same
throughput with lower latency than four clients. Larger batches provide a modest
gain for short-only bulk work, rather than a material mixed-workload improvement.
The 32-client short-input sweeps regressed to 3.70 and 5.02 requests/sec, including
a repeat with four warmup requests per client; their variable batch sizes and large
latency spikes make them unsuitable as a throughput recommendation.

Keep NF4 and the 16,384 padded-token budget for mixed inputs. Doubling that budget
produced 0.748 requests/sec at eight clients while raising peak reserved memory to
12.15 GiB, versus 0.750 requests/sec and 9.91 GiB with the smaller budget. BF16
produced 0.750 requests/sec at four clients while reserving 19.98 GiB. Eight queue
slots eliminated the rejections seen with eight clients and a four-slot queue, but
added concurrency still increased latency substantially.

The stage profile measured about 3,961 ms in the backbone for 16,384 tokens, compared
with 22 ms encoding and 12 ms in the head; GPU matrix operations dominate. Sampled
GPU utilization reached 96–100%, at roughly 345–350 W, without recorded thermal
slowdown. PyTorch reserved-memory peaks in the table exclude driver/context overhead.

The original four-record/four-slot systemd service was restored. A separate localhost
client process then measured 0.753 requests/sec and 6.35 s p95 over 24 mixed requests,
with no rejections and no queued requests remaining. No serving settings were changed
permanently. These synthetic trials do not establish accuracy or all-day capacity.
Raw results, model identity and measurement boundaries are in
[the Titan reports](benchmarks/measurements/titan-20261005).

## Packaging a release

The same `nix/package.nix` builds the standalone executable package and is imported by cfg:

```sh
nix-build default.nix -A package
nix-build default.nix -A package.wsl
nix-build default.nix -A package.python314.wsl
```

The first command builds the native NixOS package; the second uses WSL driver paths.
Executables are `clef-flash-server`, `clef-flash-python`, and `clef-flash-download`.
Cloudflare's inference helper is fetched at build time by exact revision and hash;
its source and the model weights are not vendored in this repository.

Version `0.1.0` is declared in `pyproject.toml`. After committing and publishing the release
as `v0.1.0`, cfg can fetch its archive with `uv-nix.fetchGitHubWorkspace`, pin the hash,
and import `nix/package.nix` directly. cfg keeps the NixOS module and Titan settings;
this repository owns the Python implementation, lock, tests, and package build.

To validate an unpublished checkout, pass its local path to the cfg adapter directly.
The release fetch does not depend on that path:

```nix
pkgs.callPackage /path/to/cfg/pkgs/ai/clef-flash-server.nix {
  serverSource = /path/to/clef-flash-server;
}
```

After publishing, the archive hash can be checked with:

```sh
nix-prefetch-url --unpack \
  https://github.com/jpetrucciani/clef-flash-server/archive/refs/tags/v0.1.0.tar.gz
```

## License

Apache-2.0. Cloudflare's model and inference helper are also released under Apache-2.0.
