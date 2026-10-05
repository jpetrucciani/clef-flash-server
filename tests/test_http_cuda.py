"""Exercise the real HTTP service, scheduler and CUDA model, without mocks."""

import asyncio
import gc
import json
import os
import socket
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cloudflare_clef_release
import torch
import uvicorn
from pydantic import JsonValue
from transformers import AutoTokenizer

from benchmarks.benchmark_http import make_payload
from clef_flash_server.schema import DecisionRequest
from clef_flash_server.server import Settings, create_app


@unittest.skipUnless(
    os.environ.get("CLEF_TEST_MODEL_PATH"),
    "set CLEF_TEST_MODEL_PATH for real HTTP/CUDA inference",
)
class HttpCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.listener = socket.socket()
        cls.listener.bind(("127.0.0.1", 0))
        cls.base_url = f"http://127.0.0.1:{cls.listener.getsockname()[1]}"
        cls.settings = Settings(
            Path(os.environ["CLEF_TEST_MODEL_PATH"]), batch_wait_ms=50
        )
        cls.app = create_app(cls.settings)
        cls.server = uvicorn.Server(uvicorn.Config(cls.app, log_level="warning"))

        async def serve() -> None:
            cls.loop = asyncio.get_running_loop()
            await cls.server.serve(sockets=[cls.listener])

        cls.thread = threading.Thread(target=lambda: asyncio.run(serve()), daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 90
        while not cls.server.started:
            if not cls.thread.is_alive() or time.monotonic() >= deadline:
                cls.server.should_exit = True
                raise RuntimeError("HTTP test server failed to start")
            time.sleep(0.05)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.settings.model_path, local_files_only=True
        )
        cls.payload = make_payload(cls.settings.model_path, 512, 3)
        cls.medium_payload = make_payload(cls.settings.model_path, 768, 3)
        cls.long_payload = make_payload(cls.settings.model_path, 16384, 3)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.should_exit = True
        cls.thread.join(timeout=30)
        cls.listener.close()
        if cls.thread.is_alive():
            raise RuntimeError("HTTP server did not drain and stop")
        del cls.server
        del cls.app
        gc.collect()
        torch.cuda.empty_cache()

    def post(
        self, payload: bytes, endpoint: str = "/v1/systemone"
    ) -> tuple[int, object]:
        request = Request(
            self.base_url + endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=60) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            with error:
                return error.code, json.load(error)

    def health(self) -> dict[str, JsonValue]:
        with urlopen(self.base_url + "/health", timeout=5) as response:
            return json.load(response)

    def test_explicit_batch_preserves_order_alias_and_question_ids(self) -> None:
        records = []
        for index, payload in enumerate([self.payload, self.medium_payload] * 2):
            record = json.loads(payload)
            record["model"] = "Cloudflare/clef-flash" if index % 2 else "clef-flash"
            record["questions"] = {
                f"request{index}-{key}": value
                for key, value in record["questions"].items()
            }
            records.append(record)
        before = self.health()["completed_batches"]
        status, responses = self.post(
            json.dumps(records).encode(), "/v1/systemone/batch"
        )
        self.assertEqual(status, 200)
        for record, response in zip(records, responses, strict=True):
            self.assertEqual(response["model"], record["model"])
            self.assertEqual(set(response["answers"]), set(record["questions"]))
            encoded = cloudflare_clef_release.encode_record(
                self.tokenizer, record, max_length=2**31 - 1
            )
            self.assertEqual(response["usage"]["input_tokens"], len(encoded.input_ids))
        after = self.health()
        self.assertTrue(after["fast_kernels"])
        self.assertEqual(after["completed_batches"] - before, 1)
        self.assertEqual(after["largest_batch"], 4)
        self.assertEqual(after["queued_requests"], 0)

    def test_separate_http_requests_batch_and_reject_saturation(self) -> None:
        gate = threading.Barrier(12)

        def send(index: int) -> tuple[int, object]:
            gate.wait(timeout=5)
            return self.post(self.payload)

        before = self.health()
        with ThreadPoolExecutor(max_workers=12) as executor:
            responses = list(executor.map(send, range(12)))
        successes = sum(status == 200 for status, _ in responses)
        rejected = sum(status == 429 for status, _ in responses)
        self.assertGreaterEqual(successes, 4)
        self.assertGreater(rejected, 0)
        self.assertEqual(successes + rejected, 12)
        after = self.health()
        self.assertEqual(
            after["completed_requests"] - before["completed_requests"], successes
        )
        self.assertLess(
            after["completed_batches"] - before["completed_batches"], successes
        )
        self.assertEqual(after["queued_requests"], 0)

    def test_full_length_runs_alone_and_overlimit_does_not_poison_batch(self) -> None:
        before = self.health()["completed_batches"]
        status, responses = self.post(
            b"[" + self.payload + b"," + self.long_payload + b"]", "/v1/systemone/batch"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [item["usage"]["input_tokens"] for item in responses], [512, 16384]
        )
        self.assertEqual(self.health()["completed_batches"] - before, 2)
        overlimit = json.loads(self.long_payload)
        overlimit["state"] += " neutral"
        status, detail = self.post(
            json.dumps([overlimit, json.loads(self.payload)]).encode(),
            "/v1/systemone/batch",
        )
        self.assertEqual(status, 413)
        self.assertIn("16385", detail["detail"])
        self.assertEqual(self.health()["queued_requests"], 0)
        self.assertEqual(self.post(self.payload)[0], 200)

    def test_cancelled_waiter_releases_admission_slot(self) -> None:
        endpoint = next(
            route.endpoint
            for route in self.app.routes
            if getattr(route, "path", None) == "/v1/systemone"
        )
        request = DecisionRequest.model_validate_json(self.payload)
        before = self.health()["completed_requests"]

        async def cancel_waiter() -> None:
            task = asyncio.create_task(endpoint(request))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            # Let the real worker discard the cancelled input during its window.
            await asyncio.sleep(0.1)
            response = await endpoint(request)
            self.assertEqual(response["usage"]["input_tokens"], 512)

        asyncio.run_coroutine_threadsafe(cancel_waiter(), self.loop).result(timeout=60)
        after = self.health()
        self.assertEqual(after["queued_requests"], 0)
        self.assertEqual(after["completed_requests"] - before, 1)

    def test_body_and_batch_validation(self) -> None:
        self.assertEqual(self.post(b" " * (self.settings.max_body_bytes + 1))[0], 413)
        self.assertEqual(self.post(b"[]", "/v1/systemone/batch")[0], 422)
        self.assertEqual(self.post(b"[{}]", "/v1/systemone/batch")[0], 422)
        records = [json.loads(self.payload)] * 5
        self.assertEqual(
            self.post(json.dumps(records).encode(), "/v1/systemone/batch")[0], 422
        )

    def test_z_shutdown_finishes_admitted_request(self) -> None:
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(self.post, self.long_payload)
            deadline = time.monotonic() + 5
            while self.health()["queued_requests"] == 0:
                if response.done() or time.monotonic() >= deadline:
                    self.fail("request was not observed in the collection window")
                time.sleep(0.005)
            self.server.should_exit = True
            self.assertEqual(response.result(timeout=60)[0], 200)
        self.thread.join(timeout=30)
        self.assertFalse(self.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
