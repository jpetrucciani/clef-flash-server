"""Run against real release weights when CLEF_TEST_MODEL_PATH is provided."""

import gc
import os
import unittest
from pathlib import Path

import torch
from bitsandbytes.nn import Linear4bit
from fastapi import HTTPException

from clef_flash_server.schema import DecisionRequest
from clef_flash_server.server import Engine, Settings


@unittest.skipUnless(
    os.environ.get("CLEF_TEST_MODEL_PATH"),
    "set CLEF_TEST_MODEL_PATH for real CUDA inference",
)
class CudaInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = Engine(Settings(Path(os.environ["CLEF_TEST_MODEL_PATH"])))

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.engine
        gc.collect()
        torch.cuda.empty_cache()

    def test_fast_kernels_are_bound_to_model_layers(self) -> None:
        from causal_conv1d import causal_conv1d_fn
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

        self.assertTrue(self.engine.fast_kernels)
        layers = [
            module
            for module in self.engine.model.modules()
            if isinstance(module, Qwen3_5GatedDeltaNet)
        ]
        self.assertGreater(len(layers), 0)
        for layer in layers:
            self.assertIs(layer.causal_conv1d_fn, causal_conv1d_fn)
            self.assertIs(layer.chunk_gated_delta_rule, chunk_gated_delta_rule)
            self.assertIsInstance(layer.norm, FusedRMSNormGated)

    def test_fast_convolution_matches_torch_reference(self) -> None:
        from causal_conv1d import causal_conv1d_fn
        from torch.nn import functional

        generator = torch.Generator(device="cuda").manual_seed(42)
        inputs = torch.randn(
            2, 64, 128, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        weight = torch.randn(
            64, 4, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        with torch.inference_mode():
            actual = causal_conv1d_fn(inputs, weight, activation="silu")
            expected = functional.silu(
                functional.conv1d(inputs, weight[:, None, :], padding=3, groups=64)[
                    ..., :128
                ]
            )
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

    def test_varied_length_batch_matches_individual_predictions(self) -> None:
        requests = [
            DecisionRequest.model_validate(
                {
                    "model": "clef-flash" if index % 2 else "Cloudflare/clef-flash",
                    "state": ("Customer reports a network outage. " * (index * 12 + 1)),
                    "questions": {
                        f"category{index}": {
                            "type": "choice",
                            "criteria": {
                                "billing": "Invoices and payments",
                                "technical": "Bugs and outages",
                            },
                        },
                        f"urgency{index}": {
                            "type": "score",
                            "criteria": ["Can wait", "Today"],
                        },
                        f"outage{index}": {
                            "type": "noul",
                            "instructions": "Is a service down?",
                        },
                    },
                }
            )
            for index in range(4)
        ]
        singles = [self.engine.predict(request) for request in requests]
        batched = self.engine.predict_batch(
            [self.engine.prepare(request) for request in requests]
        )
        for expected, actual in zip(singles, batched, strict=True):
            self.assertEqual(actual["model"], expected["model"])
            self.assertEqual(actual["usage"], expected["usage"])
            self.assertEqual(actual["answers"].keys(), expected["answers"].keys())
            for key, answer in actual["answers"].items():
                reference = expected["answers"][key]
                if answer["type"] == "noul":
                    self.assertAlmostEqual(
                        answer["noul"], reference["noul"], delta=0.02
                    )
                else:
                    self.assertEqual(answer["type"], reference["type"])
                    for option, probability in answer["probabilities"].items():
                        self.assertAlmostEqual(
                            probability, reference["probabilities"][option], delta=0.02
                        )
        self.assertEqual(self.engine.largest_batch, 4)

    def test_batch_limit_rejects_before_cuda_allocation(self) -> None:
        request = DecisionRequest.model_validate(
            {
                "model": "clef-flash",
                "state": "A network outage",
                "questions": {"outage": {"type": "noul"}},
            }
        )
        prepared = self.engine.prepare(request)
        before = torch.cuda.memory_allocated()
        with self.assertRaises(ValueError):
            self.engine.predict_batch([prepared] * 5)
        self.assertEqual(torch.cuda.memory_allocated(), before)

    def test_nf4_and_head_are_on_cuda(self) -> None:
        backbone = self.engine.model.language_model
        self.assertTrue(backbone.is_loaded_in_4bit)
        quantized = [
            module for module in backbone.modules() if isinstance(module, Linear4bit)
        ]
        self.assertGreater(len(quantized), 100)
        self.assertTrue(
            all(module.weight.device.type == "cuda" for module in quantized)
        )
        self.assertTrue(
            all(module.weight.quant_state.quant_type == "nf4" for module in quantized)
        )
        output_weight = backbone.get_output_embeddings().weight
        self.assertEqual(output_weight.dtype, torch.bfloat16)
        self.assertEqual(output_weight.device.type, "cuda")
        head_parameter = next(self.engine.model.head.parameters())
        self.assertEqual(head_parameter.dtype, torch.bfloat16)
        self.assertEqual(head_parameter.device.type, "cuda")

    def test_full_context_and_rejection_without_truncation(self) -> None:
        request = DecisionRequest.model_validate(
            {
                "model": "clef-flash",
                "state": "",
                "questions": {
                    "outage": {"type": "noul", "instructions": "Is a service down?"}
                },
            }
        )
        fixed = len(
            self.engine.upstream.encode_record(
                self.engine.processor.tokenizer,
                request.model_dump(),
                max_length=2**31 - 1,
            ).input_ids
        )
        request.state = " neutral" * (16384 - fixed)
        response = self.engine.predict(request)
        self.assertEqual(response["usage"]["input_tokens"], 16384)
        self.assertGreaterEqual(response["answers"]["outage"]["noul"], 0)
        self.assertLessEqual(response["answers"]["outage"]["noul"], 1)
        request.state += " neutral"
        with self.assertRaises(HTTPException) as caught:
            self.engine.predict(request)
        self.assertEqual(caught.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
