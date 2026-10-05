"""Real unit tests for admission configuration and padded batch selection."""

import unittest
from pathlib import Path

from clef_flash_server.server import Settings, select_batch


class BatchSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(Path("weights"))

    def test_empty_and_singleton(self) -> None:
        self.assertEqual(select_batch([], self.settings), [])
        self.assertEqual(select_batch([16384], self.settings), [0])

    def test_groups_short_followers_without_starving_oldest(self) -> None:
        lengths = [512, 16384, 768, 4096, 512]
        indices = select_batch(lengths, self.settings)
        self.assertEqual(indices, [0, 2, 4])
        remaining = [
            length for index, length in enumerate(lengths) if index not in indices
        ]
        self.assertEqual(select_batch(remaining, self.settings), [0])

    def test_counts_padding_and_limits_batch_size(self) -> None:
        self.assertEqual(select_batch([4096] * 8, self.settings), [0, 1, 2, 3])
        self.assertEqual(select_batch([8192, 8193], self.settings), [0])
        self.assertEqual(select_batch([16384, 16384], self.settings), [0])
        self.assertEqual(
            select_batch([4096, 4097, 4096, 4096], self.settings), [0, 1, 2]
        )

    def test_limits_total_length_ratio(self) -> None:
        self.assertEqual(select_batch([512, 1024, 256], self.settings), [0, 1])

    def test_serial_mode(self) -> None:
        settings = Settings(Path("weights"), max_batch_size=1, batch_wait_ms=0)
        self.assertEqual(select_batch([512, 512], settings), [0])

    def test_rejects_invalid_configuration(self) -> None:
        for changes in [
            {"max_batch_size": 0},
            {"max_batch_tokens": 8192},
            {"max_queued_requests": 0},
            {"batch_wait_ms": -1},
            {"batch_wait_ms": float("nan")},
            {"quantization": "int8"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Settings(Path("weights"), **changes)


if __name__ == "__main__":
    unittest.main()
