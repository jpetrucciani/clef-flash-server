"""Real filesystem checks and pure identity arithmetic, without model substitutes."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from clef_flash_server.identity import (
    MODEL_REVISION,
    REFERENCE_SHA256,
    TOKENIZER_SHA256,
    FileIdentity,
    IdentityUnavailable,
    ModelIdentity,
    Release,
    digest,
    packages,
    regular_digest,
    store_root,
)
from clef_flash_server.release_files import RELEASE_FILES


class FileIdentityTests(unittest.TestCase):
    def test_raw_and_git_blob_digests_match_real_bytes(self) -> None:
        content = b"Unicode: \xe2\x98\x83\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(content)
            sha256 = hashlib.sha256(content).hexdigest()
            git = hashlib.sha1(
                f"blob {len(content)}\0".encode() + content, usedforsecurity=False
            ).hexdigest()
            for expected in ("", sha256, git):
                with self.subTest(expected=expected):
                    observed = regular_digest(path, expected)
                    self.assertEqual(observed.bytes, len(content))
                    self.assertEqual(observed.sha256, sha256)
                    self.assertEqual(observed.etag, expected)
            path.write_bytes(b"Unicode: altered\n")
            for expected in (sha256, git, "unsupported"):
                with (
                    self.subTest(expected=expected),
                    self.assertRaises(IdentityUnavailable),
                ):
                    regular_digest(path, expected)

    def test_fifo_and_directory_are_rejected_without_waiting_for_a_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fifo = root / "model.safetensors"
            os.mkfifo(fifo)
            for path in (fifo, root):
                with (
                    self.subTest(path=path),
                    self.assertRaises((IdentityUnavailable, OSError)),
                ):
                    regular_digest(path)

    def test_snapshot_symlink_is_verified_against_target_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "blob"
            blob.write_bytes(b"actual snapshot content")
            snapshot = root / "snapshot.json"
            snapshot.symlink_to(blob)
            expected = hashlib.sha256(blob.read_bytes()).hexdigest()
            self.assertEqual(regular_digest(snapshot, expected).sha256, expected)
            blob.write_bytes(b"changed snapshot content")
            with self.assertRaises(IdentityUnavailable):
                regular_digest(snapshot, expected)

    def test_mutable_dependency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dependency.py"
            path.write_text("VERSION = '1.0'\n")
            with self.assertRaises(IdentityUnavailable):
                store_root(path)

    def test_actual_python_and_installed_dependencies_have_immutable_roots(
        self,
    ) -> None:
        self.assertTrue(store_root(Path(sys.executable)).startswith("/nix/store/"))
        observed = packages()
        self.assertGreater(len(observed), 0)
        self.assertIn("blake3", {package.name.lower() for package in observed})
        self.assertEqual(observed, packages())
        self.assertTrue(all(package.store_roots for package in observed))


class ReleaseIdentityTests(unittest.TestCase):
    def test_missing_extra_and_wrong_sized_release_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(IdentityUnavailable):
                Release.capture(root)
            for file in RELEASE_FILES:
                (root / file.path).touch()
            with self.assertRaisesRegex(IdentityUnavailable, "file size"):
                Release.capture(root)
            (root / "unexpected.safetensors").touch()
            with self.assertRaisesRegex(IdentityUnavailable, "runtime files"):
                Release.capture(root)

    def test_manifest_covers_shards_head_processor_and_tokenizer(self) -> None:
        files = {file.path: file for file in RELEASE_FILES}
        self.assertEqual(len(files), len(RELEASE_FILES))
        self.assertEqual(len(files), 13)
        self.assertTrue(
            all(
                f"model-{index:05d}-of-00004.safetensors" in files
                for index in range(1, 5)
            )
        )
        for name in (
            "joint_head.safetensors",
            "joint_head_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
        ):
            self.assertIn(name, files)
        self.assertTrue(all(file.bytes > 0 for file in RELEASE_FILES))


class ModelIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        # These values test identity arithmetic, not an observed server/model.
        file = FileIdentity("fixture", 1, "a" * 64, "a" * 64)
        self.release = Release(MODEL_REVISION, (file,), digest(["fixture"]))
        self.encoder = "fixture-encoder"

    def test_canonical_hash_is_order_independent_and_rejects_nonfinite_values(
        self,
    ) -> None:
        self.assertEqual(digest({"a": 1, "b": "雪"}), digest({"b": "雪", "a": 1}))
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                digest(value)

    def test_runtime_release_and_quantization_changes_do_not_reuse_identity(
        self,
    ) -> None:
        runtime = {
            "settings": {"max_batch_size": 4},
            "source": "source-a",
            "packages": "packages-a",
            "driver": "driver-a",
            "kernels": "kernels-a",
        }
        original = ModelIdentity.create(self.release, "nf4", self.encoder, runtime)
        self.assertEqual(
            original, ModelIdentity.create(self.release, "nf4", self.encoder, runtime)
        )
        for key in runtime:
            changed = {**runtime, key: "changed"}
            with self.subTest(key=key):
                self.assertNotEqual(
                    original.model,
                    ModelIdentity.create(
                        self.release, "nf4", self.encoder, changed
                    ).model,
                )
        self.assertNotEqual(
            original.model,
            ModelIdentity.create(self.release, "none", self.encoder, runtime).model,
        )
        changed_release = Release(MODEL_REVISION, (), digest(["different fixture"]))
        self.assertNotEqual(
            original.model,
            ModelIdentity.create(changed_release, "nf4", self.encoder, runtime).model,
        )

    def test_discovery_has_full_identity_and_retains_requested_alias(self) -> None:
        identity = ModelIdentity.create(self.release, "nf4", self.encoder, {})
        self.assertIn(f"Cloudflare/clef-flash@{MODEL_REVISION};", identity.model)
        self.assertIn(";compute=bf16;quantization=nf4;server=", identity.model)
        for alias in ("clef-flash", "Cloudflare/clef-flash"):
            document = identity.document(alias)
            self.assertEqual(document["requested_model"], alias)
            self.assertEqual(document["model"], identity.model)
            self.assertEqual(document["encoder_identity"], self.encoder)
            self.assertEqual(document["schema_version"], 1)

    def test_missing_encoder_is_rejected(self) -> None:
        with self.assertRaises(IdentityUnavailable):
            ModelIdentity.create(self.release, "nf4", "", {})


@unittest.skipUnless(
    os.environ.get("CLEF_TEST_MODEL_PATH"),
    "set CLEF_TEST_MODEL_PATH to verify the actual published release on CPU",
)
class PublishedReleaseTests(unittest.TestCase):
    def test_real_release_and_loaded_tokenizer_match_the_pinned_encoder(self) -> None:
        import blake3
        import cloudflare_clef_release
        from transformers import AutoProcessor

        root = Path(os.environ["CLEF_TEST_MODEL_PATH"])
        release = Release.capture(root)
        self.assertEqual(release.revision, MODEL_REVISION)
        self.assertEqual(len(release.files), 13)
        self.assertEqual(
            regular_digest(Path(cloudflare_clef_release.__file__)).sha256,
            REFERENCE_SHA256,
        )
        self.assertEqual(
            regular_digest(root / "tokenizer.json").sha256, TOKENIZER_SHA256
        )
        raw = (root / "tokenizer.json").read_bytes()
        processor = AutoProcessor.from_pretrained(root, local_files_only=True)
        self.assertEqual(
            json.loads(processor.tokenizer.backend_tokenizer.to_str()), json.loads(raw)
        )
        self.assertEqual(
            blake3.blake3(raw).hexdigest(),
            "16066ad27076349236a282e9351f8bab5aa9b33803f12f7ab9dca401909d7273",
        )


if __name__ == "__main__":
    unittest.main()
