"""Content-bound release and runtime identities for cache-safe discovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal

from clef_flash_server.release_files import RELEASE_FILES

MODEL_REPOSITORY = "Cloudflare/clef-flash"
MODEL_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
REFERENCE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
type ModelAlias = Literal["clef-flash", "Cloudflare/clef-flash"]
type Quantization = Literal["nf4", "none"]


class IdentityUnavailable(ValueError):
    """Inference may continue, but these files/runtime cannot establish identity."""


def digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def hexadecimal(value: str, length: int) -> bool:
    return bool(re.fullmatch(f"[0-9a-f]{{{length}}}", value))


@dataclass(frozen=True)
class FileIdentity:
    path: str
    bytes: int
    sha256: str
    etag: str


def regular_digest(path: Path, etag: str = "") -> FileIdentity:
    """Read a real regular file once and detect changes during hashing."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise IdentityUnavailable(f"identity input is not a regular file: {path}")
        sha256 = hashlib.sha256()
        git_blob = (
            hashlib.sha1(f"blob {before.st_size}\0".encode(), usedforsecurity=False)
            if hexadecimal(etag, 40)
            else None
        )
        size = 0
        while chunk := stream.read(8 * 1024 * 1024):
            size += len(chunk)
            sha256.update(chunk)
            if git_blob is not None:
                git_blob.update(chunk)
        after = os.fstat(stream.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or size != before.st_size:
            raise IdentityUnavailable(f"identity input changed during hashing: {path}")
    actual = sha256.hexdigest()
    if etag:
        if hexadecimal(etag, 64):
            expected_digest = actual
        elif git_blob is not None:
            expected_digest = git_blob.hexdigest()
        else:
            raise IdentityUnavailable(f"unsupported Hub content digest: {path}")
        if expected_digest != etag:
            raise IdentityUnavailable(
                f"Hub content digest differs from loaded file: {path}"
            )
    return FileIdentity(str(path), size, actual, etag)


@dataclass(frozen=True)
class Release:
    revision: str
    files: tuple[FileIdentity, ...]
    fingerprint: str

    @classmethod
    def capture(cls, root: Path) -> Release:
        expected = {file.path for file in RELEASE_FILES}
        observed = {
            path.name
            for path in root.iterdir()
            if path.suffix in {".json", ".safetensors", ".jinja", ".bin", ".pt", ".pth"}
        }
        if observed != expected:
            raise IdentityUnavailable(
                "release runtime files differ from the pinned Hub manifest"
            )
        files: list[FileIdentity] = []
        for pin in RELEASE_FILES:
            path = root / pin.path
            if path.stat().st_size != pin.bytes:
                raise IdentityUnavailable(
                    f"release file size differs from the pinned Hub manifest: {pin.path}"
                )
            file = regular_digest(path, pin.etag)
            files.append(FileIdentity(pin.path, file.bytes, file.sha256, pin.etag))
        payload = [asdict(file) for file in files]
        return cls(MODEL_REVISION, tuple(files), digest(payload))

    def check_unchanged(self, root: Path) -> None:
        if self != Release.capture(root):
            raise IdentityUnavailable(
                "release files changed while the model was loading"
            )


@dataclass(frozen=True)
class PackageIdentity:
    name: str
    version: str
    store_roots: tuple[str, ...]


def store_root(path: Path) -> str:
    resolved = path.resolve(strict=True)
    parts = resolved.parts
    if (
        len(parts) < 4
        or parts[:3] != ("/", "nix", "store")
        or not re.fullmatch("[0-9abcdfghijklmnpqrsvwxyz]{32}-.+", parts[3])
    ):
        raise IdentityUnavailable(
            f"dependency is outside the immutable Nix store: {path}"
        )
    return str(Path(*parts[:4]))


def packages() -> tuple[PackageIdentity, ...]:
    """Bind the actual immutable installed distributions, not just their versions."""
    result: set[PackageIdentity] = set()
    for distribution in metadata.distributions():
        files = distribution.files
        if files is None or not files:
            raise IdentityUnavailable("installed dependency lacks a file manifest")
        roots = {
            store_root(Path(str(distribution.locate_file(file))))
            for file in files
            if not file.name.endswith(".pyc")
        }
        name = distribution.metadata.get("Name")
        if not name or not roots:
            raise IdentityUnavailable("installed dependency lacks a package identity")
        result.add(PackageIdentity(name, distribution.version, tuple(sorted(roots))))
    return tuple(
        sorted(
            result,
            key=lambda package: (package.name, package.version, package.store_roots),
        )
    )


@dataclass(frozen=True)
class ModelIdentity:
    model: str
    encoder_identity: str
    release_fingerprint: str
    build_fingerprint: str

    @classmethod
    def create(
        cls,
        release: Release,
        quantization: Quantization,
        encoder_identity: str,
        build: object,
    ) -> ModelIdentity:
        if quantization not in ("nf4", "none") or not encoder_identity:
            raise IdentityUnavailable(
                "loaded precision and encoder identity are required"
            )
        fingerprint = digest(
            {"schema_version": 1, "release": release.fingerprint, "runtime": build}
        )
        model = (
            f"{MODEL_REPOSITORY}@{release.revision};compute=bf16;"
            f"quantization={quantization};server={fingerprint}"
        )
        return cls(model, encoder_identity, release.fingerprint, fingerprint)

    def document(self, requested_model: ModelAlias) -> dict[str, str | int]:
        return {
            "schema_version": 1,
            "requested_model": requested_model,
            "model": self.model,
            "encoder_identity": self.encoder_identity,
            "release_fingerprint": self.release_fingerprint,
            "build_fingerprint": self.build_fingerprint,
        }
