"""Ed25519-signed export packages (DESIGN.md §3 review, §7). No ML dependencies (§5).

Package = {filename: bytes} + manifest.json (sha256 per file, meta, key id) + manifest.sig (Ed25519 over the exact
manifest bytes) + public_key.pem. Verification recomputes every hash, checks the file set and the signature, and can
pin a trusted public key (otherwise the embedded key is used, which only proves integrity, not origin).
"""

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

RESERVED = ("manifest.json", "manifest.sig", "public_key.pem")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_or_create_key(path: Path) -> Ed25519PrivateKey:
    """Private key PEM at `path` (created 0600 on first use). Keep it out of git (data/keys/)."""
    if path.exists():
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError(f"{path} is not an Ed25519 key")
        return key
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    path.chmod(0o600)
    return key


def public_pem(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def key_id(pem: bytes) -> str:
    return sha256_hex(pem)[:16]


def build_package(files: dict[str, bytes], key: Ed25519PrivateKey, meta: dict[str, Any]) -> dict[str, bytes]:
    if not files or any(n in RESERVED for n in files):
        raise ValueError("need at least one file and no reserved names")
    pem = public_pem(key)
    manifest = json.dumps(
        {
            "files": {n: sha256_hex(b) for n, b in sorted(files.items())},
            "meta": meta,
            "key_id": key_id(pem),
            "algorithm": "Ed25519",
        },
        sort_keys=True,
    ).encode()
    sig = base64.b64encode(key.sign(manifest))
    return dict(files) | {"manifest.json": manifest, "manifest.sig": sig, "public_key.pem": pem}


def verify_package(pkg: dict[str, bytes], trusted_public_pem: bytes | None = None) -> dict[str, Any]:
    problems: list[str] = []
    try:
        manifest = json.loads(pkg["manifest.json"])
        pem = trusted_public_pem or pkg["public_key.pem"]
        pub = serialization.load_pem_public_key(pem)
        if not isinstance(pub, Ed25519PublicKey):
            raise TypeError("not an Ed25519 public key")
        pub.verify(base64.b64decode(pkg["manifest.sig"]), pkg["manifest.json"])
    except (KeyError, ValueError, TypeError) as e:
        return {"ok": False, "problems": [f"malformed package: {e}"]}
    except InvalidSignature:
        problems.append("signature does not match manifest (manifest edited or wrong key)")
        manifest = json.loads(pkg["manifest.json"])
    listed = set(manifest["files"])
    present = set(pkg) - set(RESERVED)
    problems += [f"missing file: {n}" for n in sorted(listed - present)]
    problems += [f"unlisted file: {n}" for n in sorted(present - listed)]
    problems += [
        f"hash mismatch: {n}" for n in sorted(listed & present) if sha256_hex(pkg[n]) != manifest["files"][n]
    ]
    return {"ok": not problems, "problems": problems}
