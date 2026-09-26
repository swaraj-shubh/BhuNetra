import json

import pytest

from governance import sign


def test_signed_package_verifies_and_tampering_is_detected(tmp_path):
    key = sign.load_or_create_key(tmp_path / "k")
    files = {"events.geojson": b'{"type":"FeatureCollection","features":[]}', "provenance.json": b'{"a":1}'}
    pkg = sign.build_package(files, key, meta={"export_id": "X"})
    assert sign.verify_package(pkg) == {"ok": True, "problems": []}

    tampered = dict(pkg) | {"events.geojson": b'{"type":"FeatureCollection","features":[1]}'}
    res = sign.verify_package(tampered)
    assert res["ok"] is False and any("events.geojson" in p for p in res["problems"])

    forged = json.loads(pkg["manifest.json"])  # attacker edits the hash to match the edited file
    forged["files"]["events.geojson"] = sign.sha256_hex(tampered["events.geojson"])
    res = sign.verify_package(tampered | {"manifest.json": json.dumps(forged).encode()})
    assert res["ok"] is False and any("signature" in p for p in res["problems"])


def test_key_is_persistent_and_other_keys_fail(tmp_path):
    k1 = sign.load_or_create_key(tmp_path / "k")
    assert sign.public_pem(sign.load_or_create_key(tmp_path / "k")) == sign.public_pem(k1)
    pkg = sign.build_package({"a.txt": b"x"}, k1, meta={})
    other = sign.public_pem(sign.load_or_create_key(tmp_path / "other"))
    assert sign.verify_package(pkg, trusted_public_pem=other)["ok"] is False


def test_missing_and_extra_files_fail(tmp_path):
    key = sign.load_or_create_key(tmp_path / "k")
    pkg = sign.build_package({"a.txt": b"x", "b.txt": b"y"}, key, meta={})
    assert not sign.verify_package({k: v for k, v in pkg.items() if k != "b.txt"})["ok"]
    assert not sign.verify_package(pkg | {"evil.txt": b"z"})["ok"]


def test_empty_package_rejected(tmp_path):
    with pytest.raises(ValueError):
        sign.build_package({}, sign.load_or_create_key(tmp_path / "k"), meta={})
