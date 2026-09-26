import hashlib

import pytest
import yaml

from governance import model_registry as mr
from governance import rbac


def test_weight_verification_accepts_match_and_rejects_tamper(tmp_path):
    (tmp_path / "m" / "clip").mkdir(parents=True)
    w = tmp_path / "m" / "clip" / "w.pt"
    w.write_bytes(b"weights")
    man = tmp_path / "models.yaml"
    man.write_text(yaml.safe_dump({"clip": {"sha256": {"w.pt": hashlib.sha256(b"weights").hexdigest()}}}))
    assert mr.verified.__wrapped__("clip", "w.pt", man, tmp_path / "m") == w
    w.write_bytes(b"weightz")  # tampered file
    with pytest.raises(mr.WeightMismatch):
        mr.verified.__wrapped__("clip", "w.pt", man, tmp_path / "m")
    with pytest.raises(mr.WeightMismatch):
        mr.verified.__wrapped__("clip", "other.pt", man, tmp_path / "m")  # not in manifest
    assert "sha256" in mr.verify_all(man, tmp_path / "m")["clip/w.pt"]


def test_rbac_roles_and_disabled_mode(tmp_path):
    f = tmp_path / "users.yaml"
    assert rbac.load(f) is None and rbac.authorize(None, None, "admin")["auth"] is False
    ka = rbac.add("ana", "analyst", f)
    kv = rbac.add("vic", "viewer", f)
    users = rbac.load(f)
    assert ka not in f.read_text() and kv not in f.read_text()  # only hashes stored
    assert rbac.authorize(users, ka, "analyst")["name"] == "ana"
    assert rbac.authorize(users, kv, "viewer")["name"] == "vic"
    for key, need in ((kv, "analyst"), (ka, "admin"), (None, "viewer"), ("bogus", "viewer")):
        with pytest.raises(PermissionError):
            rbac.authorize(users, key, need)
