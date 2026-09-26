"""Role-based access (M8). Offline API keys; only their sha256 is stored (data/keys/users.yaml, not in git).

Roles: viewer < analyst < admin. Enforced when the users file exists; otherwise auth is off (dev) and /stats says so.
    python -m governance.rbac add <name> <role>     # prints the new key once
"""

import hashlib
import secrets
import sys
from pathlib import Path

import yaml

from configs.settings import paths

ROLES = ("viewer", "analyst", "admin")
USERS = paths.data / "keys" / "users.yaml"


def _h(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load(path: Path = USERS) -> dict[str, dict] | None:
    """key_sha256 -> {name, role}; None when auth is disabled (no users file)."""
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {u["key_sha256"]: {"name": u["name"], "role": u["role"]} for u in data.get("users", [])}


def authorize(users: dict[str, dict] | None, key: str | None, need: str) -> dict:
    """Returns the user; raises PermissionError (no/unknown key or too low a role)."""
    if users is None:
        return {"name": "anonymous", "role": "admin", "auth": False}
    u = users.get(_h(key)) if key else None
    if u is None:
        raise PermissionError("missing or unknown API key")
    if ROLES.index(u["role"]) < ROLES.index(need):
        raise PermissionError(f"role {u['role']} < {need}")
    return u | {"auth": True}


def add(name: str, role: str, path: Path = USERS) -> str:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    data = data or {"users": []}
    key = secrets.token_urlsafe(24)
    data["users"] = [u for u in data["users"] if u["name"] != name] + [
        {"name": name, "role": role, "key_sha256": _h(key)}
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)
    return key


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "add":
        print(add(sys.argv[2], sys.argv[3]))
    else:
        sys.exit(__doc__)
