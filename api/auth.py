"""API role checks (M8). X-API-Key header -> governance/rbac.py user."""

from fastapi import Depends, Header, HTTPException


def role(need: str):
    """FastAPI dependency: X-API-Key must map to a user with at least `need` (governance/rbac.py, M8)."""
    from governance import rbac

    def dep(x_api_key: str | None = Header(default=None)) -> dict:
        try:
            return rbac.authorize(rbac.load(), x_api_key, need)
        except PermissionError as e:
            raise HTTPException(401 if not x_api_key else 403, str(e)) from e

    return dep


VIEWER = Depends(role("viewer"))
ANALYST = Depends(role("analyst"))
ADMIN = Depends(role("admin"))
