"""Authentication, authorization, and document ACL helpers."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

from fastapi import Depends, Header, HTTPException, status

from app.utils import load_config

# Simple in-memory users for on-prem bootstrap (replace with LDAP/OIDC in prod)
_USERS: dict[str, dict[str, Any]] = {
    "admin": {
        "password_sha256": hashlib.sha256(b"admin").hexdigest(),
        "roles": ["admin", "reader"],
        "access_scopes": ["internal", "confidential"],
    },
    "reader": {
        "password_sha256": hashlib.sha256(b"reader").hexdigest(),
        "roles": ["reader"],
        "access_scopes": ["internal"],
    },
}


def auth_enabled() -> bool:
    return bool(load_config().get("auth_enabled", False))


def verify_password(username: str, password: str) -> dict[str, Any] | None:
    user = _USERS.get(username)
    if not user:
        return None
    digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(digest, user["password_sha256"]):
        return None
    return {"username": username, "roles": user["roles"], "access_scopes": user["access_scopes"]}


def issue_token(username: str, roles: list[str], access_scopes: list[str]) -> str:
    """Lightweight HMAC token (avoid hard jose dependency at runtime if missing)."""
    cfg = load_config()
    secret = str(cfg.get("jwt_secret") or "change-me-in-production")
    exp = int(time.time()) + 86400
    payload = f"{username}|{','.join(roles)}|{','.join(access_scopes)}|{exp}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}|{sig}"


def parse_token(token: str) -> dict[str, Any] | None:
    try:
        username, roles_s, scopes_s, exp_s, sig = token.split("|", 4)
        cfg = load_config()
        secret = str(cfg.get("jwt_secret") or "change-me-in-production")
        payload = f"{username}|{roles_s}|{scopes_s}|{exp_s}"
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return None
        if int(exp_s) < int(time.time()):
            return None
        return {
            "username": username,
            "roles": [r for r in roles_s.split(",") if r],
            "access_scopes": [s for s in scopes_s.split(",") if s],
        }
    except Exception:  # noqa: BLE001
        return None


def document_visible(doc_access_scope: str, user_scopes: list[str] | None) -> bool:
    if not auth_enabled():
        return True
    if not user_scopes:
        return doc_access_scope in {"internal", "public", ""}
    return doc_access_scope in user_scopes or "admin" in (user_scopes or [])


async def get_current_user(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    cfg = load_config()
    if not auth_enabled():
        return {
            "username": "anonymous",
            "roles": ["reader"],
            "access_scopes": [str(cfg.get("default_access_scope") or "internal")],
        }
    admin_key = str(cfg.get("admin_api_key") or "")
    if admin_key and x_api_key and hmac.compare_digest(admin_key, x_api_key):
        return {
            "username": "api-key",
            "roles": ["admin", "reader"],
            "access_scopes": ["internal", "confidential"],
        }
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    user = parse_token(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return user


RequireUser = Depends(get_current_user)
