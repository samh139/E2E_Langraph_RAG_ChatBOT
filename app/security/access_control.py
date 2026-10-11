"""JWT validation, internal request signing, and role policy helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import jwt
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _jwks_client(url: str):
    return jwt.PyJWKClient(url)


def _jwt_key(token: str, algorithms: list[str]):
    jwks_url = os.getenv("JWT_JWKS_URL", "").strip()
    if jwks_url:
        return _jwks_client(jwks_url).get_signing_key_from_jwt(token).key

    public_key = os.getenv("JWT_PUBLIC_KEY", "").replace("\\n", "\n").strip()
    if public_key:
        return public_key

    secret = os.getenv("JWT_SECRET", "")
    if secret and all(algorithm.startswith("HS") for algorithm in algorithms):
        return secret

    raise RuntimeError(
        "Configure JWT_JWKS_URL, JWT_PUBLIC_KEY, or JWT_SECRET (for an HMAC algorithm)."
    )


def validate_access_token(token: str) -> dict[str, Any]:
    """Validate a JWT and return only the identity and role claims we use."""
    issuer = os.getenv("JWT_ISSUER", "").strip()
    audience = os.getenv("JWT_AUDIENCE", "").strip()
    if not issuer or not audience:
        raise RuntimeError("JWT_ISSUER and JWT_AUDIENCE must be configured.")

    algorithms = [
        item.strip()
        for item in os.getenv("JWT_ALGORITHMS", "RS256").split(",")
        if item.strip()
    ]
    if not algorithms:
        raise RuntimeError("JWT_ALGORITHMS must contain at least one algorithm.")

    key = _jwt_key(token, algorithms)
    claims = jwt.decode(
        token,
        key=key,
        algorithms=algorithms,
        issuer=issuer,
        audience=audience,
        options={"require": ["sub", "iss", "aud", "exp"]},
    )
    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id.strip():
        raise jwt.InvalidTokenError("JWT subject must be a non-empty string")

    role_claim = os.getenv("JWT_ROLES_CLAIM", "roles").strip() or "roles"
    raw_roles: Any = claims
    for claim_part in role_claim.split("."):
        raw_roles = raw_roles.get(claim_part) if isinstance(raw_roles, dict) else None
    if raw_roles is None:
        raw_roles = claims.get("role", [])
    if isinstance(raw_roles, str):
        raw_roles = [raw_roles]
    if not isinstance(raw_roles, list) or not all(
        isinstance(role, str) and role.strip() for role in raw_roles
    ):
        raise jwt.InvalidTokenError("JWT role claim must be a string or list of strings")

    return {
        "user_id": user_id.strip(),
        "roles": sorted({role.strip() for role in raw_roles}),
        "expires_at": int(claims["exp"]),
    }


def sign_internal_request(
    *, session_id: str, connection_id: str, message: str, principal: dict[str, Any]
) -> dict[str, Any]:
    """Sign gateway-verified identity and request fields for the internal Kafka hop."""
    secret = os.getenv("INTERNAL_CONTEXT_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("INTERNAL_CONTEXT_SECRET must be configured with 32+ characters.")
    body = {
        "session_id": session_id,
        "connection_id": connection_id,
        "message": message,
        "user_id": principal["user_id"],
        "roles": principal["roles"],
        "allowed_roles": principal["allowed_roles"],
        "issued_at": int(time.time()),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(secret.encode(), canonical, hashlib.sha256).digest()
    body["signature"] = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return body


def verify_internal_request(request: dict[str, Any]) -> dict[str, Any]:
    """Verify gateway assertion and reject stale or altered Kafka requests."""
    secret = os.getenv("INTERNAL_CONTEXT_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("INTERNAL_CONTEXT_SECRET must be configured with 32+ characters.")
    signature = request.get("signature")
    if not isinstance(signature, str):
        raise ValueError("Missing signed authentication context")
    body = {key: value for key, value in request.items() if key != "signature"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    expected = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), canonical, hashlib.sha256).digest()
    ).decode().rstrip("=")
    if not hmac.compare_digest(signature, expected):
        raise ValueError("Invalid signed authentication context")

    issued_at = body.get("issued_at")
    max_age = int(os.getenv("INTERNAL_CONTEXT_MAX_AGE_SECONDS", "60"))
    if not isinstance(issued_at, int) or abs(int(time.time()) - issued_at) > max_age:
        raise ValueError("Expired signed authentication context")
    for field in ("session_id", "connection_id", "message", "user_id"):
        if not isinstance(body.get(field), str) or not body[field].strip():
            raise ValueError(f"Invalid signed request field: {field}")
    if not isinstance(body.get("roles"), list) or not all(
        isinstance(role, str) for role in body["roles"]
    ):
        raise ValueError("Invalid signed request roles")
    if not isinstance(body.get("allowed_roles"), list) or not all(
        isinstance(role, str) for role in body["allowed_roles"]
    ):
        raise ValueError("Invalid signed request permissions")
    policy = load_rbac_policy()
    valid_roles = set(policy["roles_mapping"])
    if not body["allowed_roles"] or not set(body["allowed_roles"]).issubset(valid_roles):
        raise ValueError("Signed request contains invalid or empty permissions")
    expected_permissions = effective_permissions(body["roles"])
    if sorted(set(body["allowed_roles"])) != expected_permissions:
        raise ValueError("Signed permissions do not match the configured role policy")
    return body


@lru_cache(maxsize=1)
def load_rbac_policy() -> dict[str, Any]:
    path = PROJECT_ROOT / "app" / "configs" / "rbac_roles.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for required in ("roles_mapping", "role_permissions"):
        if not isinstance(config.get(required), dict):
            raise ValueError(f"{required} must be defined as a mapping in {path}")
    document_roles = set(config["roles_mapping"])
    for role, filenames in config["roles_mapping"].items():
        if not isinstance(filenames, list) or not all(
            isinstance(filename, str) and filename.strip() for filename in filenames
        ):
            raise ValueError(f"Invalid filename list for document role {role!r}")
    for user_role, allowed in config["role_permissions"].items():
        if not isinstance(allowed, list) or not all(
            isinstance(role, str) and role in document_roles for role in allowed
        ):
            raise ValueError(f"Invalid document-role permissions for {user_role!r}")
    return config


def effective_permissions(user_roles: list[str]) -> list[str]:
    """Return document roles permitted by verified JWT roles; unknown roles fail closed."""
    permissions = load_rbac_policy()["role_permissions"]
    resolved: set[str] = set()
    for role in user_roles:
        configured = permissions.get(role)
        if configured is not None:
            if not isinstance(configured, list) or not all(
                isinstance(item, str) for item in configured
            ):
                raise ValueError(f"Invalid permissions configured for role {role!r}")
            resolved.update(configured)
    known_permissions = set(permissions)
    resolved.intersection_update(known_permissions)
    return sorted(resolved)


def memory_scope_id(allowed_roles: list[str]) -> str:
    """Versioned scope key prevents memory created under broader access being reused."""
    config = load_rbac_policy()
    scope = {
        "version": config.get("policy_version", 1),
        "allowed_roles": sorted(set(allowed_roles)),
        "role_permissions": config["role_permissions"],
        "document_role_assignments": config["roles_mapping"],
        "security_defaults": config.get("security_defaults", {}),
    }
    canonical = json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
