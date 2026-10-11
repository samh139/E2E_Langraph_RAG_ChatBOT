"""Mint a local HS256 JWT for trying a configured RBAC role in development."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import jwt
from dotenv import load_dotenv

from app.security.access_control import load_rbac_policy


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / "app" / "configs" / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, help="A configured JWT role")
    parser.add_argument("--subject", default="local-rbac-user")
    parser.add_argument("--minutes", type=int, default=30)
    args = parser.parse_args()

    if os.getenv("APP_ENV", "").lower() != "development":
        parser.error("Test tokens can only be created when APP_ENV=development.")
    if os.getenv("JWT_ALLOW_TEST_TOKEN_CREATION", "false").lower() != "true":
        parser.error("Set JWT_ALLOW_TEST_TOKEN_CREATION=true for local testing only.")
    if os.getenv("JWT_ALGORITHMS", "").strip() != "HS256":
        parser.error("Local test token creation requires JWT_ALGORITHMS=HS256.")

    secret = os.getenv("JWT_SECRET", "")
    issuer = os.getenv("JWT_ISSUER", "").strip()
    audience = os.getenv("JWT_AUDIENCE", "").strip()
    if len(secret) < 32 or not issuer or not audience:
        parser.error("Configure a 32+ character JWT_SECRET, JWT_ISSUER, and JWT_AUDIENCE.")
    if args.minutes < 1 or args.minutes > 60:
        parser.error("--minutes must be between 1 and 60.")
    if args.role not in load_rbac_policy()["role_permissions"]:
        parser.error(f"Unknown role {args.role!r}; check app/configs/rbac_roles.yaml.")

    now = int(time.time())
    token = jwt.encode(
        {
            "sub": args.subject,
            "roles": [args.role],
            "iss": issuer,
            "aud": audience,
            "iat": now,
            "exp": now + args.minutes * 60,
        },
        secret,
        algorithm="HS256",
    )
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
