# sample_codebase/inventory/auth.py

"""
Password hashing and API token helpers.
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

PBKDF2_ITERATIONS = 600_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    """
    Hashes a password with PBKDF2-HMAC-SHA256 and a random 16 byte salt.

    The result is stored as "salt_hex$hash_hex" so the salt can be recovered
    when the password is verified.
    """
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Checks a password against a stored hash using a constant time comparison."""
    salt_hex, digest_hex = stored.split("$", 1)
    candidate = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate.split("$", 1)[1], digest_hex)


def create_api_token(ttl_minutes: int = 60) -> tuple[str, datetime]:
    """Creates a random URL safe API token and returns it with its expiry time."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    return token, expires_at


def is_token_expired(expires_at: datetime) -> bool:
    return datetime.now(timezone.utc) >= expires_at
