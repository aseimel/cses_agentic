"""Authentication profile helpers for CSES model providers.

Codex OAuth is treated as a token-source integration: this app reads Codex
auth state when needed, but never copies or rotates Codex refresh tokens.
"""

from __future__ import annotations

import base64
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.settings import get_install_dir


CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"


class AuthProvider(str, Enum):
    API_KEY = "api_key"
    OPENWEBUI = "openwebui"
    CODEX_OAUTH = "codex_oauth"


@dataclass
class AuthProfile:
    provider_id: str
    profile_id: str
    account_id: str = ""
    account_email: str = ""
    token_source: str = ""
    status: str = "unknown"
    created_at: str = ""
    last_checked_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class CodexAuthStatus:
    status: str
    message: str
    codex_installed: bool
    auth_path: str
    account_id: str = ""
    account_email: str = ""
    expires_at: str = ""
    token_source: str = "codex_auth_json"

    @property
    def signed_in(self) -> bool:
        return self.status == "signed_in"


class AuthProfileStore:
    """Persist non-secret auth metadata in the CSES install directory."""

    def __init__(self, project_root: Path | None = None):
        self.path = get_install_dir(project_root) / "auth-profiles.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "profiles": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"schema_version": 1, "profiles": []}

    def upsert(self, profile: AuthProfile) -> Path:
        payload = self.load()
        profiles = [
            item
            for item in payload.get("profiles", [])
            if not (
                item.get("provider_id") == profile.provider_id
                and item.get("profile_id") == profile.profile_id
            )
        ]
        profiles.append(profile.to_dict())
        payload["schema_version"] = 1
        payload["profiles"] = profiles
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return self.path


class CodexOAuthAuthSource:
    """Read local Codex OAuth state without owning the refresh lifecycle."""

    def __init__(self, auth_path: Path = CODEX_AUTH_PATH):
        self.auth_path = Path(auth_path)

    def check(self) -> CodexAuthStatus:
        codex_installed = shutil.which("codex") is not None or shutil.which("codex.exe") is not None
        if not self.auth_path.exists():
            return CodexAuthStatus(
                status="not_signed_in",
                message="Codex sign-in was not found. Sign in with Codex to use ChatGPT/Codex OAuth models.",
                codex_installed=codex_installed,
                auth_path=str(self.auth_path),
            )
        try:
            payload = json.loads(self.auth_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return CodexAuthStatus(
                status="unknown",
                message=f"Codex auth file could not be read: {exc}",
                codex_installed=codex_installed,
                auth_path=str(self.auth_path),
            )

        tokens = payload.get("tokens") or {}
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        account_id = tokens.get("account_id") or _jwt_claim(tokens.get("id_token"), "sub") or ""
        account_email = _jwt_claim(tokens.get("id_token"), "email") or ""
        expires_at = _jwt_expiry(tokens.get("access_token"))

        if not access_token or not refresh_token or not account_id:
            return CodexAuthStatus(
                status="unknown",
                message="Codex auth exists but does not contain the expected access, refresh, and account fields.",
                codex_installed=codex_installed,
                auth_path=str(self.auth_path),
                account_id=account_id,
                account_email=account_email,
                expires_at=expires_at,
            )

        if expires_at and _is_past(expires_at):
            return CodexAuthStatus(
                status="expired",
                message="Codex access token appears expired. Refresh sign-in with Codex.",
                codex_installed=codex_installed,
                auth_path=str(self.auth_path),
                account_id=account_id,
                account_email=account_email,
                expires_at=expires_at,
            )

        return CodexAuthStatus(
            status="signed_in",
            message=f"Signed in with Codex OAuth{f' as {account_email}' if account_email else ''}.",
            codex_installed=codex_installed,
            auth_path=str(self.auth_path),
            account_id=account_id,
            account_email=account_email,
            expires_at=expires_at,
        )

    def persist_metadata(self, project_root: Path | None = None) -> Path:
        status = self.check()
        now = datetime.now(timezone.utc).isoformat()
        profile = AuthProfile(
            provider_id=AuthProvider.CODEX_OAUTH.value,
            profile_id=status.account_email or status.account_id or "default",
            account_id=status.account_id,
            account_email=status.account_email,
            token_source=status.auth_path,
            status=status.status,
            created_at=now,
            last_checked_at=now,
        )
        return AuthProfileStore(project_root).upsert(profile)

    def access_token(self) -> str:
        payload = json.loads(self.auth_path.read_text(encoding="utf-8"))
        token = (payload.get("tokens") or {}).get("access_token")
        if not token:
            raise RuntimeError("Codex OAuth access token not found. Refresh Codex sign-in.")
        return token


def _jwt_payload(token: str | None) -> dict[str, Any]:
    if not token or "." not in token:
        return {}
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        return json.loads(base64.urlsafe_b64decode(part.encode("ascii")).decode("utf-8"))
    except Exception:
        return {}


def _jwt_claim(token: str | None, claim: str) -> str:
    value = _jwt_payload(token).get(claim)
    return str(value) if value else ""


def _jwt_expiry(token: str | None) -> str:
    exp = _jwt_payload(token).get("exp")
    if not exp:
        return ""
    try:
        return datetime.fromtimestamp(int(exp), timezone.utc).isoformat()
    except Exception:
        return ""


def _is_past(iso_time: str) -> bool:
    try:
        return datetime.fromisoformat(iso_time).astimezone(timezone.utc) <= datetime.now(timezone.utc)
    except Exception:
        return False
