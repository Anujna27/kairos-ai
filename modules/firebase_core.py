"""Framework-independent Firebase configuration and ID-token verification."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

_env_file = Path(__file__).resolve().parents[1] / ".env"
if load_dotenv is not None:
    # Resolve the project configuration independently of the directory from
    # which uvicorn, Streamlit, or a process manager was launched.
    load_dotenv(dotenv_path=_env_file)
else:
    if _env_file.is_file():
        for line in _env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


class FirebaseConfigurationError(RuntimeError):
    """Firebase server or browser configuration is missing or invalid."""


class InvalidFirebaseToken(PermissionError):
    """A browser-supplied Firebase ID token is invalid or revoked."""


_WEB_ENV_KEYS = {
    "apiKey": "FIREBASE_API_KEY",
    "authDomain": "FIREBASE_AUTH_DOMAIN",
    "projectId": "FIREBASE_PROJECT_ID",
    "storageBucket": "FIREBASE_STORAGE_BUCKET",
    "messagingSenderId": "FIREBASE_MESSAGING_SENDER_ID",
    "appId": "FIREBASE_APP_ID",
}


def _setting(name: str) -> str:
    return os.getenv(name, "").strip()


def firebase_web_config() -> tuple[dict[str, str] | None, list[str]]:
    config = {key: _setting(env_name) for key, env_name in _WEB_ENV_KEYS.items()}
    missing = [key for key, value in config.items() if not value]
    return (None, missing) if missing else (config, [])


def _admin_credentials() -> dict[str, Any]:
    raw_json = _setting("FIREBASE_ADMIN_CREDENTIALS_JSON")
    if raw_json:
        try:
            value = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise FirebaseConfigurationError("FIREBASE_ADMIN_CREDENTIALS_JSON is not valid JSON.") from exc
        if not isinstance(value, dict):
            raise FirebaseConfigurationError("Firebase Admin credentials must be a JSON object.")
        return value
    raw_file = _setting("FIREBASE_ADMIN_CREDENTIALS_FILE")
    if raw_file:
        path = Path(raw_file).expanduser()
        if not path.is_file():
            raise FirebaseConfigurationError(f"Firebase Admin credential file not found: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FirebaseConfigurationError("Firebase Admin credential file is not valid JSON.") from exc
        if not isinstance(value, dict):
            raise FirebaseConfigurationError("Firebase Admin credentials must be a JSON object.")
        return value
    keys = ("type", "project_id", "private_key", "client_email", "token_uri")
    value = {key: _setting(f"FIREBASE_ADMIN_{key.upper()}") for key in keys}
    if not all(value[key] for key in ("type", "project_id", "private_key", "client_email")):
        raise FirebaseConfigurationError("Firebase Admin service-account credentials are not configured. Set FIREBASE_ADMIN_CREDENTIALS_FILE or FIREBASE_ADMIN_CREDENTIALS_JSON.")
    value["token_uri"] = value["token_uri"] or "https://oauth2.googleapis.com/token"
    return value


def admin_credentials_configured() -> bool:
    if _setting("FIREBASE_ADMIN_CREDENTIALS_JSON"):
        return True
    raw_file = _setting("FIREBASE_ADMIN_CREDENTIALS_FILE")
    if raw_file and Path(raw_file).expanduser().is_file():
        return True
    return all(_setting(f"FIREBASE_ADMIN_{key}") for key in ("TYPE", "PROJECT_ID", "PRIVATE_KEY", "CLIENT_EMAIL"))


def _firebase_admin_verifier(token: str) -> Mapping[str, Any]:
    try:
        import firebase_admin
        from firebase_admin import auth, credentials
    except ImportError as exc:
        raise FirebaseConfigurationError("firebase-admin is not installed.") from exc
    try:
        app = firebase_admin.get_app()
    except ValueError:
        try:
            app = firebase_admin.initialize_app(credentials.Certificate(_admin_credentials()))
        except FirebaseConfigurationError:
            raise
        except Exception as exc:
            raise FirebaseConfigurationError("Firebase Admin could not be initialized.") from exc
    try:
        return auth.verify_id_token(token, app=app, check_revoked=True)
    except Exception as exc:
        raise InvalidFirebaseToken("The Firebase ID token is expired, invalid, or revoked.") from exc


def verify_id_token(token: str | None, verifier: Callable[[str], Mapping[str, Any]] | None = None) -> dict[str, str]:
    if not token or not token.strip():
        raise InvalidFirebaseToken("A Firebase ID token is required.")
    try:
        claims = (verifier or _firebase_admin_verifier)(token)
    except (FirebaseConfigurationError, InvalidFirebaseToken):
        raise
    except Exception as exc:
        raise InvalidFirebaseToken("The Firebase ID token is expired, invalid, or revoked.") from exc
    uid = str(claims.get("uid", "")).strip()
    if not uid:
        raise InvalidFirebaseToken("The verified Firebase token has no UID.")
    return {key: str(claims.get(key, "")).strip() for key in ("uid", "name", "picture", "email")}
