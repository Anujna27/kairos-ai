"""Firebase Web configuration and server-side ID-token verification."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # Deployment environments may inject variables directly.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
else:
    _local_env = Path(__file__).resolve().parents[1] / ".env"
    if _local_env.is_file():
        for _line in _local_env.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _name, _value = _line.split("=", 1)
            _value = _value.strip().strip("\"'")
            os.environ.setdefault(_name.strip(), _value)

import streamlit as st
import streamlit.components.v1 as components


class FirebaseConfigurationError(RuntimeError):
    """Raised when Firebase configuration is incomplete or unavailable."""


class InvalidFirebaseToken(PermissionError):
    """Raised when a browser-supplied Firebase ID token cannot be verified."""


_WEB_ENV_KEYS = {
    "apiKey": "FIREBASE_API_KEY",
    "authDomain": "FIREBASE_AUTH_DOMAIN",
    "projectId": "FIREBASE_PROJECT_ID",
    "storageBucket": "FIREBASE_STORAGE_BUCKET",
    "messagingSenderId": "FIREBASE_MESSAGING_SENDER_ID",
    "appId": "FIREBASE_APP_ID",
}


_firebase_auth_component = None


def _auth_component():
    global _firebase_auth_component
    if _firebase_auth_component is None:
        _firebase_auth_component = components.declare_component(
            "kairos_firebase_auth", path=str(Path(__file__).resolve().parents[1] / "web" / "firebase_auth")
        )
    return _firebase_auth_component


def _setting(env_name: str) -> str:
    return os.getenv(env_name, "").strip()


def firebase_web_config() -> tuple[dict[str, str] | None, list[str]]:
    config = {
        key: _setting(env_name)
        for key, env_name in _WEB_ENV_KEYS.items()
    }
    missing = [key for key, value in config.items() if not value]
    return (None, missing) if missing else (config, [])


def admin_credentials_configured() -> bool:
    if _setting("FIREBASE_ADMIN_CREDENTIALS_JSON"):
        return True
    credential_file = _setting("FIREBASE_ADMIN_CREDENTIALS_FILE")
    if credential_file and Path(credential_file).is_file():
        return True
    required = ("type", "project_id", "private_key", "client_email")
    return all(_setting(f"FIREBASE_ADMIN_{key.upper()}") for key in required)


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
    credential_file = _setting("FIREBASE_ADMIN_CREDENTIALS_FILE")
    if credential_file:
        path = Path(credential_file)
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
    value = {
        key: _setting(f"FIREBASE_ADMIN_{key.upper()}")
        for key in keys
    }
    if not all(value[key] for key in ("type", "project_id", "private_key", "client_email")):
        raise FirebaseConfigurationError("Firebase Admin service-account credentials are not configured.")
    value["token_uri"] = value["token_uri"] or "https://oauth2.googleapis.com/token"
    return value


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
    """Verify a Firebase ID token and return only trusted identity fields.

    ``verifier`` is injectable for tests; production callers use Firebase Admin.
    """
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
    return {
        "uid": uid,
        "name": str(claims.get("name", "")).strip(),
        "picture": str(claims.get("picture", "")).strip(),
        "email": str(claims.get("email", "")).strip(),
    }


def require_authenticated_user() -> dict[str, str] | None:
    """Render the global auth gate and return a server-verified user, if any."""
    st.markdown(
        """
        <style>
        .stApp { background:#07111f; color:#e8f1f7; }
        .auth-shell { max-width:760px; margin:12vh auto 0; text-align:center; }
        .auth-shell h1 { letter-spacing:.08em; color:#43d9e8; }
        .auth-shell p { color:#8ca5b8; }
        </style>
        <div class="auth-shell"><h1>KAIROS-AI</h1><p>Acoustic Intelligence for Predictive Machine Health Monitoring</p></div>
        """,
        unsafe_allow_html=True,
    )
    web_config, missing = firebase_web_config()
    if missing:
        st.error("KAIROS-AI is unavailable because Firebase Web configuration is missing.")
        st.caption(f"Missing environment variables: {', '.join(_WEB_ENV_KEYS[key] for key in missing)}")
        return None
    if not admin_credentials_configured():
        st.error("KAIROS-AI is unavailable because Firebase Admin credentials are missing.")
        st.caption("Configure FIREBASE_ADMIN_CREDENTIALS_JSON or the documented FIREBASE_ADMIN_* variables before starting the app.")
        return None

    payload = _auth_component()(config=web_config, key="global_firebase_auth")
    if isinstance(payload, dict) and payload.get("status") == "signed_out":
        st.session_state.pop("authenticated_user", None)
        st.session_state.pop("firebase_auth_token", None)
        return None
    if isinstance(payload, dict) and payload.get("status") == "error":
        st.error("Firebase could not initialize. Check the Firebase Web configuration.")
        return None

    token = payload.get("id_token") if isinstance(payload, dict) else None
    token = token or st.session_state.get("firebase_auth_token")
    if not token:
        return None
    try:
        user = verify_id_token(token)
    except (InvalidFirebaseToken, FirebaseConfigurationError) as exc:
        st.session_state.pop("authenticated_user", None)
        st.session_state.pop("firebase_auth_token", None)
        st.error(f"Authentication could not be verified: {exc}")
        return None
    st.session_state.firebase_auth_token = token
    st.session_state.authenticated_user = user
    return user
