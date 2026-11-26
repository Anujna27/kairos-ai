"""Streamlit Firebase UI compatibility layer."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from .firebase_core import (  # noqa: F401
    FirebaseConfigurationError,
    InvalidFirebaseToken,
    _WEB_ENV_KEYS,
    admin_credentials_configured,
    firebase_web_config,
    verify_id_token,
)

_firebase_auth_component = None


def _auth_component():
    global _firebase_auth_component
    if _firebase_auth_component is None:
        _firebase_auth_component = components.declare_component(
            "kairos_firebase_auth", path=str(Path(__file__).resolve().parents[1] / "web" / "firebase_auth")
        )
    return _firebase_auth_component


def require_authenticated_user() -> dict[str, str] | None:
    """Render the Streamlit auth component and return a verified user."""
    web_config, missing = firebase_web_config()
    if missing:
        st.error("KAIROS-AI is unavailable because Firebase Web configuration is missing.")
        st.caption(f"Missing environment variables: {', '.join(_WEB_ENV_KEYS[key] for key in missing)}")
        return None
    if not admin_credentials_configured():
        st.error("KAIROS-AI is unavailable because Firebase Admin credentials are missing.")
        return None
    payload = _auth_component()(config=web_config, key="global_firebase_auth")
    if isinstance(payload, dict) and payload.get("status") == "signed_out":
        st.session_state.pop("authenticated_user", None)
        st.session_state.pop("firebase_auth_token", None)
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
