"""Streamlit presentation and interactions for KAIROS Community."""

from __future__ import annotations

from html import escape

import streamlit as st

from modules.community_filter import redact_profanity, validate_submission
from modules.community_store import CommunityAuthorizationError, CommunityStore


def _author_label() -> str:
    """Use a friendly public label without exposing internal identity values."""
    return "Community member"


def _time(value: str) -> str:
    return value.replace("T", " ").replace("+00:00", " UTC")


def _preview(text: str, length: int = 180) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= length else f"{compact[:length].rstrip()}…"


def community() -> None:
    st.markdown(
        """
        <style>
        .community-kicker { color:#8ca5b8; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#20354d; background:#0d1b2d; }
        div[data-testid="stVerticalBlockBorderWrapper"] button[kind="tertiary"] { text-align:left; padding-left:0; font-size:1.22rem; font-weight:800; color:#e8f1f7; }
        div[data-testid="stVerticalBlockBorderWrapper"] button[kind="tertiary"]:hover { color:#43d9e8; }
        .community-preview { color:#c6d5df; font-size:.92rem; line-height:1.45; }
        div:has(> .community-fab-anchor) + div[data-testid="stButton"] button,
        div[data-testid="stButton"] button[aria-label="+"] { position:fixed; z-index:1000; right:2rem; bottom:2rem; width:3.25rem; height:3.25rem; min-height:3.25rem; border-radius:50%; padding:0; font-size:1.7rem; line-height:1; box-shadow:0 8px 24px rgba(0,0,0,.35); }
        div:has(> .community-fab-anchor) + div[data-testid="stButton"] button:hover,
        div[data-testid="stButton"] button[aria-label="+"]:hover { transform:translateY(-2px); }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("KAIROS Community")
    st.caption("Share acoustic troubleshooting knowledge with the machine-health community.")
    user = st.session_state.get("authenticated_user")
    if not user:
        return

    store = CommunityStore()
    try:
        st.session_state.setdefault("community_view", "list")
        st.session_state.setdefault("community_reply_id", None)
        st.session_state.setdefault("community_delete_id", None)
        if st.session_state.get("community_notice"):
            st.success(st.session_state.pop("community_notice"))

        if st.session_state.community_view == "ask":
            if st.button("← Back to questions", type="tertiary"):
                st.session_state.community_view = "list"
                st.rerun()
            st.subheader("Ask a question")
            with st.form("community_question_form", clear_on_submit=True):
                title = st.text_input("Question title", placeholder="e.g. What could cause a 3 kHz bearing signature?")
                description = st.text_area("Detailed description", height=180, placeholder="Include machine context, observed symptoms, and acoustic details…")
                if st.form_submit_button("Post Question", type="primary"):
                    title_error = validate_submission(title, "Question title")
                    description_error = validate_submission(description, "Description")
                    if title_error or description_error:
                        st.error(title_error or description_error)
                    else:
                        store.create_question(user["uid"], redact_profanity(title), redact_profanity(description))
                        st.session_state.community_view = "list"
                        st.session_state.community_selected = None
                        st.session_state.community_notice = "Question posted."
                        st.rerun()
            st.markdown("<div class='small'>Content is automatically redacted before it is saved.</div>", unsafe_allow_html=True)
        else:
            st.subheader("Questions")
            search = st.text_input("Search questions", placeholder="Search by title or description", label_visibility="collapsed")
            questions = store.list_questions(search)
            if not questions:
                st.info("No questions yet. Be the first to ask the community.")
            for question in questions:
                question_state = store.get_question(question["question_id"], user["uid"]) or question
                question_id = question["question_id"]
                is_owner = question["user_id"] == user["uid"]
                with st.container(border=True):
                    if st.button(question["title"], key=f"open_{question_id}", type="tertiary", use_container_width=True):
                        st.session_state.community_selected = question_id
                        st.session_state.community_reply_id = None
                        st.rerun()
                    st.markdown(f"<div class='community-preview'>{escape(_preview(question['description']))}</div>", unsafe_allow_html=True)
                    action_columns = st.columns([1, 1, .7] if is_owner else [1, 1], vertical_alignment="center")
                    action_like, action_reply = action_columns[:2]
                    with action_like:
                        liked = bool(question_state["liked"])
                        icon = ":material/thumb_up:" if liked else ":material/thumb_up_off_alt:"
                        label = "Unlike question" if liked else "Like question"
                        if is_owner:
                            st.button(str(question["like_count"]), icon=icon, help="Question authors cannot like their own questions", disabled=True, key=f"like_{question_id}")
                        elif st.button(str(question["like_count"]), icon=icon, help=label, type="primary" if liked else "secondary", key=f"like_{question_id}"):
                            try:
                                store.toggle_like(question_id, user["uid"])
                            except CommunityAuthorizationError as exc:
                                st.warning(str(exc))
                            else:
                                st.rerun()
                    with action_reply:
                        if st.button("Reply", icon=":material/reply:", type="secondary", key=f"reply_{question_id}"):
                            st.session_state.community_selected = None
                            st.session_state.community_reply_id = question_id
                            st.rerun()
                    if is_owner:
                        with action_columns[2]:
                            if st.button("", icon=":material/delete:", help="Delete question", type="tertiary", key=f"delete_{question_id}"):
                                st.session_state.community_delete_id = question_id
                                st.rerun()
                    if st.session_state.community_delete_id == question_id:
                        st.warning("Delete this question and all of its answers and likes?")
                        confirm_col, cancel_col = st.columns(2)
                        with confirm_col:
                            if st.button("Delete", type="primary", key=f"confirm_delete_{question_id}"):
                                if store.delete_question(question_id, user["uid"]):
                                    st.session_state.community_delete_id = None
                                    st.session_state.community_reply_id = None
                                    st.session_state.community_notice = "Question deleted."
                                    st.rerun()
                                st.error("This question could not be deleted because it is not owned by the current user.")
                        with cancel_col:
                            if st.button("Cancel", key=f"cancel_delete_{question_id}"):
                                st.session_state.community_delete_id = None
                                st.rerun()
                    if st.session_state.community_reply_id == question_id:
                        with st.form(f"inline_reply_form_{question_id}", clear_on_submit=True):
                            reply_text = st.text_area("Reply", height=90, placeholder="Share a useful diagnostic observation…", label_visibility="collapsed")
                            post_col, cancel_col = st.columns(2)
                            with post_col:
                                post_reply = st.form_submit_button("Post Reply", type="primary")
                            with cancel_col:
                                cancel_reply = st.form_submit_button("Cancel")
                            if cancel_reply:
                                st.session_state.community_reply_id = None
                                st.rerun()
                            if post_reply:
                                error = validate_submission(reply_text, "Reply")
                                if error:
                                    st.error(error)
                                else:
                                    cleaned_reply = redact_profanity(reply_text)
                                    store.create_answer(question_id, user["uid"], cleaned_reply)
                                    st.session_state.community_reply_id = None
                                    st.session_state.community_recent_reply = {"question_id": question_id, "text": cleaned_reply}
                                    st.rerun()
                    st.caption(f"{_author_label()} · {_time(question['created_at'])} · {question['answer_count']} answers")
                    recent_reply = st.session_state.get("community_recent_reply")
                    if recent_reply and recent_reply["question_id"] == question_id:
                        st.success("Reply posted")
                        st.write(recent_reply["text"])
                        st.session_state.community_recent_reply = None

            st.markdown("<div class='community-fab-anchor'></div>", unsafe_allow_html=True)
            if st.button("+", key="community_add_question", help="Ask a question", type="primary"):
                st.session_state.community_view = "ask"
                st.rerun()

        selected_id = st.session_state.get("community_selected")
        if selected_id and st.session_state.community_view == "list":
            question = store.get_question(selected_id, user["uid"])
            if question:
                st.divider()
                if st.button("← Back to questions", key="community_back_detail", type="tertiary"):
                    st.session_state.community_selected = None
                    st.rerun()
                st.subheader(question["title"])
                st.caption(f"Asked by {_author_label()} · {_time(question['created_at'])} · {question['like_count']} likes")
                st.write(question["description"])
                answers = store.list_answers(selected_id)
                st.subheader(f"Answers ({len(answers)})")
                if answers:
                    for answer in answers:
                        st.write(answer["answer_text"])
                        st.caption(f"by {_author_label()} · {_time(answer['created_at'])}")
                else:
                    st.info("No answers yet. Share what you know.")
                with st.form("community_answer_form", clear_on_submit=True):
                    answer_text = st.text_area("Your answer", height=130, placeholder="Share a useful diagnostic observation…")
                    if st.form_submit_button("Post Answer"):
                        error = validate_submission(answer_text, "Answer")
                        if error:
                            st.error(error)
                        else:
                            store.create_answer(selected_id, user["uid"], redact_profanity(answer_text))
                            st.success("Answer posted.")
                            st.rerun()
    except Exception as exc:
        st.error(f"Community storage is temporarily unavailable: {exc}")
    finally:
        store.close()
