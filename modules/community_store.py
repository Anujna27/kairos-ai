"""Durable SQLite storage for KAIROS Community questions, answers, and likes."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CommunityAuthorizationError(PermissionError):
    """Raised when a Community operation violates identity or ownership rules."""


def _require_user_id(user_id: str) -> str:
    identity = str(user_id or "").strip()
    if not identity:
        raise CommunityAuthorizationError("An authenticated Firebase UID is required.")
    return identity


class CommunityStore:
    def __init__(self, path: str | Path = "data/kairos_community.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS questions (
                question_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS answers (
                answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
                user_id TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS likes (
                question_id INTEGER NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (question_id, user_id)
            );
            CREATE INDEX IF NOT EXISTS idx_questions_sort ON questions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id, created_at);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def create_question(self, user_id: str, title: str, description: str) -> int:
        user_id = _require_user_id(user_id)
        cursor = self.connection.execute(
            "INSERT INTO questions (user_id, title, description, created_at) VALUES (?, ?, ?, ?)",
            (user_id, title, description, utc_now()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def create_answer(self, question_id: int, user_id: str, answer_text: str) -> int:
        user_id = _require_user_id(user_id)
        cursor = self.connection.execute(
            "INSERT INTO answers (question_id, user_id, answer_text, created_at) VALUES (?, ?, ?, ?)",
            (question_id, user_id, answer_text, utc_now()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_questions(self, search: str = "") -> list[dict[str, Any]]:
        query = """
            SELECT q.*, COUNT(DISTINCT l.user_id) AS like_count,
                   COUNT(DISTINCT a.answer_id) AS answer_count
            FROM questions q
            LEFT JOIN likes l ON l.question_id = q.question_id
            LEFT JOIN answers a ON a.question_id = q.question_id
        """
        params: list[Any] = []
        if search.strip():
            query += " WHERE q.title LIKE ? OR q.description LIKE ?"
            term = f"%{search.strip()}%"
            params.extend([term, term])
        query += " GROUP BY q.question_id ORDER BY like_count DESC, q.created_at DESC, q.question_id DESC"
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def get_question(self, question_id: int, user_id: str = "") -> dict[str, Any] | None:
        row = self.connection.execute(
            """SELECT q.*, COUNT(DISTINCT l.user_id) AS like_count,
                      EXISTS(SELECT 1 FROM likes x WHERE x.question_id=q.question_id AND x.user_id=?) AS liked
               FROM questions q LEFT JOIN likes l ON l.question_id=q.question_id
               WHERE q.question_id=? GROUP BY q.question_id""",
            (user_id, question_id),
        ).fetchone()
        return dict(row) if row else None

    def list_answers(self, question_id: int) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM answers WHERE question_id=? ORDER BY created_at ASC, answer_id ASC", (question_id,)
        ).fetchall()]

    def toggle_like(self, question_id: int, user_id: str) -> bool:
        user_id = _require_user_id(user_id)
        question = self.connection.execute(
            "SELECT user_id FROM questions WHERE question_id=?", (question_id,)
        ).fetchone()
        if question is None:
            raise ValueError("Question does not exist.")
        if question["user_id"] == user_id:
            raise CommunityAuthorizationError("Question authors cannot like their own questions.")
        try:
            with self.connection:
                existing = self.connection.execute(
                    "SELECT 1 FROM likes WHERE question_id=? AND user_id=?", (question_id, user_id)
                ).fetchone()
                if existing:
                    self.connection.execute("DELETE FROM likes WHERE question_id=? AND user_id=?", (question_id, user_id))
                    return False
                self.connection.execute("INSERT INTO likes VALUES (?, ?, ?)", (question_id, user_id, utc_now()))
                return True
        except sqlite3.IntegrityError:
            # A concurrent request may have inserted the unique like first.
            # Treat that duplicate as already liked instead of surfacing a DB error.
            existing = self.connection.execute(
                "SELECT 1 FROM likes WHERE question_id=? AND user_id=?", (question_id, user_id)
            ).fetchone()
            if existing:
                return True
            raise

    def delete_question(self, question_id: int, user_id: str) -> bool:
        """Delete a question only when it belongs to ``user_id``.

        Child rows are removed explicitly so this remains safe for databases
        created before foreign-key enforcement was enabled.
        """
        user_id = _require_user_id(user_id)
        with self.connection:
            owned = self.connection.execute(
                "SELECT 1 FROM questions WHERE question_id=? AND user_id=?",
                (question_id, user_id),
            ).fetchone()
            if owned is None:
                return False
            self.connection.execute("DELETE FROM answers WHERE question_id=?", (question_id,))
            self.connection.execute("DELETE FROM likes WHERE question_id=?", (question_id,))
            deleted = self.connection.execute(
                "DELETE FROM questions WHERE question_id=? AND user_id=?",
                (question_id, user_id),
            )
            return deleted.rowcount == 1
