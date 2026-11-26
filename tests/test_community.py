"""Behavioral tests for durable Community storage and content redaction."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.community_filter import redact_profanity, validate_submission
from modules.community_store import CommunityAuthorizationError, CommunityStore
from modules.firebase_auth import InvalidFirebaseToken, firebase_web_config, verify_id_token


class CommunityFilterTests(unittest.TestCase):
    def test_redacts_case_and_punctuation_variants_without_substrings(self) -> None:
        with patch.dict(os.environ, {"KAIROS_PROFANITY_WORDS": "badword,ass"}):
            result = redact_profanity("BAD-WORD and a.s.s, but class and badass remain.")
        self.assertEqual(result, "*** and ***, but class and badass remain.")

    def test_rejects_empty_submission(self) -> None:
        self.assertIsNotNone(validate_submission("  ", "Answer"))
        self.assertEqual(redact_profanity("  "), "")


class FirebaseAuthTests(unittest.TestCase):
    def test_valid_and_invalid_token_verification(self) -> None:
        self.assertEqual(
            verify_id_token("valid-token", verifier=lambda _: {"uid": "firebase-123", "name": "A User"})["uid"],
            "firebase-123",
        )
        with self.assertRaises(InvalidFirebaseToken):
            verify_id_token("expired-token", verifier=lambda _: (_ for _ in ()).throw(ValueError("expired")))

    def test_missing_token_is_rejected(self) -> None:
        with self.assertRaises(InvalidFirebaseToken):
            verify_id_token(None, verifier=lambda _: {"uid": "not-used"})

    def test_web_configuration_is_loaded_from_environment(self) -> None:
        values = {
            "FIREBASE_API_KEY": "web-key",
            "FIREBASE_AUTH_DOMAIN": "example.firebaseapp.com",
            "FIREBASE_PROJECT_ID": "example",
            "FIREBASE_STORAGE_BUCKET": "example.firebasestorage.app",
            "FIREBASE_MESSAGING_SENDER_ID": "123",
            "FIREBASE_APP_ID": "app-id",
        }
        with patch.dict(os.environ, values, clear=True):
            config, missing = firebase_web_config()
        self.assertEqual(missing, [])
        self.assertEqual(config["projectId"], "example")


class CommunityStoreTests(unittest.TestCase):
    def test_questions_answers_likes_sort_and_persist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "community.db"
            store = CommunityStore(path)
            first = store.create_question("alice", "First", "Details")
            second = store.create_question("bob", "Second", "Details")
            store.create_answer(first, "bob", "Try a bearing inspection.")
            self.assertTrue(store.toggle_like(second, "alice"))
            self.assertFalse(store.toggle_like(second, "alice"))
            self.assertTrue(store.toggle_like(first, "charlie"))
            self.assertTrue(store.toggle_like(first, "bob"))
            questions = store.list_questions()
            self.assertEqual(questions[0]["question_id"], first)
            self.assertEqual(questions[0]["like_count"], 2)
            self.assertEqual(questions[0]["answer_count"], 1)
            store.close()

            reopened = CommunityStore(path)
            self.assertEqual(len(reopened.list_questions()), 2)
            self.assertEqual(len(reopened.list_answers(first)), 1)
            self.assertEqual(reopened.get_question(first, "bob")["liked"], 1)
            reopened.close()

    def test_self_like_and_unauthenticated_writes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CommunityStore(Path(directory) / "community.db")
            question_id = store.create_question("owner", "Owned", "Details")
            with self.assertRaises(CommunityAuthorizationError):
                store.toggle_like(question_id, "owner")
            with self.assertRaises(CommunityAuthorizationError):
                store.create_answer(question_id, "", "Not allowed")
            with self.assertRaises(CommunityAuthorizationError):
                store.create_question("", "No identity", "Not allowed")
            store.close()

    def test_delete_is_owner_protected_and_removes_children(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CommunityStore(Path(directory) / "community.db")
            question_id = store.create_question("owner", "Owned", "Details")
            store.create_answer(question_id, "another-user", "An answer")
            store.toggle_like(question_id, "another-user")

            self.assertFalse(store.delete_question(question_id, "another-user"))
            self.assertIsNotNone(store.get_question(question_id, "another-user"))
            self.assertEqual(len(store.list_answers(question_id)), 1)

            self.assertTrue(store.delete_question(question_id, "owner"))
            self.assertIsNone(store.get_question(question_id, "owner"))
            self.assertEqual(store.list_answers(question_id), [])
            store.close()


if __name__ == "__main__":
    unittest.main()
