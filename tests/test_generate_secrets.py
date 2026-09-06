"""
Tests for scripts.generate_secrets placeholder handling.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import generate_secrets


class TestGenerateSecrets(unittest.TestCase):
    """
    Verify placeholder repair logic for local deployment secrets.
    """

    def test_replace_placeholder_runtime_secrets_keeps_db_password(self) -> None:
        """
        Example placeholder values for runtime auth secrets should be replaced,
        while an existing POSTGRES_PASSWORD entry is left untouched.
        """
        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        'POSTGRES_PASSWORD="your_strong_postgres_password_here"',
                        'MASTER_KEY="REPLACE_ME_with_a_freshly_generated_Fernet_key"',
                        "MASTER_KEY_VERSION=1",
                        'RESET_PASSWORD_TOKEN_SECRET="REPLACE_ME_with_a_random_32_byte_secret"',
                        'VERIFICATION_TOKEN_SECRET="REPLACE_ME_with_a_DIFFERENT_random_32_byte_secret"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            fresh_pairs = [
                ("POSTGRES_PASSWORD", "new-db-password"),
                ("MASTER_KEY", "fresh-master-key"),
                ("MASTER_KEY_VERSION", "1"),
                ("RESET_PASSWORD_TOKEN_SECRET", "fresh-reset-secret"),
                ("VERIFICATION_TOKEN_SECRET", "fresh-verification-secret"),
            ]

            replaced = generate_secrets._replace_placeholder_values(env_path, fresh_pairs)
            updated_text = env_path.read_text(encoding="utf-8")

            self.assertEqual(
                set(replaced),
                {"MASTER_KEY", "RESET_PASSWORD_TOKEN_SECRET", "VERIFICATION_TOKEN_SECRET"},
            )
            self.assertIn('POSTGRES_PASSWORD="your_strong_postgres_password_here"', updated_text)
            self.assertIn("MASTER_KEY=fresh-master-key", updated_text)
            self.assertIn("RESET_PASSWORD_TOKEN_SECRET=fresh-reset-secret", updated_text)
            self.assertIn(
                "VERIFICATION_TOKEN_SECRET=fresh-verification-secret",
                updated_text,
            )


if __name__ == "__main__":
    unittest.main()
