"""
Regression tests for docker-compose runtime compatibility.
"""

import re
import unittest
from pathlib import Path


class TestDockerCompose(unittest.TestCase):
    """
    Verify the Compose configuration avoids known Postgres runtime breakages.
    """

    def test_postgres_image_stays_on_existing_volume_compatible_major(self):
        """
        The tracked Docker setup uses a long-lived named volume created by
        the original Postgres 16 service. Major bumps need an explicit data
        migration plan instead of an automatic dependency update.
        """
        compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")
        image_match = re.search(
            r"^\s*image:\s*postgres:(\d+)-alpine\s*$", compose_text, re.MULTILINE
        )
        self.assertIsNotNone(image_match, "Expected a postgres alpine image in docker-compose.yml")
        self.assertEqual(
            int(image_match.group(1)),
            16,
            "docker-compose.yml should stay pinned to Postgres 16 until the volume upgrade path is implemented",
        )


if __name__ == "__main__":
    unittest.main()
