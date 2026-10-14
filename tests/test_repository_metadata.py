import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class RepositoryMetadataTest(unittest.TestCase):
    def test_citation_is_valid_and_has_no_placeholders(self):
        text = (ROOT / "CITATION.cff").read_text()
        citation = yaml.safe_load(text)
        self.assertEqual(citation["cff-version"], "1.2.0")
        self.assertEqual(citation["authors"][0]["family-names"], "Jacquesson")
        self.assertNotIn("REPLACE_WITH", text)

    def test_license_has_named_holder(self):
        text = (ROOT / "LICENSE").read_text()
        self.assertIn("Alex Jacquesson", text)
        self.assertNotIn("REPLACE_WITH", text)

    def test_local_readme_links_exist(self):
        readme = (ROOT / "README.md").read_text()
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
        local = [link.split("#", 1)[0] for link in links if "://" not in link]
        missing = [link for link in local if not (ROOT / link).exists()]
        self.assertEqual(missing, [])

    def test_ai_declaration_matches_repository_policy(self):
        text = (ROOT / "AI_USAGE.md").read_text()
        expected = {
            "Asking the research question": "No",
            "Coming up with the approach": "No",
            "Proof development": "No",
            "Writing and exposition": "No",
            "Code development": "Yes",
            "Checking for bugs": "Yes",
            "Other supporting tasks": "Yes",
        }
        for activity, answer in expected.items():
            self.assertRegex(text, rf"\| {re.escape(activity)} \| {answer} \|")


if __name__ == "__main__":
    unittest.main()
