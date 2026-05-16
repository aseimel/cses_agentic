import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class StataMCPOnlyTests(unittest.TestCase):
    def test_no_direct_stata_batch_execution(self):
        """All production Stata execution must route through MCPStataRunner."""
        scanned_suffixes = {".py", ".ps1", ".sh"}
        blocked_patterns = [
            re.compile(r"\[\s*stata_path\s*,\s*['\"]-b['\"]\s*,\s*['\"]do['\"]", re.IGNORECASE),
            re.compile(r"Start-Process\s+.*Stata", re.IGNORECASE),
            re.compile(r"os\.system\s*\(.*Stata", re.IGNORECASE),
        ]
        violations: list[str] = []
        for path in REPO_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in scanned_suffixes:
                continue
            relative = path.relative_to(REPO_ROOT)
            if any(part in {".git", ".venv", "__pycache__", "build", "dist"} for part in relative.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in blocked_patterns:
                if pattern.search(text):
                    violations.append(str(relative))
                    break
        self.assertEqual([], violations)

    def test_benchmark_has_no_local_stata_path_default(self):
        benchmark = REPO_ROOT / "scripts" / "benchmark_sweden_replication.py"
        text = benchmark.read_text(encoding="utf-8")
        self.assertNotIn("Stata-SE-19", text)
        self.assertNotIn("StataSE-64.exe", text)


if __name__ == "__main__":
    unittest.main()
