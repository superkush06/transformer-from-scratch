"""Version, changelog and the one count the README is easiest to get wrong.

Three failure modes, all of them silent, all of them cheap to close:

* a version bumped in ``pyproject.toml`` and not in ``tfs.__init__``, so
  the installed package reports something no tag matches;
* a release with no changelog section, which is how a changelog stops
  being release discipline and becomes a habit;
* a README that says the suite is *N* tests when it is not — the number
  that matters here, because "1,312" in this repository is a count of
  gradient checks and reads like a test count if nothing keeps the real
  one honest.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import tfs

ROOT = Path(__file__).resolve().parents[1]

# "84 tests" anywhere in the README — the Install snippet and the note
# under the head-line gradient count both quote it.
_TEST_COUNT = re.compile(r"\b(\d+) tests\b")
_COLLECTED = re.compile(r"^(\d+) tests collected", re.M)


def test_version_is_the_same_string_everywhere():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert tfs.__version__ == pyproject["project"]["version"]


def test_changelog_has_a_section_for_the_current_version():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert f"## [{tfs.__version__}]" in changelog


def test_readme_test_count_is_what_pytest_collects():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    match = _COLLECTED.search(proc.stdout)
    assert match, proc.stdout[-2000:]
    collected = int(match.group(1))
    quoted = {int(n) for n in _TEST_COUNT.findall((ROOT / "README.md").read_text())}
    assert quoted == {collected}, f"README says {sorted(quoted)}, pytest collects {collected}"
