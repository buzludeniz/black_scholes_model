"""Packaging and version-consistency invariants.

These guard the facts that a passing test suite cannot see. A wrong version
string, a renamed distribution, or a mistyped console entry point all leave
every unit test green while breaking `pip install`.

``pyproject.toml`` is the single source of truth for the version and the
distribution name. ``black_scholes.__version__`` is a second copy of that value,
which is exactly the kind of duplication that silently drifts, so it is pinned
to the declaration here.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest

import black_scholes

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _declared(field: str) -> str:
    """Read a top-level string field from pyproject.toml.

    A regex rather than ``tomllib`` because the project supports Python 3.10,
    where ``tomllib`` does not exist, and because the release workflow already
    parses this file the same way. Two code paths to read one number would be
    a worse problem than the one being solved here.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(rf'^{field}\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, f"no {field} declared in pyproject.toml"
    return match.group(1)


class TestVersionConsistency:
    """One version, declared once."""

    def test_pyproject_declares_a_version(self):
        assert _declared("version") == black_scholes.__version__

    def test_in_package_version_is_not_empty(self):
        assert black_scholes.__version__.strip(), "__version__ must not be blank"
        assert re.fullmatch(r"\d+\.\d+\.\d+", black_scholes.__version__), (
            f"expected a PEP 440 release version, got {black_scholes.__version__!r}"
        )

    def test_installed_metadata_matches(self):
        """The version PyPI would index is the one importlib reports."""
        try:
            dist = installed_version("black_scholes_model")
        except PackageNotFoundError:
            pytest.skip("distribution is not installed in this environment")
        assert dist == black_scholes.__version__

    def test_version_is_exported(self):
        assert "__version__" in black_scholes.__all__


class TestDistributionName:
    """The distribution name and the import name are different things."""

    def test_distribution_name(self):
        assert _declared("name") == "black_scholes_model"

    def test_installed_metadata_uses_that_name(self):
        try:
            installed_version("black_scholes_model")
        except PackageNotFoundError:
            pytest.skip("distribution is not installed in this environment")

    def test_import_name_is_the_shorter_one(self):
        """The package is imported as `black_scholes`, distributed as above.

        The reverse would also work, so this pins the choice rather than
        requiring it.
        """
        assert black_scholes.__name__ == "black_scholes"

    def test_readme_install_line_uses_the_distribution_name(self):
        readme = PYPROJECT.with_name("README.md").read_text(encoding="utf-8")
        assert "pip install black_scholes_model" in readme
        assert "pip install black-scholes" not in readme


class TestEntryPoints:
    """Console scripts must be declared and must resolve."""

    def test_bs_console_script_is_declared(self):
        text = PYPROJECT.read_text(encoding="utf-8")
        assert 'bs = "black_scholes.cli:app"' in text

    def test_gui_script_is_declared(self):
        text = PYPROJECT.read_text(encoding="utf-8")
        assert 'bs-gui = "black_scholes.gui:run_gui"' in text

    def test_entry_point_targets_exist(self):
        """The declared callables must actually be importable."""
        from black_scholes.cli import app
        from black_scholes.gui import run_gui

        assert callable(app)
        assert callable(run_gui)

    def test_python_requires_at_least_310(self):
        assert _declared("requires-python") == ">=3.10"


class TestBuildArtifactsAreNotCommitted:
    """Generated output must not be tracked."""

    def test_gitignore_covers_build_output(self):
        ignored = PYPROJECT.with_name(".gitignore").read_text(encoding="utf-8").splitlines()
        cleaned = {line.strip() for line in ignored if line.strip() and not line.startswith("#")}
        for pattern in ("dist/", "build/", "*.egg-info/", ".smoke/", ".smoke-sdist/"):
            assert pattern in cleaned, f"{pattern} must be gitignored"
