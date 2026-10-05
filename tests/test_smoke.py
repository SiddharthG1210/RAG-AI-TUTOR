"""Smoke test: proves the project installs and pytest can import it.

A smoke test checks only that the basic setup works, before any real
feature exists. If this fails, the problem is in the setup (pyproject.toml,
the src/ layout, or the virtual environment), not in the tutor's code.
"""

import tutor


def test_package_imports() -> None:
    """The ``tutor`` package imports from the installed project.

    pytest collects every function whose name starts with ``test_`` and runs
    it. A plain ``assert`` is enough: when it fails, pytest shows the values
    on both sides of the comparison.
    """
    assert tutor.__name__ == "tutor"
