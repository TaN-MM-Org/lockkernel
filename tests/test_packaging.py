"""1.5.1: the installed package lists both license files.

The Apache License 2.0 asks that a NOTICE file shipped with the work be
passed on with it. From 1.4.0 to 1.5.0 `license-files` named only
LICENSE, so NOTICE was missing from the sdist and the wheel.
"""
import importlib.metadata as md


def test_license_and_notice_are_listed_in_the_metadata():
    files = md.metadata("lockkernel").get_all("License-File") or []
    assert "LICENSE" in files
    assert "NOTICE" in files
