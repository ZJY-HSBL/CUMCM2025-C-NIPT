"""Lightweight archive tests; these are not numerical reproduction tests."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_four_original_scripts_parse():
    for q in range(1, 5):
        source = ROOT / "src" / "original" / f"question_{q}.py"
        assert source.exists()
        ast.parse(source.read_text(encoding="utf-8"))


def test_no_private_workbooks_are_in_repository():
    assert not list(ROOT.rglob("*.xlsx"))


def test_audit_records_unit_mismatch():
    audit = (ROOT / "docs" / "AUDIT.md").read_text(encoding="utf-8")
    assert "Z-score" in audit
    assert "问题二" in audit
