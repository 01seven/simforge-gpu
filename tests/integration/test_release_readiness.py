from pathlib import Path


def test_release_files_exist():
    assert Path(".gitignore").exists()
    assert Path("LICENSE").exists()
    assert Path(".github/workflows/ci.yml").exists()
    assert Path("docs/RELEASE_CHECKLIST.md").exists()


def test_ci_uses_no_gpu_test_command():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "python -m pytest -m \"not gpu\" -q" in workflow
    assert "python -m pip install -e ." in workflow


def test_license_placeholder_removed_from_readme():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "License placeholder" not in readme
    assert "MIT License" in readme

