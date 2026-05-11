import json

from simforge_gpu.cli import main
from simforge_gpu.pipeline import collect_doctor_report


def test_collect_doctor_report_is_no_gpu_safe():
    report = collect_doctor_report()

    assert "SimForge GPU Doctor" in report
    assert "Python:" in report
    assert "Package import: OK" in report
    assert "cupy   MVP backend / implemented" in report
    assert "torch  planned" in report
    assert "CuPy import:" in report
    assert "No-GPU MVP commands remain available" in report


def test_doctor_cli_prints_environment_report(capsys):
    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "SimForge GPU Doctor" in captured.out
    assert "Package import: OK" in captured.out
    assert "No-GPU MVP commands remain available" in captured.out


def test_doctor_cli_can_print_json(capsys):
    exit_code = main(["doctor", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["package_import"] == "OK"
    assert payload["backends"]["cupy"] == "MVP backend / implemented"
    assert payload["backends"]["torch"] == "planned"
    assert payload["optional_gpu_dependencies"]["cuda_execution"] in {
        "available",
        "unavailable",
    }
    assert "cuda_execution_reason" in payload["optional_gpu_dependencies"]
    assert payload["workspace"]["examples"] in {"present", "missing"}
