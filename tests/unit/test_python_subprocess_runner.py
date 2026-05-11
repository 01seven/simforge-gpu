from simforge_gpu.runners.python_subprocess import parse_script_output


def test_parse_script_output_prefers_json_array_from_last_line():
    parsed = parse_script_output("debug line\n[1.0, 2.5, 3]\n")

    assert parsed == [1.0, 2.5, 3]


def test_parse_script_output_falls_back_to_last_float():
    parsed = parse_script_output("estimate: 3.14159\n")

    assert parsed == 3.14159

