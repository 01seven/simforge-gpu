from simforge_gpu.analyzers.python_ast import analyze_source


def test_analyze_source_detects_numpy_loops_random_calls_and_outputs():
    source = """
import numpy as np

n = 1000
samples = []
for i in range(n):
    draw = np.random.normal()
    samples.append(draw)
estimate = np.mean(samples)
"""

    facts = analyze_source(source)

    assert facts.language == "python"
    assert facts.imports == ("numpy",)
    assert facts.numpy_aliases == ("np",)
    assert facts.random_calls == ("np.random.normal",)
    assert facts.outputs[-1] == "estimate"
    assert facts.for_loops[0]["type"] == "for_loop"
    assert facts.for_loops[0]["line_start"] == 6
