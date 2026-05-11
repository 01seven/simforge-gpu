from sim2gpu.analyzers.unsupported_detector import detect_unsupported


def test_detect_unsupported_flags_out_of_scope_code_and_backends():
    source = """
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import threading

data = pd.read_csv("x.csv")
with open("x.txt") as handle:
    text = handle.read()
eval("1 + 1")
exec("value = 3")
plt.plot([1, 2, 3])
"""

    unsupported = detect_unsupported(source, target_backend="torch")
    reasons = " ".join(item.reason for item in unsupported)
    codes = [item.code for item in unsupported]

    assert "--target torch" in codes
    assert "pandas" in reasons
    assert "Plotting" in reasons
    assert "File I/O" in reasons
    assert "eval" in reasons
    assert "exec" in reasons
    assert "multiprocessing" in reasons
    assert "threading" in reasons


def test_detect_unsupported_flags_classes_side_effects_and_sequential_loops():
    source = """
class Simulation:
    pass

state = 0
values = []
for i in range(10):
    state = state + i
    values.append(state)
"""

    unsupported = detect_unsupported(source)
    reasons = " ".join(item.reason for item in unsupported)
    codes = [item.code for item in unsupported]

    assert "class Simulation" in codes
    assert "values.append(...)" in codes
    assert "state = state + ..." in codes
    assert "class-heavy" in reasons
    assert "side-effect-heavy loop" in reasons
    assert "sequential dependency" in reasons


def test_detect_unsupported_assigns_standard_categories():
    source = """
import pandas as pd
import matplotlib.pyplot as plt

values = []
for i in range(3):
    values.append(i)

order = np.argsort(values)
"""

    unsupported = detect_unsupported(source, target_backend="torch")
    categories = {item.code: item.category for item in unsupported}

    assert categories["--target torch"] == "backend_not_implemented"
    assert categories["import pandas"] == "pandas_pipeline"
    assert categories["import matplotlib"] == "plotting"
    assert categories["values.append(...)"] == "side_effect_loop"
    assert categories["np.argsort"] == "unsupported_numpy_api"
