"""CuPy backend metadata for the MVP."""

from simforge_gpu.backends.base import BackendInfo

SUPPORTED_NUMPY_APIS = (
    "np.array",
    "np.zeros",
    "np.ones",
    "np.arange",
    "np.linspace",
    "np.mean",
    "np.var",
    "np.std",
    "np.sum",
    "np.sqrt",
    "np.exp",
    "np.log",
    "np.maximum",
    "np.minimum",
    "np.random.normal",
    "np.random.uniform",
    "np.random.binomial",
)

SUPPORTED_PATTERNS = (
    "monte_carlo_independent_trials",
    "simple_bootstrap_mean",
    "random_walk",
    "normal_mean_probability",
    "permutation_test",
)

CUPY_BACKEND = BackendInfo(
    name="cupy",
    display_name="CuPyBackend",
    status="implemented",
    reason="MVP backend for NumPy-style simulation code.",
    supported_patterns=SUPPORTED_PATTERNS,
    supported_numpy_apis=SUPPORTED_NUMPY_APIS,
)
