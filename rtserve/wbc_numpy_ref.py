"""Pure-numpy float64 forward pass of the gear_wbc ONNX policies (PLAN.md §1.4.3, D3).

It needs no onnxruntime, so x280/tools/pack_weights.py reads the policies through it when it packs the X280
weight blobs. It reads only the ONNX initializers and re-implements the exported graph: estimator
516-256-256-35 (ELU), latent =
est[:, 3:] / max(||est[:, 3:]||, 1e-12), actor on cat(obs[:, -86:], est[:, :3], latent). The op
sequence of the file is asserted, so a re-exported graph with a different shape fails loudly
instead of drifting. tests/test_wbc_backend.py checks this against onnxruntime in the simpy venv.
"""

from pathlib import Path

import numpy as np

# Non-Constant ops of the exported graph, in order (torch.onnx export of GearWbcMLP.forward).
EXPECTED_OPS = (
    "Gemm", "Elu", "Gemm", "Elu", "Gemm",          # estimator
    "Slice", "Slice", "ReduceL2", "Clip", "Shape", "Expand", "Div",  # vel, unit latent
    "Slice", "Slice", "Concat",                    # newest frame, cat
    "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm",  # actor
)
NORM_EPS = 1e-12  # the Clip min in the graph (F.normalize eps)
NUM_VEL_ESTIMATE = 3


def load_initializers(model_path: str) -> dict[str, np.ndarray]:
    """Initializers as float64 arrays, after checking the graph is the MLP this module implements."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(model_path)).graph
    ops = tuple(n.op_type for n in graph.node if n.op_type != "Constant")
    if ops != EXPECTED_OPS:
        raise ValueError(f"{model_path}: unexpected op sequence {ops}")
    return {t.name: numpy_helper.to_array(t).astype(np.float64) for t in graph.initializer}


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


def _layers(weights: dict, prefix: str) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = sorted(int(k.split(".")[1]) for k in weights
                 if k.startswith(prefix + ".") and k.endswith(".weight"))
    return [(weights[f"{prefix}.{i}.weight"], weights[f"{prefix}.{i}.bias"]) for i in idx]


def _mlp(layers: list, x: np.ndarray) -> np.ndarray:
    for j, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if j < len(layers) - 1:
            x = _elu(x)
    return x


class NumpyWbc:
    """One gear_wbc policy as a float64 numpy function: (B, 516) -> (B, 15)."""

    def __init__(self, model_path: str):
        weights = load_initializers(model_path)
        self.estimator = _layers(weights, "estimator")
        self.actor = _layers(weights, "actor")
        self.num_obs = self.estimator[0][0].shape[1]
        n_est = self.estimator[-1][0].shape[0]
        self.num_single_obs = self.actor[0][0].shape[1] - n_est

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        assert x.ndim == 2 and x.shape[1] == self.num_obs, x.shape
        est = _mlp(self.estimator, x)
        vel = est[:, :NUM_VEL_ESTIMATE]
        lat = est[:, NUM_VEL_ESTIMATE:]
        lat = lat / np.maximum(np.linalg.norm(lat, axis=-1, keepdims=True), NORM_EPS)
        return _mlp(self.actor, np.concatenate([x[:, -self.num_single_obs:], vel, lat], axis=1))


def load_dir(onnx_dir: str) -> dict[str, NumpyWbc]:
    """Every ``*.onnx`` in a directory keyed by file stem, as GearWbcPolicies keys them."""
    paths = sorted(Path(onnx_dir).glob("*.onnx"))
    if not paths:
        raise FileNotFoundError(f"No .onnx policies in {onnx_dir}")
    return {p.stem: NumpyWbc(str(p)) for p in paths}
