"""GPU backend for the GR00T-WholeBodyControl lower-body ONNX policies.

Rebuilds the exported ``estimator`` / ``actor`` MLPs as a torch module from the ONNX initializers
and replays them through a CUDA graph, so a policy call costs one H2D copy, one graph launch and
one D2H copy. ``run_gr00t_server.py --gear-wbc-onnx-dir`` serves them on the ``gear_wbc_action``
endpoint next to the VLA, so the simulator process never creates a CUDA context of its own (doing
so corrupts its EGL renders).
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn
import torch.nn.functional as F


# The estimator output starts with the base linear velocity estimate; the rest is a unit-norm
# latent.
NUM_VEL_ESTIMATE = 3


def _mlp(state_dict: dict, prefix: str) -> nn.Sequential:
    weights = sorted(
        (int(k.split(".")[1]), k)
        for k in state_dict
        if k.startswith(prefix) and k.endswith(".weight")
    )
    layers = []
    for i, (idx, key) in enumerate(weights):
        out_features, in_features = state_dict[key].shape
        layers.append(nn.Linear(in_features, out_features))
        layers[-1].weight.data.copy_(state_dict[key])
        layers[-1].bias.data.copy_(state_dict[f"{prefix}.{idx}.bias"])
        if i < len(weights) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class GearWbcMLP(nn.Module):
    """Torch port of ``GR00T-WholeBodyControl-{Balance,Walk}.onnx``."""

    def __init__(self, state_dict: dict):
        super().__init__()
        self.estimator = _mlp(state_dict, "estimator")
        self.actor = _mlp(state_dict, "actor")
        # The actor sees the newest single-step observation plus the estimator output.
        self.num_single_obs = self.actor[0].in_features - self.estimator[-1].out_features

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        est = self.estimator(obs_history)
        vel = est[:, :NUM_VEL_ESTIMATE]
        latent = F.normalize(est[:, NUM_VEL_ESTIMATE:], dim=-1, eps=1e-12)
        return self.actor(torch.cat([obs_history[:, -self.num_single_obs :], vel, latent], dim=1))


def load_onnx_state_dict(model_path: str) -> dict:
    graph = onnx.load(model_path).graph
    return {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in graph.initializer}


class CudaGraphPolicy:
    """Batch-1 policy replayed from a captured CUDA graph; takes and returns CPU tensors."""

    def __init__(self, module: GearWbcMLP, device: str):
        self.module = module.to(device).eval()
        self.num_obs = self.module.estimator[0].in_features
        self.static_in = torch.zeros(1, self.num_obs, device=device)
        self.host_in = torch.zeros(1, self.num_obs).pin_memory()
        with torch.no_grad():
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.module(self.static_in)
            torch.cuda.current_stream(device).wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = self.module(self.static_in)
        self.host_out = torch.zeros(self.static_out.shape).pin_memory()
        self.stream = torch.cuda.current_stream(device)

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.host_in.copy_(input_tensor)
        self.static_in.copy_(self.host_in, non_blocking=True)
        self.graph.replay()
        self.host_out.copy_(self.static_out, non_blocking=True)
        self.stream.synchronize()
        # The caller keeps the action across steps, so hand back a copy, not the pinned buffer.
        return self.host_out.clone()


class GearWbcPolicies:
    """Every ``*.onnx`` lower-body policy in a directory, keyed by file stem, as one endpoint."""

    def __init__(self, onnx_dir: str, device: str = "cuda"):
        paths = sorted(Path(onnx_dir).glob("*.onnx"))
        if not paths:
            raise FileNotFoundError(f"No .onnx policies in {onnx_dir}")
        self.policies = {
            p.stem: CudaGraphPolicy(GearWbcMLP(load_onnx_state_dict(str(p))), device) for p in paths
        }
        print(f"  GEAR WBC policies on {device}: {', '.join(self.policies)}")

    def get_action(self, policy: str, obs: np.ndarray) -> dict:
        return {"action": self.policies[policy](torch.from_numpy(obs)).numpy()}
