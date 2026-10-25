"""gear_wbc ONNX policies -> the X280 weight blob (PLAN.md WP-3.1, §1.4.4; format in x280/fw/wbc_mlp.h).

The blob holds both policies (fp16 weights) or one (fp32 weights: two fp32 policies need 3.76 MB,
twice the LIM, so DP4 puts one per tile). Weights are stored transposed, W^T[in][out], so the
X280's column-broadcast GEMV streams one contiguous row per input; biases stay fp32; every array is
64 B aligned so the host can write it with full WC lines. The header carries a CRC-32 (zlib) over
everything after its first 20 bytes, and the CRC-32 of each source .onnx file for provenance.

Before packing, the graph is checked against the architecture the C kernel hard-codes (op
sequence via rtserve.wbc_numpy_ref, then every Gemm/Elu/Slice/ReduceL2/Clip/Concat attribute and
constant), so a re-exported policy with a different shape fails here instead of on the hart.

  simpy $RT/x280/tools/pack_weights.py --dtype fp16 --out $RT/x280/fw/build/wbc_fp16.blob
  simpy $RT/x280/tools/pack_weights.py --dtype fp32 --policies balance --out $RT/x280/fw/build/wbc_fp32_balance.blob
  simpy $RT/x280/tools/pack_weights.py --describe
(simpy runs from $R, so give absolute paths.) Measured: fp16 Balance+Walk = 1,889,088 B, ends
0x081D_E340 at 0x0801_1000 (7,360 B spare); fp32 one policy = 1,882,688 B, ends 0x081D_CA40.
"""

import argparse
import os
from pathlib import Path
import struct
import sys
import zlib

import numpy as np

from rtserve import wbc_numpy_ref

# --- Format constants (mirror x280/fw/wbc_mlp.h) ------------------------------------------------

MAGIC = 0x42434257          # "WBCB"
VERSION = 1
HDR_BYTES = 320
CRC_FROM = 20
ALIGN = 64
LAYOUT_WT = 1
WDTYPE = {"fp32": 0, "fp16": 1}
NP_DTYPE = {"fp32": np.dtype("<f4"), "fp16": np.dtype("<f2")}
BLOB_BASE = 0x0801_1000     # PLAN.md §1.4.4 weights line
LIM_END = 0x081E_0000       # 15 x 128 KiB from 0x0800_0000 at reset (tt-isa-documentation L2CPUTile/Caches.md:73, 92)

POLICIES = ("GR00T-WholeBodyControl-Balance", "GR00T-WholeBodyControl-Walk")  # index = WBC_POLICY_*
POLICY_ALIASES = {"balance": 0, "walk": 1}
LAYERS = ("estimator.0", "estimator.2", "estimator.4", "actor.0", "actor.2", "actor.4", "actor.6")
DIMS = ((516, 256), (256, 256), (256, 35), (121, 512), (512, 256), (256, 256), (256, 15))  # (in, out)
OBS_DIM, FRAME, N_VEL, N_EST = 516, 86, 3, 35
NORM_EPS = np.float32(1e-12)
INT64_MAX = 2**63 - 1

_HDR_HEAD = struct.Struct("<16I")              # magic .. reserved[4]
_LAYER = struct.Struct("<4I")                  # in, out, w_off, b_off
_TAG_BYTES = 32
assert _HDR_HEAD.size + 2 * len(LAYERS) * _LAYER.size + _TAG_BYTES == HDR_BYTES

ONNX_DIR = os.environ.get("ONNX") or str(
    Path.home() / "experiments/gr00t/Isaac-GR00T-n1.6/external_dependencies/GR00T-WholeBodyControl"
    / "gr00t_wbc/sim2mujoco/resources/robots/g1/policy")


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


# --- Graph check --------------------------------------------------------------------------------


def describe(model_path: str) -> list[str]:
    """One line per non-Constant node with its attributes and constant inputs resolved."""
    import onnx
    from onnx import helper, numpy_helper

    graph = onnx.load(str(model_path)).graph
    const = {}
    for n in graph.node:
        if n.op_type == "Constant":
            const[n.output[0]] = numpy_helper.to_array(n.attribute[0].t).tolist()
    inits = {t.name: tuple(t.dims) for t in graph.initializer}
    lines = []
    for n in graph.node:
        if n.op_type == "Constant":
            continue
        args = []
        for i in n.input:
            if i in const:
                args.append(repr(const[i]))
            elif i in inits:
                args.append(f"{i}{list(inits[i])}")
            else:
                args.append(i or "-")
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in sorted(attrs.items())}
        lines.append(f"{n.op_type}({', '.join(args)}) -> {n.output[0]} {attrs if attrs else ''}".rstrip())
    return lines


def check_graph(model_path: str) -> None:
    """Raise ValueError unless the graph is exactly the MLP wbc_mlp.c implements."""
    import onnx
    from onnx import helper, numpy_helper

    wbc_numpy_ref.load_initializers(model_path)  # op sequence (EXPECTED_OPS)
    graph = onnx.load(str(model_path)).graph
    const = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in graph.node if n.op_type == "Constant"}
    nodes = [n for n in graph.node if n.op_type != "Constant"]
    inits = {t.name: tuple(t.dims) for t in graph.initializer}
    errs = []

    def attrs(n):
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}

    gemms = [n for n in nodes if n.op_type == "Gemm"]
    for n, prefix, (i, o) in zip(gemms, LAYERS, DIMS):
        a = attrs(n)
        if (a.get("alpha", 1.0), a.get("beta", 1.0), a.get("transA", 0), a.get("transB", 0)) != (1.0, 1.0, 0, 1):
            errs.append(f"{prefix}: Gemm attrs {a}")
        if list(n.input[1:]) != [f"{prefix}.weight", f"{prefix}.bias"]:
            errs.append(f"{prefix}: Gemm inputs {list(n.input)}")
        if inits.get(f"{prefix}.weight") != (o, i) or inits.get(f"{prefix}.bias") != (o,):
            errs.append(f"{prefix}: shapes {inits.get(prefix + '.weight')} {inits.get(prefix + '.bias')}")
    for n in nodes:
        if n.op_type == "Elu" and attrs(n).get("alpha", 1.0) != 1.0:
            errs.append(f"Elu alpha {attrs(n)}")
    # Slices in graph order: est[:, 0:3], est[:, 3:], input[0:] (batch), frame[:, -86:].
    want = [(0, N_VEL, -1), (N_VEL, INT64_MAX, -1), (0, INT64_MAX, 0), (-FRAME, INT64_MAX, 1)]
    slices = [n for n in nodes if n.op_type == "Slice"]
    for n, (s, e, ax) in zip(slices, want):
        got = tuple(int(const[x].reshape(-1)[0]) for x in n.input[1:4])
        step = int(const[n.input[4]].reshape(-1)[0]) if len(n.input) > 4 else 1
        if got != (s, e, ax) or step != 1:
            errs.append(f"Slice {n.output[0]}: (start, end, axis, step) = {got + (step,)}")
    red = next(n for n in nodes if n.op_type == "ReduceL2")
    if list(attrs(red).get("axes", [])) != [-1] or attrs(red).get("keepdims", 1) != 1:
        errs.append(f"ReduceL2 attrs {attrs(red)}")
    clip = next(n for n in nodes if n.op_type == "Clip")
    if np.float32(const[clip.input[1]]) != NORM_EPS or (len(clip.input) > 2 and clip.input[2]):
        errs.append(f"Clip min {const.get(clip.input[1])} max {clip.input[2:]}")
    cat = next(n for n in nodes if n.op_type == "Concat")
    order = [x.split("/")[1] for x in cat.input]
    if attrs(cat).get("axis") != 1 or order != ["Slice_3_output_0", "Slice_output_0", "Div_output_0"]:
        errs.append(f"Concat axis {attrs(cat).get('axis')} inputs {list(cat.input)}")
    div = next(n for n in nodes if n.op_type == "Div")
    if div.input[0] != slices[1].output[0]:
        errs.append(f"Div numerator {div.input[0]}")
    if graph.input[0].type.tensor_type.shape.dim[1].dim_value != OBS_DIM:
        errs.append("input width")
    if errs:
        raise ValueError(f"{model_path}: graph differs from the wbc_mlp.c architecture: " + "; ".join(errs))


# --- Pack / parse -------------------------------------------------------------------------------


def parse_policies(spec: str | None) -> tuple[int, ...]:
    if not spec:
        return (0, 1)
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok in POLICY_ALIASES:
            out.append(POLICY_ALIASES[tok])
        elif tok in (p.lower() for p in POLICIES):
            out.append([p.lower() for p in POLICIES].index(tok))
        else:
            raise ValueError(f"unknown policy {tok!r} (balance, walk)")
    return tuple(sorted(set(out)))


def pack(onnx_dir: str = ONNX_DIR, dtype: str = "fp16", policies=(0, 1), tag: str | None = None) -> bytes:
    """The blob as bytes. Weights are cast from the ONNX float32 values (fp16: round to nearest even)."""
    if dtype not in WDTYPE:
        raise ValueError(f"dtype must be fp32 or fp16, got {dtype!r}")
    policies = tuple(sorted(set(policies)))
    if not policies or any(p not in (0, 1) for p in policies):
        raise ValueError(f"policies must be a non-empty subset of (0, 1), got {policies}")
    layer_tab = [[(0, 0, 0, 0)] * len(LAYERS) for _ in POLICIES]
    src_crc = [0, 0]
    body = bytearray()
    off = HDR_BYTES

    def put(arr: np.ndarray) -> int:
        nonlocal off
        start = _align(off)
        body.extend(b"\0" * (start - off))
        body.extend(arr.tobytes())
        off = start + arr.nbytes
        return start

    for p in policies:
        path = Path(onnx_dir) / f"{POLICIES[p]}.onnx"
        check_graph(str(path))
        src_crc[p] = zlib.crc32(path.read_bytes())
        w = wbc_numpy_ref.load_initializers(str(path))  # float64 copies of float32: exact
        for l, (prefix, (i, o)) in enumerate(zip(LAYERS, DIMS, strict=True)):
            W = w[f"{prefix}.weight"].astype(np.float32)
            b = w[f"{prefix}.bias"].astype(np.float32)
            assert W.shape == (o, i) and b.shape == (o,), (prefix, W.shape, b.shape)
            wt = np.ascontiguousarray(W.T).astype(NP_DTYPE[dtype])
            if not np.isfinite(wt).all():
                raise ValueError(f"{POLICIES[p]} {prefix}: weights overflow {dtype}")
            w_off = put(wt)
            b_off = put(b.astype("<f4"))
            layer_tab[p][l] = (i, o, w_off, b_off)
    total = _align(off)
    body.extend(b"\0" * (total - off))

    mask = sum(1 << p for p in policies)
    tag = (tag or f"pack_weights v{VERSION} {dtype}").encode()[:_TAG_BYTES - 1]
    hdr = bytearray(_HDR_HEAD.pack(MAGIC, VERSION, HDR_BYTES, total, 0, WDTYPE[dtype], LAYOUT_WT, mask,
                                   len(LAYERS), ALIGN, src_crc[0], src_crc[1], 0, 0, 0, 0))
    for p in range(len(POLICIES)):
        for row in layer_tab[p]:
            hdr += _LAYER.pack(*row)
    hdr += tag.ljust(_TAG_BYTES, b"\0")
    blob = bytearray(hdr + body)
    assert len(blob) == total
    struct.pack_into("<I", blob, 16, zlib.crc32(bytes(blob[CRC_FROM:])))
    return bytes(blob)


def parse(blob: bytes) -> dict:
    """Header fields plus per-policy [(W^T as float32 (in, out), bias (out,)), ...] (checks the CRC)."""
    head = _HDR_HEAD.unpack_from(blob, 0)
    (magic, version, hdr_bytes, total, crc, wdtype, layout, mask, n_layers, align) = head[:10]
    if magic != MAGIC or version != VERSION or hdr_bytes != HDR_BYTES:
        raise ValueError(f"bad header magic={magic:#x} version={version} hdr_bytes={hdr_bytes}")
    if total > len(blob) or zlib.crc32(bytes(blob[CRC_FROM:total])) != crc:
        raise ValueError("bad total_len or CRC")
    dt = {v: NP_DTYPE[k] for k, v in WDTYPE.items()}[wdtype]
    out = {"total_len": total, "crc32": crc, "wdtype": wdtype, "dtype": dt.name, "layout": layout,
           "policy_mask": mask, "n_layers": n_layers, "align": align, "src_crc32": head[10:12],
           "tag": bytes(blob[HDR_BYTES - _TAG_BYTES:HDR_BYTES]).rstrip(b"\0").decode(), "layers": {}}
    for p in range(len(POLICIES)):
        if not (mask >> p) & 1:
            continue
        layers = []
        for l in range(n_layers):
            i, o, w_off, b_off = _LAYER.unpack_from(blob, _HDR_HEAD.size + (p * len(LAYERS) + l) * _LAYER.size)
            wt = np.frombuffer(blob, dt, i * o, w_off).reshape(i, o).astype(np.float32)
            b = np.frombuffer(blob, "<f4", o, b_off).copy()
            layers.append((wt, b, w_off, b_off))
        out["layers"][POLICIES[p]] = layers
    return out


def end_address(blob_len: int, base: int = BLOB_BASE) -> int:
    return base + blob_len


# --- CLI ----------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--onnx-dir", default=ONNX_DIR)
    ap.add_argument("--dtype", choices=tuple(WDTYPE), default="fp16")
    ap.add_argument("--policies", default=None, help="balance,walk (default both)")
    ap.add_argument("--out", default=None, help="blob path (default: print the summary only)")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=BLOB_BASE, help="LIM load address")
    ap.add_argument("--host-only", action="store_true",
                    help="allow a blob that does not fit in LIM (x86 tests only)")
    ap.add_argument("--describe", action="store_true", help="print and check both ONNX graphs, then exit")
    args = ap.parse_args(argv)

    if args.describe:
        for name in POLICIES:
            path = Path(args.onnx_dir) / f"{name}.onnx"
            print(f"== {path}")
            print("\n".join(describe(str(path))))
            check_graph(str(path))
            print("check: OK (estimator 516-256-256-35 ELU, L2-normalised est[3:35], "
                  "actor cat(obs[430:516], est[0:3], latent) 121-512-256-256-15 ELU)")
        return 0

    policies = parse_policies(args.policies)
    blob = pack(args.onnx_dir, args.dtype, policies)
    end = end_address(len(blob), args.base)
    names = "+".join(POLICIES[p].rsplit("-", 1)[1] for p in policies)
    crc = struct.unpack_from("<I", blob, 16)[0]
    print(f"blob {len(blob)} B ({args.dtype}, {names}), crc32 {crc:#010x}; at {args.base:#010x} it ends "
          f"{end:#010x} (LIM end {LIM_END:#010x}, spare {LIM_END - end} B)")
    if end > LIM_END:
        msg = (f"blob ends {end:#010x} > LIM end {LIM_END:#010x} by {end - LIM_END} B"
               + ("; fp32 fits one policy per tile (DP4): --policies balance|walk" if args.dtype == "fp32" else ""))
        if not args.host_only:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 2
        print(f"HOST-ONLY: {msg}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes(blob)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
