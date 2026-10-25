"""WP-2.1b: write a BF16 copy of the cloudwalk G1 checkpoint as its own hub repo for the TT port (PLAN.md D11).

Why: ``cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate@2a5b61cc`` stores 517 of its 1106 tensors as F32 (DiT,
LLM layers 12-15, encoders, decoder, vlln; index total_size 9,813,153,664 B). The port reads tensors as stored
(``MK/models/experimental/gr00t/common/weights.py:1076-1078``, no cast), so its host plan would carry fp32 where it
assumes bf16 (``weights.py:1765-1768``) and double its cache bytes (U:tt "Cloudwalk checkpoint dtype mismatch").
The PyTorch reference (Gr00tPolicy, the goldens' bf16 run) casts the whole model to bf16 at load
(``R/gr00t/policy/gr00t_policy.py:82-84``), so a copy rounded once with torch's round-to-nearest-even
``.to(torch.bfloat16)`` is faithful to what it computes.

What is written (never into ``~/.cache/huggingface``)::

    <out-hub>/models--cloudwalk-research--GR00T-N1.6-G1-PnPAppleToPlate-bf16/
        refs/main                      -> BF16_SHA (synthetic, :func:`synthetic_sha`)
        snapshots/<BF16_SHA>/          config.json embodiment_id.json processor_config.json statistics.json
                                       (byte copies), model.safetensors.index.json (same 2-shard weight_map,
                                       total_size 6,573,217,664), model-0000{1,2}-of-00002.safetensors
        conversion.json                timings, sha256 of every output file, the checks that passed

Every key stays in the shard the source put it in, in the same data order; BF16 tensors are copied byte for
byte, F32 tensors are rounded. Checks: source shard sha256 == its LFS blob name, 1106 keys, every output dtype
BF16, per-shard ``8 + header + data == file size``, sum == total_size == 6,573,217,664 (the base N1.6 byte count,
``MK/.../common/configs.py:830``), and (``--verify full``) every output tensor re-read through ``safetensors``
equals ``source.to(bf16)``. The snapshot dir appears atomically (rename) only after all checks pass.

    nice -n 19 taskset -c 16,17,20-31 ttpy -m rttt.convert_bf16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np

SRC_REPO = "cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate"
SRC_SHA = "2a5b61cc7660bdef4cf17ad18dc5dd97f13e4ce5"  # DP1 revision (refs/main of the local snapshot)
BF16_REPO = SRC_REPO + "-bf16"
CONVERTER = "rttt.convert_bf16 v1 (torch .to(bfloat16), round-to-nearest-even)"
EXPECT_N_TENSORS = 1106
EXPECT_N_F32 = 517
EXPECT_SRC_BYTES = 9_813_153_664
EXPECT_BF16_BYTES = 6_573_217_664  # == configs.N16_CONFIG.checkpoint.n_bytes (MK common/configs.py:830)
MIN_FREE_BYTES = 40 * 10**9  # PLAN.md WP-2.1b preflight
NON_WEIGHT_FILES = ("config.json", "embodiment_id.json", "processor_config.json", "statistics.json")
INDEX = "model.safetensors.index.json"
DEFAULT_SRC = Path.home() / ".cache/huggingface/hub" / ("models--" + SRC_REPO.replace("/", "--")) / "snapshots" / SRC_SHA
DEFAULT_OUT_HUB = Path.home() / ".cache/gr00t-tt-g1/hub"
FORBIDDEN_ROOT = Path.home() / ".cache/huggingface"


def synthetic_sha(src_repo: str = SRC_REPO, src_sha: str = SRC_SHA) -> str:
    """40-hex revision of the BF16 copy: a pure function of the source revision and the converter, so a rerun
    lands on the same snapshot dir and a changed converter can never alias an old copy."""
    return hashlib.sha1(f"{src_repo}@{src_sha}|{CONVERTER}".encode()).hexdigest()


BF16_SHA = synthetic_sha()


def repo_dir(hub: Path, repo: str = BF16_REPO) -> Path:
    return Path(hub) / ("models--" + repo.replace("/", "--"))


def snapshot_path(hub: Path = DEFAULT_OUT_HUB) -> Path:
    return repo_dir(hub) / "snapshots" / BF16_SHA


# --- safetensors framing (8-byte LE header length, JSON header, data; offsets relative to the data start) --------


def read_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def encode_header(entries: dict, meta: dict | None) -> bytes:
    """JSON header padded with spaces to a multiple of 8 bytes, as the safetensors writer aligns it."""
    h = ({"__metadata__": meta} if meta else {}) | entries
    b = json.dumps(h, separators=(",", ":")).encode()
    return b + b" " * (-len(b) % 8)


def sha256_file(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def check_shard_framing(path: Path) -> tuple[dict, int]:
    """(entries, data bytes) of one shard; raises unless offsets tile [0, data) exactly and the file ends there."""
    n, h = read_header(path)
    h.pop("__metadata__", None)
    spans = sorted((tuple(v["data_offsets"]), k) for k, v in h.items())
    pos = 0
    for (a, b), k in spans:
        if a != pos:
            raise ValueError(f"{path.name}: {k} starts at {a}, expected {pos} (hole or overlap)")
        pos = b
    if 8 + n + pos != path.stat().st_size:
        raise ValueError(f"{path.name}: 8 + header {n} + data {pos} != file size {path.stat().st_size}")
    return h, pos


# --- conversion -----------------------------------------------------------------------------------------------


def _bf16_bytes(raw: memoryview, shape: list) -> bytes:
    import torch

    a = np.frombuffer(raw, dtype="<f4").copy()  # owned, writable: torch.from_numpy needs it
    if not np.isfinite(a).all():
        raise ValueError("non-finite F32 value in the source")
    t = torch.from_numpy(a).reshape(shape).to(torch.bfloat16)
    return t.view(torch.int16).numpy().tobytes()


def convert_shard(src: Path, dst: Path) -> dict:
    """Write ``dst`` with every tensor of ``src`` in source data order, F32 rounded to BF16, BF16 copied."""
    n, h = read_header(src)
    meta = h.pop("__metadata__", None)
    order = sorted(h, key=lambda k: h[k]["data_offsets"][0])
    out, pos, counts = {}, 0, {"BF16": 0, "F32": 0}
    for k in order:
        e = h[k]
        if e["dtype"] not in counts:
            raise ValueError(f"{src.name}: {k} has dtype {e['dtype']}; only F32/BF16 are expected")
        counts[e["dtype"]] += 1
        nb = int(np.prod(e["shape"], dtype=np.int64)) * 2
        out[k] = {"dtype": "BF16", "shape": e["shape"], "data_offsets": [pos, pos + nb]}
        pos += nb
    base = 8 + n
    with open(src, "rb") as fs, mmap.mmap(fs.fileno(), 0, access=mmap.ACCESS_READ) as mm, open(dst, "wb") as fd:
        hb = encode_header(out, meta)
        fd.write(struct.pack("<Q", len(hb)))
        fd.write(hb)
        view = memoryview(mm)
        try:
            for k in order:
                a, b = h[k]["data_offsets"]
                raw = view[base + a: base + b]
                data = bytes(raw) if h[k]["dtype"] == "BF16" else _bf16_bytes(raw, h[k]["shape"])
                if len(data) != out[k]["data_offsets"][1] - out[k]["data_offsets"][0]:
                    raise AssertionError(f"{k}: wrote {len(data)} bytes")
                fd.write(data)
                del raw
        finally:
            view.release()
    return {"tensors": len(order), "src_dtypes": counts, "data_bytes": pos}


def verify_values(src_dir: Path, dst_dir: Path, keys: list[str]) -> int:
    """Re-read through ``safetensors`` (an independent parser) and compare with ``source.to(bf16)``."""
    import torch
    from safetensors import safe_open

    wm = json.loads((dst_dir / INDEX).read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k in keys:
        by_shard.setdefault(wm[k], []).append(k)
    n = 0
    for shard, ks in sorted(by_shard.items()):
        with safe_open(str(src_dir / shard), "pt") as fs, safe_open(str(dst_dir / shard), "pt") as fd:
            for k in ks:
                d = fd.get_tensor(k)
                s = fs.get_tensor(k)
                if d.dtype != torch.bfloat16 or not torch.equal(d, s.to(torch.bfloat16)):
                    raise AssertionError(f"{k}: output differs from source.to(bfloat16)")
                n += 1
    return n


def verify_output(src_dir: Path, dst_dir: Path) -> dict:
    """Structural checks of a written snapshot against its source (no tensor values)."""
    si = json.loads((src_dir / INDEX).read_text())
    di = json.loads((dst_dir / INDEX).read_text())
    if di["weight_map"] != si["weight_map"]:
        raise AssertionError("weight_map differs from the source (2-shard map not kept)")
    total, n = 0, 0
    for shard in sorted(set(di["weight_map"].values())):
        dh, data = check_shard_framing(dst_dir / shard)
        _, sh = read_header(src_dir / shard)
        sh.pop("__metadata__", None)
        if set(dh) != set(sh):
            raise AssertionError(f"{shard}: key set differs from the source shard")
        for k, e in dh.items():
            if e["dtype"] != "BF16":
                raise AssertionError(f"{k}: dtype {e['dtype']} != BF16")
            if e["shape"] != sh[k]["shape"]:
                raise AssertionError(f"{k}: shape {e['shape']} != source {sh[k]['shape']}")
            if di["weight_map"][k] != shard:
                raise AssertionError(f"{k}: index says {di['weight_map'][k]}, header is in {shard}")
        total += data
        n += len(dh)
    if n != EXPECT_N_TENSORS:
        raise AssertionError(f"{n} tensors != {EXPECT_N_TENSORS}")
    if total != EXPECT_BF16_BYTES or int(di["metadata"]["total_size"]) != EXPECT_BF16_BYTES:
        raise AssertionError(f"data bytes {total} / total_size {di['metadata']['total_size']} != {EXPECT_BF16_BYTES}")
    for f in NON_WEIGHT_FILES:
        if sha256_file(dst_dir / f) != sha256_file(src_dir / f):
            raise AssertionError(f"{f}: copy differs from the source")
    return {"n_tensors": n, "total_size": total, "all_bf16": True, "weight_map_kept": True}


def preflight(src: Path, out_hub: Path) -> dict:
    out_hub = Path(out_hub).expanduser().resolve()
    if out_hub == FORBIDDEN_ROOT.resolve() or FORBIDDEN_ROOT.resolve() in out_hub.parents:
        raise SystemExit(f"refusing to write under {FORBIDDEN_ROOT} (PLAN.md WP-2.1b)")
    if not (src / INDEX).is_file():
        raise SystemExit(f"source snapshot missing: {src}")
    out_hub.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_hub).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"only {free / 1e9:.1f} GB free under {out_hub}; need >= {MIN_FREE_BYTES / 1e9:.0f} GB")
    si = json.loads((src / INDEX).read_text())
    if int(si["metadata"]["total_size"]) != EXPECT_SRC_BYTES or len(si["weight_map"]) != EXPECT_N_TENSORS:
        raise SystemExit(f"unexpected source index: total_size {si['metadata']['total_size']}, "
                         f"{len(si['weight_map'])} keys")
    return {"free_bytes_before": free, "out_hub": str(out_hub)}


def check_source_blobs(src: Path) -> dict:
    """HF stores LFS files as blobs named by their sha256: hashing them proves the source is the DP1 revision."""
    out = {}
    for shard in sorted(set(json.loads((src / INDEX).read_text())["weight_map"].values())):
        p = src / shard
        want = Path(os.readlink(p)).name if p.is_symlink() else None
        got = sha256_file(p)
        if want is not None and got != want:
            raise SystemExit(f"{shard}: sha256 {got} != blob name {want}")
        out[shard] = got
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-hub", type=Path, default=DEFAULT_OUT_HUB,
                    help="HF_HUB_CACHE root of the copy (never ~/.cache/huggingface)")
    ap.add_argument("--verify", choices=("full", "structure"), default="full")
    ap.add_argument("--verify-only", action="store_true", help="re-check an existing copy, write nothing")
    ap.add_argument("--threads", type=int, default=4, help="torch intra-op threads")
    a = ap.parse_args(argv)
    import torch

    torch.set_num_threads(a.threads)
    src = a.src.expanduser()
    pre = preflight(src, a.out_hub)
    out_hub = Path(pre["out_hub"])
    rdir = repo_dir(out_hub)
    final = rdir / "snapshots" / BF16_SHA
    rec = {"src": f"{SRC_REPO}@{SRC_SHA}", "src_dir": str(src), "dst": f"{BF16_REPO}@{BF16_SHA}",
           "dst_dir": str(final), "converter": CONVERTER, "pid": os.getpid(),
           "affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0), **pre}
    t0 = time.monotonic()
    rec["src_sha256"] = check_source_blobs(src)
    rec["t_src_hash_s"] = round(time.monotonic() - t0, 2)

    if a.verify_only:
        work = final
    else:
        if final.exists():
            raise SystemExit(f"{final} exists; use --verify-only to re-check it")
        work = rdir / "snapshots" / f".tmp-{BF16_SHA[:12]}-{os.getpid()}"
        work.mkdir(parents=True)
        t1 = time.monotonic()
        si = json.loads((src / INDEX).read_text())
        shards, total = {}, 0
        for shard in sorted(set(si["weight_map"].values())):
            shards[shard] = convert_shard(src / shard, work / shard)
            total += shards[shard]["data_bytes"]
            print(f"  {shard}: {shards[shard]}", flush=True)
        n_f32 = sum(s["src_dtypes"]["F32"] for s in shards.values())
        if n_f32 != EXPECT_N_F32:
            raise AssertionError(f"{n_f32} F32 source tensors != {EXPECT_N_F32}")
        (work / INDEX).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": si["weight_map"]},
                                             indent=2) + "\n")
        for f in NON_WEIGHT_FILES:
            shutil.copyfile(src / f, work / f)  # follows the blob symlink: a real file
        rec["shards"] = shards
        rec["t_convert_s"] = round(time.monotonic() - t1, 2)

    t2 = time.monotonic()
    rec["checks"] = verify_output(src, work)
    if a.verify == "full":
        keys = list(json.loads((src / INDEX).read_text())["weight_map"])
        rec["checks"]["values_equal_src_to_bf16"] = verify_values(src, work, keys)
    rec["t_verify_s"] = round(time.monotonic() - t2, 2)
    t3 = time.monotonic()
    rec["sha256"] = {p.name: sha256_file(p) for p in sorted(work.iterdir()) if p.is_file()}
    rec["t_dst_hash_s"] = round(time.monotonic() - t3, 2)
    if not a.verify_only:
        work.rename(final)
        (rdir / "refs").mkdir(exist_ok=True)
        (rdir / "refs" / "main").write_text(BF16_SHA)
    rec["t_total_s"] = round(time.monotonic() - t0, 2)
    rec["free_bytes_after"] = shutil.disk_usage(out_hub).free
    name = "conversion.json" if not a.verify_only else "verify.json"
    (rdir / name).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in rec if k not in ("shards",)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
