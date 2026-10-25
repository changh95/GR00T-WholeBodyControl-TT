"""x280d against the simulated tile (PLAN.md WP-3.4, T-3.4 dry-run review, the T-3.3 software path, T-3.8 analogues).

No device is opened: every session runs on kmd.FakeDevice with the monitor's x86 build as the hart
(x280/mailbox_sim.py). Servers bind 127.0.0.1 on random free ports (never 5555-5572).

    source $RT/env.sh
    nice -n 19 taskset -c 0,1,8-11,16,17,24-27 bash -c 'source $RT/env.sh && xpy -m pytest tests/x280/test_x280d_sim.py -q'
    (T-3.4 runs it with simpy, together with test_kmd_fake.py.)

Needs the firmware and kernel builds (make -C $RT/x280/fw x280 x86; make -C $RT/x280/fw -f Makefile.kernel
kernel-all) and libwcio (built here if missing).
"""

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import zmq

RT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RT))

from x280 import regs  # noqa: E402

HOST = RT / "x280" / "host"
FW = RT / "x280" / "fw"
# Off the real-time cores 2,3,4,6,7,12-15 and their SMT siblings (task rule; override with X280_CPUS).
X280_CPUS = os.environ.get("X280_CPUS", "0,1,8-11,16,17,24-27")
NICE = ["nice", "-n", "19", "taskset", "-c", X280_CPUS]
XPY = RT / ".venv-x280" / "bin" / "python"
PY = str(XPY) if XPY.exists() else sys.executable
X280D = RT / "x280" / "x280d.py"
OBS516 = Path(os.environ.get("WBC_OBS516_NPZ", RT / "runs/x280/20261024_211935_obs516_kernel/obs516.npz"))
BAL, WALK = "GR00T-WholeBodyControl-Balance", "GR00T-WholeBodyControl-Walk"
# Invented CMFW telemetry (tag -> raw CMFW u32) for the decoding tests only; the sim itself publishes none (D4).
FAKE_TELEMETRY = {regs.TAG_AICLK: 800, regs.TAG_ASIC_TEMPERATURE: (45 << 16) | 0x8000, regs.TAG_VCORE: 750,
                  regs.TAG_TDP: 31, regs.TAG_TDC: 41, regs.TAG_INPUT_POWER: 62}


def _build():
    so, src = HOST / "libwcio.so", HOST / "wcio.c"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(NICE + ["make", "-s", "-C", str(HOST)], check=True)
    need = [FW / "build" / p for p in ("monitor.bin", "park_stub.bin", "x86/libmonitor_x86.so",
                                       "kernel/wbc_fp16.blob", "kernel/wbc_fp32_balance.blob",
                                       "kernel/wbc_fp32_walk.blob", "kernel/libwbc_mlp.so", "payloads/hang.bin")]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        pytest.skip(f"firmware/kernel build outputs missing: {missing}", allow_module_level=True)


_build()
from rtloop import wire  # noqa: E402
from x280 import x280d as D  # noqa: E402
from x280 import mailbox_sim as MS  # noqa: E402
from x280.fw import mailbox_layout as ml  # noqa: E402
from x280.host import kmd as K  # noqa: E402


def _cpu_set(spec: str) -> set:
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _sim_cpu() -> int:
    """A CPU for the sim hart's spin thread inside the daemon's own pin set (X280_CPUS)."""
    return _sim_cpus(1)[0]


def _sim_cpus(n: int) -> list:
    """n CPUs, one per sim hart spin thread (DP4 fp32 runs two harts): an unpinned spinner competes with x280d and
    the other hart, and a preempted hart misses the 4 ms INFER deadline (seen as an L2CPU2 timeout under load)."""
    allowed = sorted((os.sched_getaffinity(0) & _cpu_set(X280_CPUS)) or _cpu_set(X280_CPUS))
    return allowed[-n:]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
    assert not 5555 <= p <= 5572
    return p


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(RT)}


class Daemon:
    """`x280d.py serve --backend sim` in a subprocess, plus a REQ socket to it."""

    def __init__(self, tmp_path, *extra):
        self.port = _free_port()
        self.url = f"tcp://127.0.0.1:{self.port}"
        self.log = tmp_path / f"x280d_{self.port}.log"
        self.log_dir = tmp_path / f"x280d_{self.port}"
        tiles = extra[list(extra).index("--tile") + 1] if "--tile" in extra else "3"
        cpus = ",".join(map(str, _sim_cpus(len(tiles.split(",")))))
        cmd = NICE + [PY, str(X280D), "serve", "--backend", "sim", "--port", str(self.port), "--sim-cpus",
                      cpus, "--log-dir", str(self.log_dir), *extra]
        self.proc = subprocess.Popen(cmd, cwd=RT, env=_env(), stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        self.ctx = zmq.Context()
        self.sock = None
        self._connect()
        rep = self.rpc("ping", timeout_ms=60_000)
        assert rep.get("status") == "ok", (rep, self.log.read_text())

    def _connect(self):
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.url)

    def rpc(self, ep, data=None, timeout_ms=10_000):
        self.sock.send(wire.request(ep, data))
        if not self.sock.poll(timeout_ms):
            self._connect()
            raise TimeoutError(f"{ep}: no reply; log:\n{self.log.read_text()[-3000:]}")
        return wire.loads(self.sock.recv())

    def act(self, policy, obs, tick=0, seq=0, ep="gear_wbc_action", **kw):
        return self.rpc(ep, {"policy": policy, "obs": np.asarray(obs, np.float32).reshape(1, 516), "tick": tick,
                             "seq": seq, **kw})

    def admin(self, cmd, timeout_ms=60_000, **kw):
        return self.rpc("x280d_admin", {"cmd": cmd, **kw}, timeout_ms)

    def stop(self):
        if self.sock is not None:
            self.sock.close(0)
        self.ctx.term()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        return self.proc.returncode


@pytest.fixture
def daemon(tmp_path):
    made = []

    def start(*extra):
        d = Daemon(tmp_path, *extra)
        made.append(d)
        return d
    yield start
    for d in made:
        rc = d.stop()
        assert rc == 0, f"x280d exited {rc}:\n{d.log.read_text()[-3000:]}"


@pytest.fixture
def hold(monkeypatch, tmp_path):
    monkeypatch.setenv("CHIP_SESSION_ID", f"test-hold-{os.getpid()}")
    return tmp_path / "ledger.json"


def _ref(blob_path):
    return D.X86Ref(Path(blob_path).read_bytes())


def _rng_obs(rng, n):
    return rng.standard_normal((n, 516)).astype(np.float32)


# --- T-3.4: the dry-run review ---------------------------------------------------------------------------------

OP_RE = re.compile(r"^(\d{6}) (\S+)\s*(?:\((\d+),(\d+)\))?\s*(?:0x([0-9a-f]{16}))?\s*(?:(\d+)B)?\s*"
                   r"(?:= 0x([0-9a-f]{8}))?\s*(.*)$")
TX_RE = re.compile(r"^(\d{7}) (\S+)\s+(\S+) (\S+) (\d+)B a64=(\d) a4=(\d) (\S+)(?: = 0x([0-9a-f]{8}))?$")


def parse_dry_run(text: str) -> dict:
    """{steps: [labels], ops: [{i, kind, xy, addr, size, value, detail, step}], txn: [...], summary: {...}}"""
    out = {"steps": [], "ops": [], "txn": [], "summary": None}
    section, step = None, None
    for line in text.splitlines():
        if line.startswith("## OPS"):
            section, step = "ops", None
            continue
        if line.startswith("## TXN"):
            section, step = "txn", None
            continue
        if line.startswith("## SUMMARY "):
            out["summary"] = json.loads(line[len("## SUMMARY "):])
            section = None
            continue
        if line.startswith("STEP "):
            step = line[5:]
            if section == "ops":
                out["steps"].append(step)
            continue
        if section == "ops":
            m = OP_RE.match(line)
            assert m, line
            i, kind, x, y, addr, size, value, detail = m.groups()
            out["ops"].append({"i": int(i), "kind": kind, "xy": (int(x), int(y)) if x else None,
                               "addr": int(addr, 16) if addr else None, "size": int(size) if size else 0,
                               "value": int(value, 16) if value else None, "detail": detail, "step": step})
        elif section == "txn":
            m = TX_RE.match(line)
            assert m, line
            seq, op, xy, noc, size, a64, a4, kind, value = m.groups()
            out["txn"].append({"seq": int(seq), "op": op, "xy": xy, "noc": None if noc == "-" else int(noc, 16),
                               "size": int(size), "a64": a64 == "1", "a4": a4 == "1", "kind": kind, "step": step})
    return out


STEP_PREFIXES = ("step 1:", "step 2:", "step 3:", "steps 4-5:", "step 6:", "step 7:", "step 8:", "step 9:",
                 "step 10:")


def check_boot_dry_run(d: dict, tile: int = 3) -> None:
    """Every T-3.4 assertion on the parsed `x280d.py boot --tile 3 --dry-run` output."""
    xy = regs.L2CPU_XY[tile]
    assert len(d["steps"]) == len(STEP_PREFIXES) and all(s.startswith(p) for s, p in zip(d["steps"], STEP_PREFIXES, strict=True))
    ops = d["ops"]
    by = {p: [o for o in ops if o["step"] and o["step"].startswith(p)] for p in STEP_PREFIXES}
    writes = [o for o in ops if o["kind"] in ("W32", "WC64")]
    # step 1: keeper fd first (legacy), windows, then reads only; interlock values
    assert ops[0]["kind"] == "OPEN" and "legacy" in ops[0]["detail"]
    s1 = by["step 1:"]
    assert not [o for o in s1 if o["kind"] in ("W32", "WC64")]
    r = {o["addr"]: o["value"] for o in s1 if o["kind"] == "R32" and o["xy"] == regs.ARC_XY}
    assert r[regs.L2CPU_RESET] == 0x0F and r[regs.PLL4_CNTL_1] == regs.PLL4_CNTL_1_DEFAULT
    assert r[regs.PLL4_CNTL_5] == regs.PLL4_CNTL_5_DEFAULT and r[regs.PLL4_USE_POSTDIV] & 0xF == 0xF
    way = [o for o in s1 if o["kind"] == "R32" and o["addr"] == regs.CCACHE0_WAYENABLE]
    assert way and way[0]["value"] == 0 and way[0]["xy"] == xy
    assert [o["kind"] for o in s1].index("NOTE") < [o["kind"] for o in s1].index("R32")  # settle before any read
    # step 2: LIM init = every line, contiguous, fenced per line, nothing else written
    s2 = [o for o in by["step 2:"] if o["kind"] in ("W32", "WC64")]
    assert all(o["kind"] == "WC64" and "fence_each" in o["detail"] and "lim_init" in o["detail"] for o in s2)
    a = regs.LIM_BASE
    for o in s2:
        assert o["addr"] == a
        a += 64 * int(re.search(r"x(\d+) lines", o["detail"]).group(1))
    assert a == regs.LIM_END
    # step 3: the image as full lines, then read back line by line (16 UC words per kmd line read)
    img = (FW / "build" / "monitor.bin").read_bytes()
    nlines = (len(img) + 63) // 64
    s3w = [o for o in by["step 3:"] if o["kind"] in ("W32", "WC64")]
    assert len(s3w) == 1 and s3w[0]["addr"] == regs.IMAGE_BASE and f"x{nlines} lines" in s3w[0]["detail"]
    assert "image" in s3w[0]["detail"]
    rb = [o for o in by["step 3:"] if o["kind"] == "RWORDS" and o["xy"] == xy]
    assert [(o["addr"], o["size"]) for o in rb] == [(regs.IMAGE_BASE + 64 * i, 64) for i in range(nlines)]
    rbt = [t for t in d["txn"] if t["step"] and t["step"].startswith("step 3:") and t["op"] == "UC_R32"
           and t["xy"] == f"({xy[0]},{xy[1]})"]
    assert [t["noc"] for t in rbt] == [regs.IMAGE_BASE + 4 * i for i in range(nlines * 16)]
    # steps 4-5: GP scratch, RNMI handlers, reset vectors, in that order, values pinned; then read back
    s45 = [(o["addr"], o["value"]) for o in by["steps 4-5:"] if o["kind"] == "W32"]
    stub = (FW / "build" / "park_stub.bin").read_bytes()
    want = [(regs.GP_SCRATCH + 4 * i, int.from_bytes(stub[4 * i:4 * i + 4], "little")) for i in range(16)]
    for h in range(4):
        want += [(regs.rnmi_handler(h), regs.PARK_STUB), (regs.rnmi_handler(h) + 4, 0),
                 (regs.rnmi_exc_handler(h), regs.RNMI_EXC_STUB), (regs.rnmi_exc_handler(h) + 4, 0)]
    for h in range(4):
        want += [(regs.reset_vector(h), regs.RESET_PC), (regs.reset_vector(h) + 4, 0)]
    assert s45 == want
    rb45 = {o["addr"] for o in by["steps 4-5:"] if o["kind"] == "R32" and "transient" in o["detail"]}
    assert {a for a, _ in want} <= rb45
    # steps 6 and 8: exactly tt-bh-linux's one-step PLL4 walks, each write preceded by the relock interlock reads
    # through the transient mapping (review F5), CNTL_0 and PLL_LOCK the last two before the store (verify B1, B2:
    # CNTL_1, CNTL_5, USE_POSTDIV, CNTL_0, PLL_LOCK, then the W32, nothing in between)
    assert [(o["addr"], o["value"]) for o in by["step 6:"] if o["kind"] == "W32"] == list(regs.PLL4_STEP[200][2])
    assert [(o["addr"], o["value"]) for o in by["step 8:"] if o["kind"] == "W32"] == list(regs.PLL4_STEP[800][2])
    # every CNTL_1 write keeps refdiv and the postdiv byte as step 1 read them (S1c silicon: 0x0040_0102)
    c1w = [o["value"] for st in ("step 6:", "step 8:") for o in by[st] if o["kind"] == "W32" and
           o["addr"] == regs.PLL4_CNTL_1]
    assert len(c1w) == 128 and {v & 0xFFFF for v in c1w} == {r[regs.PLL4_CNTL_1] & 0xFFFF} == {0x0102}
    for st in ("step 6:", "step 8:"):
        seq = [o for o in by[st] if o["kind"] in ("W32", "R32") and o["xy"] == regs.ARC_XY]
        ws = [i for i, o in enumerate(seq) if o["kind"] == "W32"]
        pre = [regs.PLL4_CNTL_1, regs.PLL4_CNTL_5, regs.PLL4_USE_POSTDIV, regs.PLL4_CNTL_0, regs.PLL_LOCK]
        assert all([o["addr"] for o in seq[i - 5:i]] == pre and all("transient" in o["detail"] for o in seq[i - 5:i])
                   for i in ws), st
        allops = by[st]  # the reads are the last ops before each store: no other op (e.g. a refusal) in between
        wi = [j for j, o in enumerate(allops) if o["kind"] == "W32"]
        assert all([o["addr"] for o in allops[j - 2:j]] == [regs.PLL4_CNTL_0, regs.PLL_LOCK] for j in wi), st
    s8r = {o["addr"]: o["value"] for o in by["step 8:"] if o["kind"] == "R32" and o["xy"] == regs.ARC_XY}
    assert s8r[regs.PLL4_CNTL_1] == regs.PLL4_CNTL_1_DEFAULT and s8r[regs.PLL4_CNTL_5] == regs.PLL4_CNTL_5_DEFAULT
    # step 7 and the whole log: EXACTLY ONE L2CPU_RESET write = preceding read | 0x80, [3:0] = 0xF, then read back
    rst = [o for o in ops if o["kind"] == "W32" and o["addr"] == regs.L2CPU_RESET]
    assert len(rst) == 1 and rst[0]["step"].startswith("step 7:")
    i = ops.index(rst[0])
    prev = [o for o in ops[:i] if o["kind"] == "R32" and o["addr"] == regs.L2CPU_RESET][-1]
    nxt = next(o for o in ops[i + 1:] if o["kind"] in ("R32", "W32"))
    assert rst[0]["value"] == prev["value"] | 0x80 and rst[0]["value"] & 0xF == 0xF
    assert (nxt["kind"], nxt["addr"], nxt["value"]) == ("R32", regs.L2CPU_RESET, rst[0]["value"])
    # step 9: L2PF values, then the cleanup registration on the keeper fd
    assert [(o["addr"], o["value"]) for o in by["step 9:"] if o["kind"] == "W32"] == list(regs.L2PF_WRITES)
    cl = [o for o in by["step 9:"] if o["kind"] == "IOCTL" and "SET_NOC_CLEANUP" in o["detail"]]
    assert len(cl) == 1 and f"addr=0x{regs.MB_HOST_GONE:x}" in cl[0]["detail"] and "data=0x474f4e45" in cl[0]["detail"]
    # step 10: monitor checks, then one mailbox command (SELFTEST) by the §1.4.4 protocol
    s10 = by["step 10:"]
    assert any(o["kind"] == "RWORDS" and o["addr"] == ml.ADDR["L1"] and o["size"] == 64 for o in s10)
    assert [o["addr"] for o in s10 if o["kind"] in ("W32", "WC64")] == [ml.ADDR["L0"], regs.MB_SEQ_REQ]
    # zero writes to WAYENABLE or anything reaching GDDR; every write target in the allowlist
    bad = [dd for dd in regs.DENY if regs.GDDR_REACHING in dd.why or dd.name == "CCACHE0_WAYENABLE"]
    for o in writes:
        assert not [dd for dd in bad if dd.lo <= o["addr"] <= dd.hi and regs._kind(o["xy"]) == dd.tiles], o
        allow, denies, why = regs.classify_write(o["xy"], o["addr"], 4 if o["kind"] == "W32" else 64)
        assert allow is not None and all(dd.exception == allow.ctx for dd in denies), (o, why)
    assert not [o for o in ops if o["kind"] == "REFUSED"]
    # libwcio: every LIM-init store a 64 B-aligned 64 B line followed by sfence; no partial write before init
    tx = d["txn"]
    init = [j for j, t in enumerate(tx) if t["step"] and t["step"].startswith("step 2:") and t["op"] == "WC64"]
    assert len(init) == regs.LIM_LINES
    assert all(tx[j]["size"] == 64 and tx[j]["a64"] and tx[j + 1]["op"] == "SFENCE" for j in init)
    inited = set()
    for t in tx:
        if t["op"] == "WC64" and t["noc"] is not None:
            inited.add(t["noc"] & ~63)
        elif t["op"] == "UC_W32" and t["noc"] is not None and regs.LIM_BASE <= t["noc"] < regs.LIM_END:
            assert t["noc"] & ~63 in inited, t
    summ = d["summary"]["txn"]
    assert summ["wc64_all_aligned_64B"] and summ["wc64_all_fenced"] and summ["lim_partial_before_init"] == 0
    assert summ["refused"] == 0 and not summ["log_wrapped"]
    assert d["summary"]["fake_ioctls"].count("SET_NOC_CLEANUP") == 1
    assert set(d["summary"]["fake_ioctls"]) <= set(K.IOCTL_NAMES.values())


@pytest.fixture(scope="module")
def dry_run_text(tmp_path_factory):
    r = subprocess.run(NICE + [PY, str(X280D), "boot", "--tile", "3", "--dry-run"], cwd=RT, env=_env(),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_dry_run_boot_is_the_section_5_2_sequence(dry_run_text):
    """T-3.4 on a fresh `x280d.py boot --tile 3 --dry-run`."""
    d = parse_dry_run(dry_run_text)
    check_boot_dry_run(d)
    res = json.loads(dry_run_text.strip().splitlines()[-1])
    assert res["boot"]["l2cpu_reset"] == {"before": 0x0F, "written": 0x8F, "read_back": 0x8F}
    assert res["boot"]["selftest"]["3"]["pass"] and res["boot"]["monitor"]["3"]["fw_magic"] == ml.MAGIC
    assert "/dev/tenstorrent was not opened" in dry_run_text


def test_committed_dry_run_artefact():
    """The T-3.4 artefact $RT/runs/dryrun_boot.txt (if it was generated) passes the same review."""
    p = RT / "runs" / "dryrun_boot.txt"
    if not p.exists():
        pytest.skip("runs/dryrun_boot.txt not generated (xpy x280/x280d.py boot --tile 3 --dry-run > runs/dryrun_boot.txt)")
    check_boot_dry_run(parse_dry_run(p.read_text()))


def test_sim_and_dry_run_never_open_the_device(monkeypatch, hold, capsys):
    real_open = os.open

    def guard(path, *a, **kw):
        assert "tenstorrent" not in str(path), f"os.open({path}) in a sim session"
        return real_open(path, *a, **kw)
    monkeypatch.setattr(os, "open", guard)
    monkeypatch.setattr(K.RealBackend, "open", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("RealBackend")))
    assert D.main(["probe", "--json", "--dry-run", "--ledger", str(hold)]) == 0
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rep["dirty"] is False and rep["l2cpu_reset"] == 0x0F and rep["backend"] == "sim"
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot()
    finally:
        s.close()


def test_device_backend_refuses_outside_devx(monkeypatch, hold, capsys):
    """The device path stops at kmd.assert_devx_hold before /dev/tenstorrent/0 is opened."""
    monkeypatch.delenv("WITH_DEVICE_RESET_AFTER", raising=False)

    def boom(*a, **kw):
        raise AssertionError("os.open reached")
    monkeypatch.setattr(K.os, "open", boom)
    assert D.main(["probe", "--json", "--ledger", str(hold)]) == 2
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "DEVX" in rep["error"]


# --- probe / boot preconditions ------------------------------------------------------------------------------------


def _patched_chip(monkeypatch, **chip_kw):
    orig = MS.SimChip.create.__func__

    def create(cls, tiles=(3,), **kw):
        return orig(cls, tiles, **{**kw, **chip_kw})
    monkeypatch.setattr(MS.SimChip, "create", classmethod(create))


@pytest.mark.parametrize("chip_kw,rc,why", [
    ({"l2cpu_reset": 0x8F}, 3, "released without a chip reset"),
    ({"enabled_l2cpu": 0x7}, 4, "harvested"),
])
def test_probe_reports_dirty_or_unusable(monkeypatch, hold, capsys, chip_kw, rc, why):
    _patched_chip(monkeypatch, **chip_kw)
    assert D.main(["probe", "--json", "--dry-run", "--ledger", str(hold)]) == rc
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert why in json.dumps(rep)
    assert rep["dirty"] is (rc == 3)


# S1c's two silicon probes (2026-10-25, p150a, CMFW 19.12.0.0), before and after a chip reset (regs.PLL4_CNTL_1_DEFAULT)
S1C_PROBES = [RT / "runs/sessions/20261025_050343_2067005_s1c/x280_probe.1.json",
              *sorted((RT / "runs/sessions").glob("*_s1c_probe2/x280_probe.1.json"))[-1:]]


def _recorded_chip(monkeypatch, rec: dict) -> None:
    """Every SimChip of this process reads the registers of `rec` (an `x280d probe --json` report): L2CPU_RESET,
    PLL4 CNTL_0/1/5 + USE_POSTDIV, PLL_LOCK, ENABLED_L2CPU, NoC translation and the heartbeat. The tile registers
    (WAYENABLE, CCACHE0) stay FakeDevice's: the S1c probe skipped them (invariant 4, PLL4 read dirty then)."""
    orig = MS.SimChip.create.__func__

    def create(cls, tiles=(3,), **kw):
        chip = orig(cls, tiles, **{**kw, "l2cpu_reset": rec["l2cpu_reset"], "enabled_l2cpu": rec["enabled_l2cpu"],
                                   "noc_translation": rec["noc_translation"], "heartbeat": rec["timer_heartbeat"]})
        for reg, key in ((regs.PLL4_CNTL_0, "cntl_0"), (regs.PLL4_CNTL_1, "cntl_1"), (regs.PLL4_CNTL_5, "cntl_5"),
                         (regs.PLL4_USE_POSTDIV, "use_postdiv")):
            chip.poke32(regs.ARC_XY, reg, rec["pll4"][key])
        chip.poke32(regs.ARC_XY, regs.PLL_LOCK, rec["pll_lock"])
        chip.update_l2cpuclk_telemetry()
        return chip
    monkeypatch.setattr(MS.SimChip, "create", classmethod(create))


@pytest.mark.parametrize("probe", S1C_PROBES, ids=lambda p: p.parent.name)
def test_s1c_silicon_registers_pass_probe_e0_and_the_boot_dry_run(monkeypatch, hold, capsys, probe):
    """S1c read PLL4 CNTL_1 0x0040_0102 and was called dirty only by the old pll.c:226 default (postdiv 0). The same
    values in FakeDevice now give a clean probe, a passing E0 with s1c.sh's S2 GO (rules unchanged) and the T-3.4
    boot dry-run: both walks keep refdiv and the postdiv byte, one L2CPU_RESET RMW 0x0F -> 0x8F."""
    from x280.tools import m3_jobs as M
    if not probe.exists():
        pytest.skip(f"{probe} is not recorded in this checkout")
    rec = json.loads(probe.read_text())
    assert rec["backend"] == "device" and rec["reasons"] == ["PLL4 not at the CMFW defaults"]  # S1c's verdict
    assert rec["pll4"] == {"cntl_0": 0x12, "cntl_1": 0x0040_0102, "cntl_5": 0x0101_0101, "use_postdiv": 0xF}
    assert (rec["pll_lock"], rec["l2cpu_reset"], rec["enabled_l2cpu"], rec["noc_translation"]) == (0x1F, 0x0F, 0xF, 1)
    assert rec["pll4_walk_ready"] and rec["tiles"]["3"] == {"enabled": True, "released": False}
    _recorded_chip(monkeypatch, rec)
    assert D.main(["probe", "--json", "--dry-run", "--ledger", str(hold)]) == 0
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rep["dirty"] is False and rep["reasons"] == [] and rep["pll4_default"] and rep["pll4_walk_ready"]
    assert rep["pll4"] == rec["pll4"] and rep["pll_lock"] == rec["pll_lock"]
    assert rep["tiles"]["3"]["wayenable"] == 0  # the batch now opens: the first L2CPU-tile read of a session
    assert D.main(["e0", "--dry-run", "--ledger", str(hold)]) == 0
    e0 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert e0["pass"] and e0["pll4_default"] and e0["pll4"] == rec["pll4"] and e0["pll4_walk_ready"]
    v = M.s2_verdict(e0, e0, True)
    assert v["go"] and not v["why"] and not v["notes"] and v["e0_keeper"]["pll4_default"] is True
    assert D.main(["boot", "--tile", "3", "--dry-run", "--ledger", str(hold)]) == 0
    out = capsys.readouterr().out
    check_boot_dry_run(parse_dry_run(out))
    res = json.loads(out.strip().splitlines()[-1])
    assert res["boot"]["l2cpu_reset"] == {"before": 0x0F, "written": 0x8F, "read_back": 0x8F}
    assert res["boot"]["selftest"]["3"]["pass"]


def test_probe_calls_the_pll_c_source_value_dirty_and_unexplained(monkeypatch, hold, capsys):
    """CNTL_1 0x0040_0002 (FW pll.c:226, postdiv 0) was never read on silicon: probe exit 3 naming it unexplained,
    no L2CPU-tile access and no write; E0 fails with the same reason."""
    rec = {"l2cpu_reset": 0x0F, "enabled_l2cpu": 0xF, "noc_translation": 1, "timer_heartbeat": 1000, "pll_lock": 0x1F,
           "pll4": {"cntl_0": 0x12, "cntl_1": regs.PLL4_CNTL_1_SOURCE, "cntl_5": regs.PLL4_CNTL_5_DEFAULT,
                    "use_postdiv": 0xF}}
    _recorded_chip(monkeypatch, rec)
    assert D.main(["probe", "--json", "--dry-run", "--ledger", str(hold)]) == 3
    out = capsys.readouterr().out
    rep = json.loads(out.strip().splitlines()[-1])
    assert rep["dirty"] and not rep["pll4_default"] and len(rep["reasons"]) == 1
    assert rep["reasons"][0].startswith("PLL4 not at the CMFW defaults: CNTL_1 0x00400002 is the bh_arc source value")
    assert "unexplained" in rep["reasons"][0] and "wayenable" not in rep["tiles"]["3"]
    ops = parse_dry_run(out)["ops"]
    assert ops and not [o for o in ops if o["kind"] in ("W32", "WC64")]
    assert not [o for o in ops if o["xy"] is not None and o["xy"] != regs.ARC_XY and o["kind"] in ("R32", "RWORDS")]
    assert D.main(["e0", "--dry-run", "--ledger", str(hold)]) == 1
    e0 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert e0["pass"] is False and "unexplained" in e0["dirty"][0]


@pytest.mark.parametrize("poke,exc", [
    ((regs.ARC_XY, regs.L2CPU_RESET, 0x8F), D.Dirty),
    ((regs.L2CPU_XY[3], regs.CCACHE0_WAYENABLE, 0xF), D.Dirty),
    ((regs.ARC_XY, regs.PLL4_CNTL_1, regs.PLL4_200_CNTL_1), D.Dirty),
    ((regs.ARC_XY, regs.PLL4_CNTL_1, regs.PLL4_CNTL_1_SOURCE), D.Dirty),  # pll.c:226's postdiv 0: unexplained
    ((regs.ARC_XY, regs.PLL_LOCK, 0x0F), D.Dirty),  # verify B3: PLL4 unlocked -> not walk-ready, refused at the start
    ((regs.ARC_XY, regs.PLL4_CNTL_0, 0x13), D.Dirty),  # verify B3: CNTL_0 != 0x12 (reset bit / relock in flight)
    ((regs.ARC_XY, regs.PLL4_CNTL_0, 0x02), D.Dirty),  # bypass clear (pll.c:319 order): not walk-ready
])
def test_boot_refuses_before_any_write(hold, poke, exc):
    s = D.Session("sim", ledger_path=hold)
    s.chip.poke32(*poke)
    try:
        with pytest.raises(exc, match="walk-ready" if poke[1] in (regs.PLL_LOCK, regs.PLL4_CNTL_0) else None):
            s.boot()
        assert not [o for o in s.k.oplog if o.kind in ("W32", "WC64")]
        assert s.chip.hart(3).state == "reset"
    finally:
        s.close()


def test_boot_refuses_a_spent_ledger_tile(hold):
    s = D.Session("sim", ledger_path=hold)
    try:
        s.open()
        s.ledger_hold(1)
        s.ledger.mark_released(3, 0)
        with pytest.raises(D.Dirty, match="ledger"):
            s.boot()
        assert not [o for o in s.k.oplog if o.kind in ("W32", "WC64")]
    finally:
        s.close()


# --- serving (W-WBC) -----------------------------------------------------------------------------------------------


def test_serve_sim_fp16_contract_and_bit_parity(daemon):
    d = daemon()
    info = d.rpc("info")
    assert info["backend"] == "x280-sim:v3-fp16" and info["proto"] == 2 and info["mode"] == "booted"
    assert sorted(info["policies"]) == [BAL, WALK] and info["boot_parity"]["pass"]
    assert info["parity_tol"] == 1e-2 and info["compute_us_src"] == "host_turnaround"
    ref = _ref(D.BLOB_FP16)
    rng = np.random.default_rng(5)
    for i, obs in enumerate(_rng_obs(rng, 200)):
        p = (BAL, WALK)[i & 1]
        rep = d.act(p, obs, tick=1000 + i, seq=i)
        assert "error" not in rep, rep
        assert rep["tick"] == 1000 + i and rep["seq"] == i and np.isfinite(rep["compute_us"])
        a = rep["action"]
        assert a.dtype == np.float32 and a.shape == (1, 15)
        np.testing.assert_array_equal(a[0], ref(i & 1, 2, obs))  # v2 on the hart == the x86 v2 build, bit for bit
    for bad, msg in (({"policy": BAL, "obs": np.zeros((1, 515), np.float32)}, "obs must be"),
                     ({"policy": "nope", "obs": np.zeros((1, 516), np.float32)}, "unknown policy"),
                     ({"policy": BAL, "obs": np.zeros((1, 516), np.float64)}, "obs must be")):
        assert msg in d.rpc("gear_wbc_action", bad)["error"]
    assert "Unknown endpoint" in d.rpc("predict_debug", {})["error"]
    st = d.admin("status")
    assert st["counts"]["ok"] == 200 and st["tiles"]["3"]["n_runs"] >= 200 and st["tiles"]["3"]["state"] == 1


def test_serve_sim_dp4_fp32_two_tiles(daemon):
    d = daemon("--dtype", "fp32", "--tile", "3,2")
    info = d.rpc("info")
    assert info["backend"] == "x280-sim:v0-fp32" and info["policy_tile"] == {BAL: 3, WALK: 2}
    assert sorted(info["policies"]) == [BAL, WALK]
    refs = {0: _ref(D.BLOB_FP32[0]), 1: _ref(D.BLOB_FP32[1])}
    rng = np.random.default_rng(6)
    for i, obs in enumerate(_rng_obs(rng, 40)):
        rep = d.act((BAL, WALK)[i & 1], obs, tick=i, seq=i)
        np.testing.assert_array_equal(rep["action"][0], refs[i & 1](i & 1, 0, obs))
    st = d.admin("status")
    assert st["tiles"]["3"]["n_runs"] >= 20 and st["tiles"]["2"]["n_runs"] >= 20


def test_dp4_release_is_one_rmw_and_both_cleanups_fire(hold):
    s = D.Session("sim", (3, 2), dtype="fp32", ledger_path=hold)
    try:
        r = s.boot()
        assert r["l2cpu_reset"] == {"before": 0x0F, "written": 0xCF, "read_back": 0xCF}
        assert len([o for o in s.k.oplog if o.kind == "W32" and o.addr == regs.L2CPU_RESET]) == 1
        assert s.k.backend.names().count("SET_NOC_CLEANUP") == 2 and len(s.extra_fds) == 1
        assert s.ledger.state(2) == s.ledger.state(3) == "released"
    finally:
        s.close(keep_chip=True)
    for t in (3, 2):
        assert s.chip.peek32(regs.L2CPU_XY[t], regs.MB_HOST_GONE) == regs.HOST_GONE_VALUE
    s.chip.shutdown()


def test_watchdog_stall_then_rnmi_park_recovers(daemon):
    """A wedged hart -> heartbeat stall -> error reply (rt_sim falls back); PARK protocol -> warm entry -> serving."""
    d = daemon()
    obs = np.ones(516, np.float32)
    assert "error" not in d.act(BAL, obs)
    assert d.admin("inject", what="hang")["injected"] == "hang"
    time.sleep(0.02)
    # first tick after the hang: the last heartbeat sample predates it, so the INFER runs into the 4 ms deadline
    t0 = time.perf_counter()
    rep = d.act(BAL, obs, tick=1, seq=1)
    assert "x280 timeout" in rep["error"] and time.perf_counter() - t0 < 0.05
    time.sleep(0.02)
    # next tick: unchanged for > 5 ms and the hart is not BUSY -> watchdog error at once, no mailbox write
    t0 = time.perf_counter()
    rep = d.act(BAL, obs, tick=2, seq=2)
    assert "heartbeat stalled" in rep["error"] and time.perf_counter() - t0 < 0.004
    p = d.admin("park")
    assert p["pass"] and p["n_reentry"] == [0, 1] and p["state"] == ml.STATE["REENTERED"]
    assert p["mncause"] == (1 << 63) | 2
    for i in range(5):
        assert "error" not in d.act(WALK, obs, tick=3 + i, seq=3 + i)
    st = d.admin("status")
    assert st["counts"]["stalls"] == 1 and st["counts"]["timeouts"] == 1 and st["tiles"]["3"]["n_reentry"] == 1


def test_e2_rehearsal_checksum_and_rnmi_rearm_10x(daemon):
    """T-3.E2's host part on the sim: LOAD+CALL checksum 10/10; hang -> PARK -> CALL -> hang ... 10/10."""
    d = daemon()
    r = d.admin("e2", n=10, timeout_ms=120_000)
    assert r["pass"], r
    assert all(r["checksum"]) and [x["n_reentry"] for x in r["rearm"]] == list(range(1, 11))
    assert r["load_hang"]["sim_payload"] == "hang" and r["load_checksum"]["sim_payload"] == "checksum"
    # the payloads overwrote the blob region: INFER now fails (MB_ERR_BLOB) until the blob is LOADed again
    rep = d.act(BAL, np.zeros(516, np.float32), tick=5, seq=5)
    assert "x280 fault" in rep["error"] and f"0x{ml.ERR['BLOB']:x}" in rep["error"]
    assert d.admin("load_blob")["len"] == D.BLOB_FP16.stat().st_size == 1_889_088
    assert "error" not in d.act(BAL, np.zeros(516, np.float32), tick=6, seq=6)


def test_crc_mismatch_is_a_fault_reply_then_recovers(daemon):
    """T-3.8 case 3 on the sim: a corrupted obs CRC -> retried 50 us -> FAULT 0xC0C -> error reply -> next tick OK."""
    d = daemon()
    obs = np.full(516, 0.25, np.float32)
    assert d.admin("inject", what="crc")["armed"] == "crc"
    rep = d.act(BAL, obs, tick=1, seq=1)
    assert "x280 fault" in rep["error"] and "mcause 0xc0c" in rep["error"]
    rep = d.act(BAL, obs, tick=2, seq=2)
    assert "error" not in rep
    st = d.admin("status")
    assert st["counts"]["faults"] == 1 and st["tiles"]["3"]["n_crc_fail"] == 1


def test_early_dispatch_hit_and_miss(daemon):
    d = daemon("--early")
    obs = np.random.default_rng(2).standard_normal(516).astype(np.float32)
    plain = d.act(WALK, obs, tick=10, seq=10)["action"]
    rep = d.act(WALK, obs, tick=11, seq=11, ep="gear_wbc_dispatch")
    assert rep["dispatched"] and rep["tick"] == 11
    col = d.rpc("gear_wbc_collect", {"tick": 11, "seq": 11})
    assert col["early"] and col["tick"] == 11
    np.testing.assert_array_equal(col["action"], plain)
    d.act(BAL, obs, tick=12, seq=12, ep="gear_wbc_dispatch")
    assert "error" not in d.act(WALK, obs, tick=12, seq=12)  # policy changed: synchronous call, the dispatch is a miss
    assert "no dispatch pending" in d.rpc("gear_wbc_collect", {"tick": 12, "seq": 12})["error"]
    c = d.admin("status")["counts"]
    assert c["early_hit"] == 1 and c["early_miss"] == 1


def test_interlock_loss_degrades_then_recovers(hold):
    """PLL4 off its defaults (a relock caught mid-run) -> every reply is an error until the 1 Hz check passes again."""
    import threading
    s = D.Session("sim", ledger_path=hold)
    port = _free_port()
    srv = D.Server(s, port)
    try:
        s.boot()
        for t in s.tiles:
            with s.batch(t):
                s.load_blob(t)
        th = threading.Thread(target=srv.run, daemon=True)
        th.start()
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://127.0.0.1:{port}")

        def call(i):
            sock.send(wire.request("gear_wbc_action", {"policy": BAL, "obs": np.zeros((1, 516), np.float32),
                                                       "tick": i, "seq": i}))
            assert sock.poll(5000)
            return wire.loads(sock.recv())
        assert "error" not in call(0)
        s.chip.poke32(regs.ARC_XY, regs.PLL4_CNTL_5, 0x0F0F_0F0F)
        t0 = time.monotonic()
        while "interlock" not in call(1).get("error", ""):  # a load-induced timeout reply is not the degrade
            assert time.monotonic() - t0 < 3.0
            time.sleep(0.05)
        assert "interlock" in call(2)["error"]
        s.chip.poke32(regs.ARC_XY, regs.PLL4_CNTL_5, regs.PLL4_CNTL_5_DEFAULT)
        t0 = time.monotonic()
        while "error" in call(3):
            assert time.monotonic() - t0 < 3.0
            time.sleep(0.05)
        sock.close(0)
        ctx.term()
    finally:
        srv.stop = True
        time.sleep(0.3)
        srv.close()
        s.close()


# --- T-3.8 analogue: x280d gone -> 'GONE' -> the monitor idles -> a new x280d attaches without a release ---------


def test_host_gone_then_attach_without_release(hold):
    s1 = D.Session("sim", ledger_path=hold)
    s1.boot()
    with s1.batch(3):
        s1.load_blob(3)
    s1.check_parity()
    chip = s1.chip
    s1.close(keep_chip=True)  # keeper fd closed: the KMD's cleanup write lands in L38
    t0 = time.monotonic()
    while chip.peek32(regs.L2CPU_XY[3], ml.ADDR["L1"] + ml.OFFSETS["l1.state"]) != ml.STATE["HOST_GONE"]:
        assert time.monotonic() - t0 < 2.0
        time.sleep(0.002)
    s2 = D.Session("sim", ledger_path=hold, chip=chip)
    try:
        r = s2.attach()
        assert r["tiles"]["3"]["fw_magic"] == ml.MAGIC
        assert not [o for o in s2.k.oplog if o.kind == "W32" and o.addr == regs.L2CPU_RESET]
        assert s2.k.backend.names().count("SET_NOC_CLEANUP") == 2  # one per keeper fd (s1's and s2's)
        assert chip.peek32(regs.L2CPU_XY[3], regs.MB_HOST_GONE) == 0
        bp = s2.check_parity()  # the blob registration survived (monitor persistent state): no reload needed
        assert bp["pass"]
        assert s2.ledger.state(3) == "released" and chip.hart(3).boots == 1
    finally:
        s2.close()


def test_attach_refuses_without_a_release_in_this_hold(hold):
    s = D.Session("sim", ledger_path=hold)
    try:
        with pytest.raises(D.Dirty, match="not released"):
            s.attach()
    finally:
        s.close()


# --- S1c (no release): E0, E1a, host-only bench ------------------------------------------------------------------


def test_no_release_e0_e1a_and_host_only_bench(hold):
    s = D.Session("sim", ledger_path=hold, cross_read=True, hart=False)
    try:
        s.open()
        s.mode = "no_release"
        e0 = s.e0()
        assert e0["pass"] and e0["tiles"]["3"]["cross_read_equal"] and e0["noc_translation"] == 1
        assert e0["pll4_walk_ready"] and e0["pll_lock"] & 0x10  # what boot steps 6/8 will require (review F5)
        # review F7: no per-hart device (BEU 0x0170_0000..) is read while the harts are in reset
        assert "beu_hart0" not in e0["tiles"]["3"]
        assert not [o for o in s.k.oplog if o.addr is not None and regs.beu_base(0) <= o.addr < regs.beu_base(3) + 0x1000]
        r = D.run_admin(s, "lim_init", method="A", seed=7)
        assert r["lines"] == regs.LIM_LINES
        c = D.run_admin(s, "lim_check", seed=7, stride=4, partial=100, cleanup_check=True)
        assert c["pass"], c
        assert c["mismatches"] == 0 and c["partial_writes"] == 100 and c["cleanup_ok"] and not c["ecc_fail_delta"]
        b = D.run_admin(s, "bench_host_only", n=200)
        assert b["n"] == 200 and b["total_us"]["p50"] > 0
        assert s.chip.peek32(regs.ARC_XY, regs.L2CPU_RESET) == 0x0F  # nothing released
        wrong = D.run_admin(s, "lim_check", seed=8, stride=16)
        assert not wrong["pass"] and wrong["mismatches"] > 0
    finally:
        s.close()


def test_lim_writers_refused_under_a_live_monitor(hold):
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot()
        for cmd, kw in (("lim_init", {}), ("lim_check", {"partial": 1}), ("lim_check", {"cleanup_check": True})):
            with pytest.raises(D.X280Error, match="no-release"):
                D.run_admin(s, cmd, **kw)
    finally:
        s.close()


# --- standalone subcommands and the T-3.5 client -----------------------------------------------------------------


def test_parity_subcommand_vs_x86_and_logged_ort(tmp_path):
    if OBS516.exists():
        src = OBS516
        n = 400
    else:
        src = tmp_path / "obs.npz"
        rng = np.random.default_rng(9)
        np.savez(src, obs=_rng_obs(rng, 64), policy=np.array([BAL, WALK] * 32))
        n = 64
    out = tmp_path / "parity.npz"
    r = subprocess.run(NICE + [PY, str(X280D), "parity", "--backend", "sim", "--obs", str(src), "--n", str(n),
                               "--out", str(out), "--sim-cpus", str(_sim_cpu())], cwd=RT, env=_env(),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["pass"] and rep["n_served"] == n
    for v in rep["per_policy"].values():
        assert v["bits_differ_vs_x86"] == 0
        if "max_abs_vs_ort" in v:
            assert v["max_abs_vs_ort"] <= 1e-2  # fp16 (T-3.1: 6.8e-3 / 7.4e-3 worst case)
    z = np.load(out)
    assert z["act_dev"].shape == (n, 15) and z["served"].all()


def test_bench_rtt_client(daemon):
    d = daemon()
    r = subprocess.run(NICE + [PY, str(X280D), "bench", "--rtt", "--via", d.url, "--n", "300", "--period-ms", "2",
                               "--alternate"], cwd=RT, env=_env(), capture_output=True, text=True, timeout=120)
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["ok"] == 300 and rep["timeouts"] == 0 and not rep["errors"], rep
    assert rep["rtt_us"]["p50"] < 2000.0
    # PLAN T-3.5's gates are x280d side and alone decide pass; the client RTT is reported, never gated
    assert set(rep["checks"]) == {"x280d_p99_le_compute_p99_plus_500us", "x280d_p99_le_limit",
                                  "host_overhead_p99_le_300us", "no_timeouts_or_errors"}
    assert rep["pass"] == all(rep["checks"].values()) and rep["gates"].startswith("PLAN T-3.5")
    assert "client_rtt_p99_le_limit" in rep["reported"] and rep["reported"]["client"]["cpus"] is None
    assert (r.returncode == 0) == rep["pass"]


def test_bench_rtt_client_placement_and_its_refusals(daemon, monkeypatch):
    """--cpus places the client as rt_sim's tick thread (lowest CPU) and zmq I/O thread (highest); --spin busy-polls.
    Both belong to --rtt; a --cpus outside CHIP_SAFE_CPUS is refused before any call (x280d's own rule)."""
    d = daemon()
    lo, hi = _sim_cpus(3)[:2]
    r = subprocess.run(NICE + [PY, str(X280D), "bench", "--rtt", "--via", d.url, "--n", "50", "--period-ms", "2",
                               "--cpus", f"{hi},{lo}", "--spin"], cwd=RT, env=_env(), capture_output=True, text=True,
                       timeout=120)
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["ok"] == 50 and rep["reported"]["client"] == {"cpus": f"{hi},{lo}", "spin": True, "thread_cpus": [lo],
                                                             "io_cpu": hi}, rep
    r = subprocess.run(NICE + [PY, str(X280D), "bench", "--via", d.url, "--cpus", str(lo)], cwd=RT, env=_env(),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "without --rtt" in r.stderr
    safe = sorted(_cpu_set(X280_CPUS) - {hi})
    r = subprocess.run(NICE + [PY, str(X280D), "bench", "--rtt", "--via", d.url, "--n", "5", "--cpus", str(hi)],
                       cwd=RT, env={**_env(), "CHIP_SAFE_CPUS": ",".join(map(str, safe))}, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 2 and "outside CHIP_SAFE_CPUS" in r.stderr
    assert d.admin("status")["counts"]["calls"] == 50  # the refused runs sent nothing


def test_bench_rtt_host_row_needs_data_on_the_device(monkeypatch):
    """The 0.3 ms host-overhead gate is waived only for the sim (compute_us is its turnaround there); on the device a
    run without compute_us has no host-overhead data and fails that row instead of passing it vacuously."""
    import argparse
    import threading
    port = _free_port()
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.bind(f"tcp://127.0.0.1:{port}")
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            if rep.poll(20):
                req = wire.loads(rep.recv())
                rep.send(wire.dumps({"backend": "x280:l2cpu3:h0@800MHz:v2-fp16"} if req["endpoint"] == "info" else
                                    {"action": [0.0] * 15, "x280d_us": 600.0, "mailbox_us": 590.0}))
    th = threading.Thread(target=serve, daemon=True)
    th.start()
    try:
        a = argparse.Namespace(via=f"tcp://127.0.0.1:{port}", obs=None, n=20, period_ms=1.0, timeout_ms=1000.0,
                               alternate=False, slow=False, out=None, cpus=None, spin=False)
        assert D.cmd_bench_rtt(a) == 1
    finally:
        stop.set()
        th.join(2)
        rep.close(0)
        ctx.term()


def test_via_admin_subcommands(daemon):
    d = daemon()
    env = _env()
    r = subprocess.run(NICE + [PY, str(X280D), "selftest", "--via", d.url], cwd=RT, env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    st = json.loads(r.stdout.strip().splitlines()[-1])
    assert st["pass"] and st["lines"] == regs.LIM_LINES
    r = subprocess.run(NICE + [PY, str(X280D), "ping", "--via", d.url], cwd=RT, env=env, capture_output=True,
                       text=True, timeout=30)
    assert r.returncode == 0 and json.loads(r.stdout)["info"]["backend"] == "x280-sim:v3-fp16"
    r = subprocess.run(NICE + [PY, str(X280D), "lim-init", "--via", d.url], cwd=RT, env=env, capture_output=True,
                       text=True, timeout=30)
    assert r.returncode == 1 and "no-release" in r.stdout
    r = subprocess.run(NICE + [PY, str(X280D), "bench", "--via", d.url, "--iters", "50"], cwd=RT, env=env,
                       capture_output=True, text=True, timeout=60)
    b = json.loads(r.stdout.strip().splitlines()[-1])
    assert b["iters"] == 50 and b["nondet"] == 0 and b["rc"] == 0


def test_rtloop_zmqwbc_against_the_sim(daemon):
    """rt_sim's own client (rtloop.wbc_backends.ZmqWbc) against x280d. ZmqWbc checks parity at connect with the
    server's own info.parity_tol, capped at 1e-2 (MAX_PARITY_TOL): fp32 (DP4) reports 5e-5 and fp16 1e-2, so both
    connect; fp16 differs from ORT by ~5e-3 by design (T-3.1: 6.8e-3 worst case)."""
    try:
        from rtloop import wbc_backends as wb
        ort = wb.OrtCpu()
    except Exception as e:  # noqa: BLE001 - xpy has no onnxruntime; run under simpy
        pytest.skip(f"needs onnxruntime (simpy): {e}")
    d32 = daemon("--dtype", "fp32", "--tile", "3,2")
    c = wb.ZmqWbc(d32.url, ort, timeout_ms=5)
    try:
        assert c.backend == wb.WbcBackend.X280_SIM and c.proto == 2 and c.parity_max_err < 5e-5
        obs = np.random.default_rng(1).standard_normal((1, 516)).astype(np.float32)
        a, meta = c(BAL, obs, tick=7, seq=7)
        assert not meta["fallback"] and float(np.max(np.abs(a - ort(BAL, obs)))) < 5e-5
    finally:
        c.close()
    d16 = daemon()
    c16 = wb.ZmqWbc(d16.url, ort, timeout_ms=5)
    try:
        assert c16.backend == wb.WbcBackend.X280_SIM and c16.parity_tol == 1e-2 and c16.parity_max_err < 1e-2
    finally:
        c16.close()


# --- device-session helpers on the sim: E1F gate, E2 heartbeat, E3 powertest, E4 matrix, E5a blob switch, E0 peek --


def test_method_a_boot_refused_on_the_device_and_names_stage0(monkeypatch, hold):
    """verify finding: a device `boot` / `serve --boot` (method A: host LIM init, which S1c/S1d showed poisons the LIM
    ECC) is refused without X280_ALLOW_METHOD_A=1 (exit 2), before any Session exists, i.e. before anything could
    open the device, and the refusal names --stage0; `serve` on the device without a mode lists --stage0 too.
    Session.boot() refuses the same on a non-sim session before any op. The sim and --dry-run boot as before."""
    env = {k: v for k, v in _env().items() if k not in ("X280_ALLOW_METHOD_A", "X280_E1F_APPROVED")}
    port = _free_port()
    for argv in (["boot", "--tile", "3"], ["boot", "--tile", "3", "--load-blob"],
                 ["serve", "--boot", "--port", str(port), "--tile", "3"]):
        r = subprocess.run(NICE + [PY, str(X280D), *argv], cwd=RT, env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 2, (argv, r.stdout, r.stderr)
        assert "method-A boot refused" in r.stderr and "--stage0" in r.stderr and "X280_ALLOW_METHOD_A=1" in r.stderr
        assert "step 1" not in r.stderr and "keeper fd" not in r.stderr  # nothing was opened
        if argv[0] == "boot":
            assert json.loads(r.stdout.splitlines()[-1]) == {"boot": {"error": D.METHOD_A_REFUSED}, "steps": []}
    r = subprocess.run(NICE + [PY, str(X280D), "serve", "--port", str(port)], cwd=RT, env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 2 and "--stage0" in r.stderr and "--boot is method A" in r.stderr, r.stderr
    monkeypatch.delenv("X280_ALLOW_METHOD_A", raising=False)
    s = D.Session("sim", ledger_path=hold)
    try:
        s.sim = False  # what a device session is to boot(); the guard must come before self.open()
        with pytest.raises(D.X280Error, match="--stage0"):
            s.boot()
        assert s.k.fd is None and not [o for o in s.k.oplog if o.kind in ("W32", "WC64", "OPEN")]
    finally:
        s.sim = True
        s.close()
    s = D.Session("sim", ledger_path=hold)  # the sim keeps method A, no override needed
    try:
        out = s.boot()
        assert out["selftest"]["3"]["pass"] and [o for o in s.k.oplog if o.kind == "WC64" and "lim_init" in o.detail]
    finally:
        s.close()
    r = subprocess.run(NICE + [PY, str(X280D), "boot", "--tile", "3", "--dry-run"], cwd=RT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "step 2: LIM init, method A" in r.stderr, r.stderr[-2000:]


def test_stage0_refused_without_approval_and_works_in_sim(monkeypatch, hold):
    monkeypatch.delenv("X280_E1F_APPROVED", raising=False)
    s = D.Session("sim", ledger_path=hold)
    try:
        with pytest.raises(D.X280Error, match="DP3"):
            s.boot_stage0()
        assert not [o for o in s.k.oplog if o.kind in ("W32", "WC64")] and s.k.fd is None
    finally:
        s.close()
    port = _free_port()
    r = subprocess.run(NICE + [PY, str(X280D), "serve", "--backend", "sim", "--stage0", "--port", str(port)],
                       cwd=RT, env={k: v for k, v in _env().items() if k != "X280_E1F_APPROVED"},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "DP3(d)" in r.stderr
    monkeypatch.setenv("X280_E1F_APPROVED", "1")  # sim only: the chip is a FakeDevice
    s = D.Session("sim", ledger_path=hold)
    try:
        out = s.boot_stage0()
        assert out["tiles"]["3"]["selftest"]["pass"] and out["l2cpu_reset"]["written"] == 0x8F
        assert not [o for o in s.k.oplog if o.kind == "WC64" and "lim_init" in o.detail]  # the host never inits LIM
        assert s.chip.peek(regs.L2CPU_XY[3], regs.GP_SCRATCH, 64) == (FW / "build" / "park_stub.bin").read_bytes()
        assert [o.kind for o in s.k.oplog if o.kind == "ASSUME_INIT"] == ["ASSUME_INIT"]
        with s.batch(3):
            s.load_blob(3)
        assert s.check_parity()["pass"]
    finally:
        s.close()


def test_e2_heartbeat_e3_powertest_e4_matrix_e5_blob_switch(hold):
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot()
        with s.batch(3):
            s.load_blob(3)
        hb = s.heartbeat_rate(3, secs=0.05)
        assert hb["pass"] and hb["changes_per_s"] > 1000
        pt = s.powertest(3, fd_churn=3, gap_s=0.02)
        assert pt["pass"] and pt["hart"] and len(pt["pll4_default_after_each"]) == 3
        assert [c[0] for c in s.k.backend.calls].count("open") == 4  # keeper + 3 legacy churn opens
        m = s.bench_matrix(3, iters=20)
        got = {(r["blob"], r["variant"], r["policy"]) for r in m["runs"]}
        assert ("fp16", "v2", BAL) in got and ("fp16", "v2s", WALK) in got and ("fp16", "v3", WALK) in got
        assert ("fp32_balance", "v0", BAL) in got and ("fp32_walk", "v0", WALK) in got
        assert all(r["rc"] == 0 and r["nondet"] == 0 and r["iters"] == 20 for r in m["runs"])
        assert s.blobs[3]["dtype"] == "fp16" and s.check_parity()["pass"]  # the session blob is back
        rng = np.random.default_rng(4)
        obs = _rng_obs(rng, 16)
        pol = np.array([0, 1] * 8)
        s.use_blob(3, "fp32_walk")
        r = s.parity(obs, pol, variant="v0")
        assert r["pass"] and list(r["per_policy"]) == [WALK] and r["n_served"] == 8
        with pytest.raises(D.X280Error, match="needs fp16"):
            s.parity(obs, pol, variant="v2")
        with pytest.raises(D.X280Error, match="needs fp16"):
            s.parity(obs, pol, variant="v3")
        s.use_blob(3, "session")
        r = s.parity(obs, pol, variant="v3")  # bit-identical to x86 v2 (X86_FAMILY), both policies
        assert r["pass"] and r["variant"] == "v3" and r["n_served"] == 16
        assert all(x["bits_differ_vs_x86"] == 0 for x in r["per_policy"].values())
    finally:
        s.close()


def test_no_release_powertest_without_a_hart(hold):
    s = D.Session("sim", ledger_path=hold, hart=False)
    try:
        s.open()
        s.mode = "no_release"
        r = D.run_admin(s, "powertest", fd_churn=2, gap_s=0.01)
        assert r["pass"] and r["hart"] is False and r["max_heartbeat_gap_ms"] is None
    finally:
        s.close()


def test_e0_peek_cross_check_against_a_fake_server_g1(daemon, tmp_path):
    """`e0 --via --peek`: x280d's register reads vs server_g1's x280_peek (here a stub REP that answers from regs
    defaults, i.e. what a correct read_reg returns on a fresh chip)."""
    import threading
    d = daemon("--no-release")  # sim, no release: the S1c shape
    port = _free_port()
    vals = {regs.L2CPU_RESET: 0x0F, regs.PLL4_CNTL_1: regs.PLL4_CNTL_1_DEFAULT, regs.PLL4_CNTL_5: regs.PLL4_CNTL_5_DEFAULT,
            regs.PLL4_USE_POSTDIV: 0xF, regs.CCACHE0_WAYENABLE: 0, regs.CCACHE0_CONFIG: 0x0609_1004}
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.bind(f"tcp://127.0.0.1:{port}")
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            if rep.poll(50):
                req = wire.loads(rep.recv())["data"]
                rep.send(wire.dumps({"values": [vals[req["addr"]]], "x": req["x"], "y": req["y"]}))
    th = threading.Thread(target=serve, daemon=True)
    th.start()
    try:
        r = subprocess.run(NICE + [PY, str(X280D), "e0", "--via", d.url, "--peek", f"tcp://127.0.0.1:{port}"],
                           cwd=RT, env=_env(), capture_output=True, text=True, timeout=60)
        out = json.loads(r.stdout.strip().splitlines()[-1])
        assert r.returncode == 0 and out["pass"] and out["peek"]["agree"] and len(out["peek"]["rows"]) == 6, out
    finally:
        stop.set()
        th.join(2)
        rep.close(0)
        ctx.term()


# --- review:x280 fixes on the sim (F1, F3-F9) ---------------------------------------------------------------------


def test_e1f_boot_then_rnmi_park_returns_to_crt0(monkeypatch, hold):
    """F1: E1F leaves the reset vectors at the stage-0 stub (consumed only at reset, ISA README.md:46); the park stub
    now jumps to crt0 at the fixed 0x0800_0000 (park_stub.S), so a PARK after an E1F boot re-enters the monitor
    (spike shows the same with SPIKE_E1F_VECTORS, test_spike.py). F10: the first PARK comes straight after the boot,
    with no code LOAD in between: it runs the park stub only because hart 0 fenced after E1F (crt0's entry fence.i,
    step E's FENCE_I), dropping stage 0's line from its L1I (the sim models that line; spike's per-instruction cache
    does not)."""
    monkeypatch.setenv("X280_E1F_APPROVED", "1")  # sim only: the chip is a FakeDevice
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot_stage0()
        assert s.chip.peek32(regs.L2CPU_XY[3], regs.reset_vector(0)) == regs.STAGE0_ENTRY
        for i in range(3):
            r = s.park(3)
            assert r["pass"] and r["n_reentry"] == (i, i + 1), (r, s.chip.status())
        with s.batch(3):
            s.load_blob(3)
        assert s.check_parity()["pass"]
    finally:
        s.close()


def test_stage0_boot_then_attach_and_park_without_a_release(monkeypatch, hold):
    """The stage-0 boot is every M3 session's since E1F (2026-10-25), so T-3.8 (kill -9, then `serve --attach`) and
    S2's E3(d) attach to a stage-0 hart: no release, no stage 0 again, LIM taken as the hart left it, the blob
    registration kept, and PARK still returns through the park stub to crt0 (n_reentry + 1)."""
    monkeypatch.setenv("X280_E1F_APPROVED", "1")  # sim only: the chip is a FakeDevice
    s1 = D.Session("sim", ledger_path=hold)
    s1.boot_stage0()
    with s1.batch(3):
        s1.load_blob(3)
    assert s1.check_parity()["pass"] and s1.mode == "booted"
    chip = s1.chip
    s1.close(keep_chip=True)  # keeper fd closed: 'GONE' in L38, the monitor idles
    t0 = time.monotonic()
    while chip.peek32(regs.L2CPU_XY[3], ml.ADDR["L1"] + ml.OFFSETS["l1.state"]) != ml.STATE["HOST_GONE"]:
        assert time.monotonic() - t0 < 2.0
        time.sleep(0.002)
    monkeypatch.delenv("X280_E1F_APPROVED")  # an attach needs no E1F approval: it boots nothing
    s2 = D.Session("sim", ledger_path=hold, chip=chip)
    try:
        r = s2.attach()
        assert r["tiles"]["3"]["fw_magic"] == ml.MAGIC and s2.mode == "attached"
        writes = [o for o in s2.k.oplog if o.kind in ("W32", "WC64")]
        assert not [o for o in writes if o.addr in (regs.L2CPU_RESET, regs.STAGE0_ENTRY_WORD)]
        assert not [o for o in writes if o.addr == regs.GP_SCRATCH]  # no stage-0 / park-stub rewrite
        assert chip.peek32(regs.L2CPU_XY[3], regs.reset_vector(0)) == regs.STAGE0_ENTRY  # used only at reset
        assert s2.check_parity()["pass"] and chip.hart(3).boots == 1
        p = s2.park(3)
        assert p["pass"] and p["n_reentry"] == (0, 1)
        assert s2.check_parity()["pass"]
    finally:
        s2.close()


def _serve_in_thread(sess):
    import threading
    port = _free_port()
    srv = D.Server(sess, port)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    return srv, th, f"tcp://127.0.0.1:{port}"


def _calls(url: str, n: int, tick0: int) -> dict:
    """n W-WBC calls; an unpinned sim hart may miss the 4 ms INFER deadline now and then on a loaded host."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(url)
    rng, out = np.random.default_rng(tick0), {"ok": 0, "errors": []}
    try:
        for i in range(n):
            obs = rng.standard_normal((1, 516)).astype(np.float32)
            sock.send(wire.request("gear_wbc_action", {"policy": (BAL, WALK)[i & 1], "obs": obs, "tick": tick0 + i,
                                                       "seq": tick0 + i}))
            assert sock.poll(5000)
            rep = wire.loads(sock.recv())
            if "error" in rep:
                out["errors"].append(rep["error"])
            else:
                out["ok"] += 1
    finally:
        sock.close(0)
    return out


def test_t38_kill9_attach_then_park_hold_after_a_stage0_boot(monkeypatch, hold):
    """T-3.8 cases 1 and 2 as they ran on silicon (s4, 2026-10-25), in one process (the sim chip lives in its x280d
    process): serve --stage0, blob, calls; kill -9 (the keeper fd closes, the KMD's cleanup writes 'GONE', the monitor
    idles); serve --attach (blob reload, parity, calls); `x280d park --via URL --hold-s 2` through its admin endpoint.
    On silicon the RNMI ran stage 0 from the line hart 0's L1I still held at 0x2001_0100 (no fence.i on hart 0 since
    E1F): LIM zero-filled, every L1/L36 field 0, every call stalled. crt0's fence.i on entry and step E's FENCE_I after
    the swap drop that line, so the RNMI runs the park stub, the hart re-enters warm and serves again."""
    monkeypatch.setenv("X280_E1F_APPROVED", "1")  # sim only: the chip is a FakeDevice
    s1 = D.Session("sim", ledger_path=hold)
    s1.boot_stage0()
    chip = s1.chip
    assert chip.hart(3).status()["l1i_gp"] is None  # crt0 dropped stage 0's line before the host swapped the stub in
    with s1.batch(3):
        s1.load_blob(3)
    srv1, th1, url1 = _serve_in_thread(s1)
    try:
        assert _calls(url1, 50, 0)["ok"] >= 48
    finally:
        srv1.stop = True
        th1.join(5)
        srv1.close()
    s1.close(keep_chip=True)  # kill -9: the keeper fd closes, 'GONE' lands in L38
    t0 = time.monotonic()
    while chip.peek32(regs.L2CPU_XY[3], ml.ADDR["L1"] + ml.OFFSETS["l1.state"]) != ml.STATE["HOST_GONE"]:
        assert time.monotonic() - t0 < 2.0
        time.sleep(0.002)
    monkeypatch.delenv("X280_E1F_APPROVED")
    s2 = D.Session("sim", ledger_path=hold, chip=chip)
    srv2 = None
    try:
        assert s2.attach()["tiles"]["3"]["state"] == ml.STATE["HOST_GONE"]
        with s2.batch(3):
            s2.load_blob(3)
        assert s2.check_parity()["pass"]
        srv2, th2, url2 = _serve_in_thread(s2)
        assert _calls(url2, 50, 1000)["ok"] >= 48
        p = D._admin_call(url2, "park", 30.0, tile=3, hold_s=2.0)
        assert p["pass"] and p["n_reentry"] == [0, 1] and p["state"] == ml.STATE["REENTERED"], (p, chip.status())
        assert p["mncause"] == (1 << 63) | 2 and "stage0_words" not in p
        assert chip.peek(regs.L2CPU_XY[3], regs.IMAGE_BASE, len(s2.image)) == s2.image  # LIM kept
        r = _calls(url2, 20, 2000)
        assert r["ok"] >= 18 and not [e for e in r["errors"] if "stalled" in e], r  # silicon: every call stalled
        st = chip.hart(3).status()
        assert (st["boots"], st["rnmis"], st["reentries"], st["l1i_gp"]) == (1, 1, 1, None)  # crt0 fenced again
        assert not [e for _, e in chip.hart(3).events if "stage 0" in e and "RNMI" in e]
    finally:
        if srv2 is not None:
            srv2.stop = True
            th2.join(5)
            srv2.close()
        s2.close()


def test_a_stale_stage0_line_reruns_stage0_and_park_names_it(monkeypatch, hold):
    """What T-3.8 case 2 looked like on silicon, reproduced with stage 0 put back into the sim hart's L1I (as a crt0
    without its entry fence.i left it): the RNMI runs stage 0, so LIM is zero-filled (L1 and L36 read 0: fw_magic,
    state, heartbeat, seq_done, n_reentry, build_id), the done word 0x081E_0000 is written again and the entry word
    is 0; PARK fails and says so."""
    monkeypatch.setenv("X280_E1F_APPROVED", "1")  # sim only: the chip is a FakeDevice
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot_stage0()
        with s.batch(3):
            s.load_blob(3)
        hart = s.chip.hart(3)
        hart.l1i_load((FW / "build" / "stage0.bin").read_bytes())
        assert hart.status()["l1i_gp"] == "stage0 (stale)"
        p = s.park(3)
        assert not p["pass"] and p["n_reentry"] == (0, 0) and (p["state"], p["mncause"], p["mnepc"], p["seq"]) == (0, 0, 0, 0)
        assert p["stage0_words"] == {"done": regs.LIM_END, "entry": 0} and "ran stage 0" in p["why"], p
        with s.batch(3):
            l1, l36 = s.mb[3].l1(), s.mb[3].l36()
        assert not any(l1.values()) and not any(l36.values()), (l1, l36)
        assert hart.state == "stage0" and "stale GP-scratch line" in hart.events[-1][1]
    finally:
        s.close()


def test_powertest_watch_ends_with_a_pll4_read(hold):
    """S2's E3 watches server_g1 start -> stop -> start in powertest --watch-s windows: each ends with a PLL4 read, so
    every server_g1 event is followed by one (PLAN T-3.E3 'PLL4 at default values after each event')."""
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot()
        r = s.powertest(3, fd_churn=0, watch_s=0.05)
        assert r["pass"] and r["pll4_default_after_each"] == [True]
        r = s.powertest(3, fd_churn=2, watch_s=0.02, gap_s=0.02)
        assert r["pll4_default_after_each"] == [True, True, True]
    finally:
        s.close()


def test_admin_load_into_monitor_region_writes_nothing(hold):
    """F4: `x280d load --dst 0x08001000` (one digit off 0x0801_1000) is refused before any data line is written,
    by x280d's own copy of monitor.c cmd_load's range check and, below it, by kmd's post-release ownership rule."""
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot()
        for dst, n in ((0x0800_1000, 256), (regs.MAILBOX_BASE, 64), (regs.WEIGHTS_BASE + 4, 64),
                       (regs.LIM_END - 64, 128), (regs.WEIGHTS_BASE, 0)):
            k0 = len(s.k.oplog)
            with pytest.raises((D.X280Error, K.KmdError)):
                D.run_admin(s, "load", data=bytes(n), dst=dst)
            assert not [str(o) for o in s.k.oplog[k0:] if o.kind in ("WC64", "W32")], (hex(dst), n)
        with s.batch(3):
            with pytest.raises(K.PolicyError, match="monitor region"):
                s.k.wc_write_lines(0x0800_1000, bytearray(64))
        assert s.chip.peek(regs.L2CPU_XY[3], regs.IMAGE_BASE, len(s.image)) == s.image
        with s.batch(3):
            assert s.selftest(3)["pass"]
    finally:
        s.close()


def test_no_release_paths_refuse_a_released_tile(hold):
    """F3: S1c-style sessions (no-release serve, lim-init, lim-check, host-only bench, standalone e0) are dirty on
    a released tile; attach and probe still see it."""
    chip = MS.SimChip.create(())
    chip.poke32(regs.ARC_XY, regs.L2CPU_RESET, 0x8F)
    s = D.Session("sim", ledger_path=hold, hart=False, chip=chip)
    try:
        s.open()
        inter = s.interlocks()
        assert inter["released_tiles"] == [3] and any("released (a live hart)" in r for r in inter["dirty"])
        assert not s.interlocks(released_ok=True)["dirty"]
        s.mode = "no_release"
        with pytest.raises(K.PolicyError, match="lim_init"):
            s.lim_init(3)
    finally:
        s.close()
        chip.shutdown()


def test_attach_resumes_a_hart_left_in_the_park_stub(hold):
    """F8: x280d died inside park() with the RNMI bit still set: attach finishes the PARK protocol."""
    s1 = D.Session("sim", ledger_path=hold)
    s1.boot()
    with s1.batch(3):
        s1.load_blob(3)
    s1.ledger.mark_parked(3)
    with s1.batch(3):
        s1.k.rnmi_set(3, 1)
    chip = s1.chip
    t0 = time.monotonic()
    while chip.hart(3).state != "rnmi":
        assert time.monotonic() - t0 < 2.0
        time.sleep(0.002)
    s1.close(keep_chip=True)
    s2 = D.Session("sim", ledger_path=hold, chip=chip)
    try:
        r = s2.attach()
        t = r["tiles"]["3"]
        assert t["n_reentry"] == 1 and t["resumed_park"]["n_reentry"] == [0, 1] and not t["resumed_park"]["heartbeat_live"]
        assert s2.ledger.state(3) == "released" and chip.peek32(regs.L2CPU_XY[3], regs.RNMI_TRIGGER) == 0
        assert s2.check_parity()["pass"]
    finally:
        s2.close()


def test_probe_reports_a_resumable_parked_hart(monkeypatch, hold):
    s1 = D.Session("sim", ledger_path=hold)
    s1.boot()
    with s1.batch(3):
        s1.k.rnmi_set(3, 1)
    chip = s1.chip
    time.sleep(0.05)
    s1.close(keep_chip=True)
    s2 = D.Session("sim", ledger_path=hold, chip=chip, hart=False, cross_read=True)
    try:
        s2.open()
        inter = s2.interlocks(released_ok=True)
        assert inter["tiles"]["3"]["released"] and not inter["dirty"]
        s2.k.assume_lim_initialised(3, regs.LIM_BASE, regs.LIM_END, "test")
        with s2.batch(3):
            assert s2.k.read_rnmi_trigger(3) == 1
    finally:
        s2.close()
        chip.shutdown()


def test_boot_stops_and_burns_on_a_relock_during_step_8(hold):
    """F5 through x280d: a CMFW relock mid-walk after the release -> Dirty (exit 3), the tile is burned in the
    ledger, and no PLL4 write follows the relock."""
    s = D.Session("sim", ledger_path=hold)
    n = [0]

    def hook():
        n[0] += 1
        if n[0] == 130:  # 120 writes of step 6, then 10 into step 8
            s.chip.poke32(regs.ARC_XY, regs.PLL4_CNTL_1, regs.PLL4_CNTL_1_DEFAULT)
            s.chip.poke32(regs.ARC_XY, regs.PLL4_CNTL_5, regs.PLL4_CNTL_5_DEFAULT)
    s.k.step_sleep = hook
    try:
        with pytest.raises(D.Dirty, match="PLL4 walk to 800 MHz"):
            s.boot()
        assert s.ledger.state(3) == "burned"
        pll_w = [o for o in s.k.oplog if o.kind == "W32" and o.addr in (regs.PLL4_CNTL_1, regs.PLL4_CNTL_5)]
        assert len(pll_w) == 130
    finally:
        s.close()


def test_e0_with_noc_translation_off_skips_the_cross_read(hold):
    """F6: translation off -> no (8,29) window, no translated read, E0 fails with the reason recorded."""
    chip = MS.SimChip.create((), noc_translation=0)
    s = D.Session("sim", ledger_path=hold, hart=False, cross_read=True, chip=chip)
    try:
        s.open()
        s.mode = "no_release"
        e0 = s.e0()
        assert e0["noc_translation"] == 0 and not e0["pass"] and "translation off" in e0["tiles"]["3"]["cross_read"]
        assert "ccache3x" not in s.k.windows
        assert not [o for o in s.k.oplog if o.xy == regs.L2CPU_XY_TRANSLATED[3] and o.kind != "NOTE"]
    finally:
        s.close()
        chip.shutdown()


def test_bench_pilot_deadline_scalar_cap_and_quiesce_before_load(hold, monkeypatch):
    """F9: BENCH's deadline comes from a 10-iteration pilot; v0/v2s are capped at 100 iterations in the matrix;
    after a BENCH timeout the hart is quiesced (finished or RNMI-parked) before the session blob is LOADed again,
    and a LOAD while a command still runs is refused before any write."""
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot()
        with s.batch(3):
            s.load_blob(3)
        r = s.bench(3, 0, 40, variant="v2")
        assert r["pilot"]["iters"] == D.BENCH_PILOT_ITERS and r["pilot"]["deadline_s"] >= 10.0 and r["iters"] == 40
        monkeypatch.setattr(D, "BENCH_SCALAR_ITERS_MAX", 12)
        m = s.bench_matrix(3, iters=30)
        assert {(x["variant"], x["iters"]) for x in m["runs"]} == {("v2", 30), ("v2s", 12), ("v3", 30), ("v0", 12)}
        mb = s.mb[3]
        with s.batch(3):
            seq = mb.issue(ml.CMD["BENCH"], policy=0, flags=ml.variant_flags(2), arg_len=4000,
                           obs=np.zeros(516, np.float32))
            k0 = len(s.k.oplog)
            with pytest.raises(D.X280Error, match="still running"):
                mb.load(regs.WEIGHTS_BASE, s.blobs[3]["data"], ml.LOAD_KIND["BLOB"])
            assert not [o for o in s.k.oplog[k0:] if o.kind in ("WC64", "W32")]
            q = s.quiesce(3, wait_s=0.0)  # the long BENCH is still running: RNMI park, the stale seq is adopted
            assert q["parked"] and mb.idle() and mb.seq == seq
        s.use_blob(3, "session")
        assert s.check_parity()["pass"]
    finally:
        s.close()


# --- --status-pub (PLAN §4.3 item 7, the HUD chip panel) -----------------------------------------------------------

STATUS_KEYS = {"src", "pid", "t_ns", "seq", "backend", "uptime_s", "tile", "tile_state", "monitor_state",
               "heartbeat_age_us", "heartbeat", "seq_done", "cycles_last", "cycles_max", "compute_us_last", "duty",
               "x280d_us_last", "x280d_us_p50", "x280d_us_p99", "calls_ok", "n_runs",
               "n_overrun", "n_reentry", "beu_accrued", "ccache_ecc", "variant", "fw_ver", "build_id", "telemetry"}
TELEMETRY_KEYS = {"aiclk_mhz", "l2cpuclk_mhz", "asic_temp_c", "vcore_mv", "tdp_w", "tdc_a", "input_power_w",
                  "timer_heartbeat"}


def _sub(endpoint: str, n: int = 3, timeout_s: float = 5.0, until=None) -> list:
    """Up to n status messages (or until `until(msg)` holds) within timeout_s."""
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"status")
    sub.connect(endpoint)
    out, t0 = [], time.monotonic()
    try:
        while len(out) < n and time.monotonic() - t0 < timeout_s:
            if sub.poll(200):
                topic, body = sub.recv_multipart()
                assert topic == b"status"
                out.append(wire.loads(body))
                if until is not None and until(out[-1]):
                    break
    finally:
        sub.close(0)
        ctx.term()
    return out


def test_status_pub_contract_on_the_fake_device(monkeypatch, hold):
    """x280d --status-pub on kmd.FakeDevice + the sim hart: [b"status", wire msg] at 2-10 Hz, every contract key,
    telemetry decoded from a CMFW table holding this test's FAKE_TELEMETRY (Server(fake_telemetry=True), a
    unit-test-only switch), and the publishing thread never touches kmd."""
    import threading
    _patched_chip(monkeypatch, telemetry=FAKE_TELEMETRY)
    s = D.Session("sim", ledger_path=hold)
    srv = D.Server(s, _free_port(), status_pub="0", status_hz=10, fake_telemetry=True)
    main = threading.get_ident()
    seen_threads = set()
    real_read32, real_words = s.k.read32, s.k.read_words

    def r32(*a, **kw):
        seen_threads.add(threading.get_ident())
        return real_read32(*a, **kw)

    def rw(*a, **kw):
        seen_threads.add(threading.get_ident())
        return real_words(*a, **kw)
    s.k.read32, s.k.read_words = r32, rw
    try:
        assert srv.pub.endpoint.startswith("tcp://127.0.0.1:") and srv.info()["status_pub"] == srv.pub.endpoint
        assert not 5555 <= int(srv.pub.endpoint.rsplit(":", 1)[1]) <= 5574  # never a campaign / contract port
        s.boot()
        with s.batch(3):
            s.load_blob(3)
        srv.health(force=True)
        for i in range(60):  # served calls fill compute_us_last / duty
            rep = srv.gear_wbc_action(BAL, np.zeros((1, 516), np.float32), tick=i, seq=i)
            assert "error" not in rep
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:  # the serving loop's cadence: refresh between requests
            srv.status_tick()
            time.sleep(0.02)
        msgs = _sub(srv.pub.endpoint, n=4)
        assert len(msgs) >= 3
        m = msgs[-1]
        assert STATUS_KEYS <= set(m), STATUS_KEYS - set(m)
        assert m["src"] == "x280d" and m["pid"] == os.getpid() and m["backend"] == srv.info()["backend"]
        assert [x["seq"] for x in msgs] == sorted(x["seq"] for x in msgs) and len({x["seq"] for x in msgs}) == len(msgs)
        assert 0 < m["t_ns"] <= time.clock_gettime_ns(time.CLOCK_MONOTONIC) and m["uptime_s"] > 0
        assert m["tile"] == 3 and m["tile_state"] == "released" and m["monitor_state"] == "idle"
        assert m["n_runs"] >= 60 and m["n_reentry"] == 0 and m["beu_accrued"] == 0 and m["n_overrun"] == 0
        assert m["seq_done"] == s.mb[3].seq and m["fw_ver"] == ml.FW_VER and m["build_id"] == s.monitor[3]["l36"]["build_id"]
        assert m["variant"] == "v3" and m["ccache_ecc"] == {"fix": 0, "fail": 0}
        assert m["compute_us_last"] > 0 and 0 < m["duty"] < 1 and m["heartbeat_age_us"] is not None
        assert isinstance(m["heartbeat"], int) and m["heartbeat"] > 0 and m["calls_ok"] == 60 == srv.counts["ok"]
        assert 0 < m["x280d_us_p50"] <= m["x280d_us_p99"] and m["x280d_us_last"] > 0
        tel = m["telemetry"]
        assert m["telemetry_source"] == D.TELEMETRY_SOURCE_TEST == srv.info()["telemetry_source"]
        assert set(tel) == TELEMETRY_KEYS and tel["l2cpuclk_mhz"] == [800] * 4 and tel["aiclk_mhz"] == 800
        assert tel["asic_temp_c"] == 45.5 and tel["vcore_mv"] == 750 and tel["tdp_w"] == 31 and tel["tdc_a"] == 41
        assert tel["input_power_w"] == 62 and tel["timer_heartbeat"] >= 123_456
        gaps = np.diff([x["t_ns"] for x in msgs]) / 1e9
        assert np.all(gaps > 1 / 10.0 * 0.5) and np.all(gaps < 1 / 2.0)  # 2-10 Hz
        assert seen_threads == {main}  # kmd is touched by the serving thread only
    finally:
        srv.close()
        s.close()


def test_status_pub_unknown_values_are_none_before_boot(monkeypatch, hold):
    """Before a hart serves, the monitor fields are None (never invented); telemetry is still read (from this test's
    fake CMFW table)."""
    _patched_chip(monkeypatch, telemetry=FAKE_TELEMETRY)
    s = D.Session("sim", ledger_path=hold, hart=False)
    srv = D.Server(s, _free_port(), status_pub="0", fake_telemetry=True)
    try:
        s.open()
        s.mode = "no_release"
        srv.health(force=True)
        srv.status_tick(force=True)
        m = _sub(srv.pub.endpoint, n=1)[0]
        assert m["monitor_state"] is None and m["cycles_last"] is None and m["n_runs"] is None
        assert m["duty"] is None and m["ccache_ecc"] == {"fix": None, "fail": None}
        assert m["heartbeat"] is None and m["heartbeat_age_us"] is None and m["calls_ok"] == 0
        assert m["x280d_us_last"] is None and m["x280d_us_p50"] is None and m["x280d_us_p99"] is None
        assert m["telemetry"]["aiclk_mhz"] == 800 and m["tile_state"] in (None, "fresh")
    finally:
        srv.close()
        s.close()


def test_status_snapshot_values_that_move(monkeypatch, hold):
    """The HUD's live X280 values (user request 2026-10-25): the raw L1 heartbeat (the hart's mtime: it advances
    between snapshots, while heartbeat_age_us stays ~0 on a healthy hart), x280d_us last/p50/p99 over the last 50 OK
    rows of the tick ring, and the served-call count. The snapshot reads the heartbeat word once, in the _hb_check
    heartbeat_age_us already used: no new device read."""
    _patched_chip(monkeypatch, telemetry=FAKE_TELEMETRY)
    s = D.Session("sim", ledger_path=hold)
    srv = D.Server(s, _free_port(), status_pub="0", fake_telemetry=True)
    try:
        s.boot()
        with s.batch(3):
            s.load_blob(3)
        srv.health(force=True)
        for i in range(70):
            assert "error" not in srv.gear_wbc_action(BAL, np.zeros((1, 516), np.float32), tick=i, seq=i)
        mb = s.mb[3]
        reads = []
        real_hb = mb.heartbeat
        monkeypatch.setattr(mb, "heartbeat", lambda: reads.append(1) or real_hb())
        a = srv.status_snapshot()
        assert len(reads) == 1 and a["heartbeat"] == srv.hb[3][0]
        time.sleep(0.05)
        b = srv.status_snapshot()
        assert len(reads) == 2 and b["heartbeat"] > a["heartbeat"] and b["heartbeat_age_us"] < 1000
        ok = srv._recent_ok(50)
        assert len(ok) == 50 and b["calls_ok"] == 70 == srv.counts["ok"]
        x_us = ok["x280d_us"].astype(np.float64)
        assert b["x280d_us_last"] == round(float(x_us[-1]), 1)
        assert b["x280d_us_p50"] == round(float(np.percentile(x_us, 50)), 1)
        assert b["x280d_us_p99"] == round(float(np.percentile(x_us, 99)), 1)
        srv._record(time.perf_counter_ns(), 70, 70, 0, 3, 0, D.ST_STALL, D.EARLY_NONE)  # a failed call: not served
        c = srv.status_snapshot()
        assert c["calls_ok"] == 70 and c["x280d_us_last"] == b["x280d_us_last"]
    finally:
        srv.close()
        s.close()


def test_sim_status_stream_never_carries_chip_telemetry(monkeypatch, hold, daemon):
    """verify:all D4: `serve --backend sim` publishes telemetry None with telemetry_source 'none (sim)', never chip
    numbers - not even when the FakeDevice's table holds values (the default Server, as the CLI builds it)."""
    d = daemon("--status-pub", "0")
    info = d.rpc("info")
    assert info["telemetry_source"] == D.TELEMETRY_SOURCE_SIM == "none (sim)"
    for i in range(5):
        assert "error" not in d.act(BAL, np.ones(516, np.float32), tick=i, seq=i)
    msgs = _sub(info["status_pub"], n=3, timeout_s=8.0)
    assert len(msgs) >= 2
    for m in msgs:
        assert m["telemetry"] is None and m["telemetry_source"] == "none (sim)" and "telemetry_error" not in m
        assert m["monitor_state"] == "idle"  # the sim's own (simulated) monitor state is still published
    _patched_chip(monkeypatch, telemetry=FAKE_TELEMETRY)
    s = D.Session("sim", ledger_path=hold, hart=False)
    srv = D.Server(s, _free_port(), status_pub="0")
    called = []
    monkeypatch.setattr(s.k, "read_status_telemetry", lambda: called.append(1) or {})
    try:
        s.open()
        s.mode = "no_release"
        srv.status_tick(force=True)
        m = _sub(srv.pub.endpoint, n=1)[0]
        assert m["telemetry"] is None and m["telemetry_source"] == "none (sim)" and not called
    finally:
        srv.close()
        s.close()


def test_status_pub_daemon_flag(daemon):
    """`serve --status-pub 0`: a random port, reported in info(); messages flow while W-WBC calls are served."""
    d = daemon("--status-pub", "0")
    ep = d.rpc("info")["status_pub"]
    assert ep and ep.startswith("tcp://127.0.0.1:") and not 5555 <= int(ep.rsplit(":", 1)[1]) <= 5574
    for i in range(20):
        assert "error" not in d.act(BAL, np.ones(516, np.float32), tick=i, seq=i)
    # the snapshot is refreshed between requests at the publish rate, so the first message may predate call 20
    msgs = _sub(ep, n=50, timeout_s=8.0, until=lambda m: (m["n_runs"] or 0) >= 20)
    assert msgs and msgs[-1]["n_runs"] >= 20 and msgs[-1]["src"] == "x280d", msgs[-1:]
    assert msgs[-1]["snapshot_age_ms"] < 1000


def test_serve_sim_gives_the_hart_its_own_cpu(tmp_path):
    """x280d's own spin-thread CPU handling: without --sim-cpus, serve --backend sim puts the spinning sim hart on a
    CPU of its mask that x280d does not use (and says so); with a single CPU it warns and polls by sleeping."""
    cpus = sorted(_cpu_set(X280_CPUS) & os.sched_getaffinity(0))[:2]
    if len(cpus) < 2:
        pytest.skip("needs two CPUs in X280_CPUS")
    for mask, shared in ((cpus, False), (cpus[:1], True)):
        port = _free_port()
        log = tmp_path / f"serve_{len(mask)}.log"
        cmd = ["nice", "-n", "19", "taskset", "-c", ",".join(map(str, mask)), PY, str(X280D), "serve", "--backend",
               "sim", "--port", str(port), "--log-dir", str(tmp_path / f"d{len(mask)}")]
        proc = subprocess.Popen(cmd, cwd=RT, env=_env(), stdout=open(log, "w"), stderr=subprocess.STDOUT)
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://127.0.0.1:{port}")
        try:
            sock.send(wire.request("info"))
            assert sock.poll(60_000), log.read_text()[-2000:]
            info = wire.loads(sock.recv())
            assert info["sim_shared_cpu"] is shared, (info, log.read_text()[-2000:])
            text = log.read_text()
            assert ("polls by sleeping" in text) if shared else (f"hart spin thread(s) on CPU {mask[-1]}" in text)
        finally:
            sock.close(0)
            ctx.term()
            proc.terminate()
            assert proc.wait(15) == 0, log.read_text()[-2000:]


# --- verify:all D12 (method B guards), D9 (CHIP_SAFE_CPUS), D1 (the tick ring for analyze_run) ---------------------


class FakeLimWriter:
    """A stand-in for server_g1's x280_lim_write debug endpoint (ttnn write_to_core in 64 B chunks): a REP thread that
    writes each request's bytes into the fake chip's LIM and counts them; `after(i)` runs after request i (1-based)."""

    def __init__(self, chip, tile: int = 3, after=None):
        import threading
        self.chip, self.tile, self.after, self.reqs = chip, tile, after, []
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        port = _free_port()
        self.sock.bind(f"tcp://127.0.0.1:{port}")
        self.url = f"tcp://127.0.0.1:{port}"
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._serve, daemon=True)
        self._th.start()

    def _serve(self):
        while not self._stop.is_set():
            if self.sock.poll(50):
                req = wire.loads(self.sock.recv())
                d = req["data"]
                assert req["endpoint"] == "x280_lim_write" and (d["x"], d["y"]) == regs.L2CPU_XY[self.tile]
                assert d["chunk"] == 64 and d["addr"] % 64 == 0 and len(d["data"]) % 64 == 0
                self.chip.poke(regs.L2CPU_XY[self.tile], d["addr"], bytes(d["data"]))
                self.reqs.append((d["addr"], len(d["data"])))
                if self.after:
                    self.after(len(self.reqs))
                self.sock.send(wire.dumps({"ok": True, "n": len(d["data"])}))

    def close(self):
        self._stop.set()
        self._th.join(2)
        self.sock.close(0)
        self.ctx.term()


LO_B, HI_B = 0x0810_0000, 0x0810_4000  # 16 KiB = 4 x_lim_write requests of 4 KiB


def test_lim_init_method_b_writes_through_server_g1_with_method_a_guards(hold):
    """verify:all D12: method B (server_g1's x280_lim_write) on a no-release session: the whole range passes
    kmd.lim_init_check first, then each 4 KiB request its own check inside its own l2cpu_batch, and the lines it
    wrote read back with lim_check. A PLL4 change after request 2 stops it before request 3 (per-request batch)."""
    s = D.Session("sim", ledger_path=hold, hart=False)
    w = FakeLimWriter(s.chip)
    try:
        s.open()
        s.interlocks()
        s.mode = "no_release"
        r = D.run_admin(s, "lim_init", method="B", lo=LO_B, hi=HI_B, seed=2, vla_url=w.url)
        assert r["lines"] == (HI_B - LO_B) // 64 and r["method"] == "B"
        assert w.reqs == [(a, 4096) for a in range(LO_B, HI_B, 4096)]
        checks = [o for o in s.k.oplog if o.kind == "CHECK"]
        assert [(o.addr, o.size) for o in checks] == [(LO_B, HI_B - LO_B)] + [(a, 4096) for a in range(LO_B, HI_B, 4096)]
        assert not [o for o in s.k.oplog if o.kind in ("W32", "WC64")]  # kmd itself wrote nothing
        c = D.run_admin(s, "lim_check", lo=LO_B, hi=HI_B, seed=2)
        assert c["pass"] and c["mismatches"] == 0, c
        w.close()

        def relock(i):
            if i == 2:
                s.chip.poke32(regs.ARC_XY, regs.PLL4_CNTL_5, 0x0F0F_0F0F)
        w = FakeLimWriter(s.chip, after=relock)
        with pytest.raises(K.InterlockError, match="PLL4 not at CMFW defaults"):
            s.lim_init(3, method="B", lo=LO_B, hi=HI_B, seed=3, vla_url=w.url)
        assert len(w.reqs) == 2
        s.chip.poke32(regs.ARC_XY, regs.PLL4_CNTL_5, regs.PLL4_CNTL_5_DEFAULT)
    finally:
        w.close()
        s.close()


def test_lim_init_method_b_refuses_a_released_tile_before_any_request(hold):
    """verify:all D12: kmd.lim_init's per-call release check (review F3) guards method B too: on a released tile not
    one x280_lim_write request is sent and nothing is marked initialised."""
    chip = MS.SimChip.create(())
    chip.poke32(regs.ARC_XY, regs.L2CPU_RESET, 0x8F)
    s = D.Session("sim", ledger_path=hold, hart=False, chip=chip)
    w = FakeLimWriter(chip)
    try:
        s.open()
        assert s.interlocks()["released_tiles"] == [3]
        s.mode = "no_release"
        with pytest.raises(K.PolicyError, match="released=True"):
            s.lim_init(3, method="B", lo=LO_B, hi=HI_B, seed=2, vla_url=w.url)
        assert w.reqs == [] and not [o for o in s.k.oplog if o.kind == "ASSUME_INIT"]
        assert chip.peek(regs.L2CPU_XY[3], LO_B, 64) == bytes(64)
    finally:
        w.close()
        s.close()
        chip.shutdown()


def test_outside_safe_cpus_parses_taskset_lists(monkeypatch):
    import argparse
    monkeypatch.setenv("CHIP_SAFE_CPUS", "0,1,8-11,16,17,24-27")
    ns = argparse.Namespace
    assert D._outside_safe_cpus(ns(cpus="9", sim_cpus="8-10")) is None
    assert D._outside_safe_cpus(ns(cpus=None, sim_cpus=None)) is None
    assert "--cpus 5 is outside" in D._outside_safe_cpus(ns(cpus="5", sim_cpus=None))
    assert "--sim-cpus 5,21 is outside" in D._outside_safe_cpus(ns(cpus="9", sim_cpus="5,21"))
    assert "not a CPU list" in D._outside_safe_cpus(ns(cpus="x", sim_cpus=None))
    monkeypatch.delenv("CHIP_SAFE_CPUS")
    assert D._outside_safe_cpus(ns(cpus="5", sim_cpus="21")) is None  # no confinement: §1.1's pins apply


def test_serve_refuses_cpus_outside_chip_safe_cpus(tmp_path):
    """verify:all D9: like server_g1, `x280d serve` exits 2 before binding anything when --cpus or --sim-cpus leaves
    CHIP_SAFE_CPUS. The CPU named here is still a non-real-time one, so a missed refusal could harm nothing."""
    cpus = sorted(_cpu_set(X280_CPUS) & os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("needs two CPUs in X280_CPUS")
    safe, other = str(cpus[0]), str(cpus[1])
    for flag in ("--cpus", "--sim-cpus"):
        port = _free_port()
        r = subprocess.run(NICE + [PY, str(X280D), "serve", "--backend", "sim", "--port", str(port), flag, other,
                                   "--log-dir", str(tmp_path / flag.strip("-"))], cwd=RT,
                           env={**_env(), "CHIP_SAFE_CPUS": safe}, capture_output=True, text=True, timeout=60)
        assert r.returncode == 2, (r.returncode, r.stderr[-2000:])
        assert f"{flag} {other} is outside CHIP_SAFE_CPUS={safe}" in r.stderr and "READY" not in r.stderr
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", port))  # nothing was bound


def test_sim_cpus_takes_the_taskset_list_the_safe_cpus_check_reads():
    """verify2 N4: --sim-cpus is parsed as the same taskset list the CHIP_SAFE_CPUS check reads (it used to be int()
    per comma, so a range passed the check and then crashed). A range inside the safe set runs T-3.4's dry-run boot;
    one outside it, or a malformed one, exits 2 cleanly before the boot starts."""
    cpus = sorted(_cpu_set(X280_CPUS) & os.sched_getaffinity(0))
    lo = next((c for c in cpus if c + 1 in cpus), None)
    if lo is None:
        pytest.skip("needs two adjacent CPUs in X280_CPUS")
    rng, safe = f"{lo}-{lo + 1}", ",".join(map(str, cpus))
    assert D._cpu_seq(rng) == [lo, lo + 1] and D._cpu_seq("21,5,5") == [21, 5]  # one per hart, in tile order
    with pytest.raises(ValueError):
        D._cpu_seq("9-8")  # empty, as taskset says

    def boot(spec: str, safe_cpus: str) -> subprocess.CompletedProcess:
        return subprocess.run(NICE + [PY, str(X280D), "boot", "--tile", "3", "--dry-run", "--sim-cpus", spec], cwd=RT,
                              env={**_env(), "CHIP_SAFE_CPUS": safe_cpus}, capture_output=True, text=True, timeout=300)

    r = boot(rng, safe)
    assert r.returncode == 0, r.stderr[-3000:]
    assert json.loads(r.stdout.strip().splitlines()[-1])["boot"]["selftest"]["3"]["pass"]
    for spec, safe_cpus, why in ((rng, str(lo), f"--sim-cpus {rng} is outside CHIP_SAFE_CPUS={lo}"),
                                 (f"{lo}-x", safe, f"--sim-cpus '{lo}-x' is not a CPU list")):
        r = boot(spec, safe_cpus)
        assert r.returncode == 2 and why in r.stderr, (r.returncode, r.stderr[-2000:])
        assert "Traceback" not in r.stderr and "## OPS" not in r.stdout and "step 1" not in r.stderr


def test_dump_writes_the_tick_ring_for_analyze_run(daemon, tmp_path):
    """T-3.7 (verify:all D1): `x280d dump --via URL --out DIR` makes the serving daemon write x280d_ticks.npy, oldest
    row first, into DIR (the session dir, where analyze_run's heartbeat gate looks); without --via it refuses."""
    d = daemon()
    for i in range(30):
        assert "error" not in d.act((BAL, WALK)[i & 1], np.full(516, 0.1, np.float32), tick=100 + i, seq=i)
    out = tmp_path / "session"
    out.mkdir()
    r = subprocess.run(NICE + [PY, str(X280D), "dump", "--via", d.url, "--out", str(out)], cwd=RT, env=_env(),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["ring"] == str(out / D.RING_FILE) and rep["rows"] == rep["calls"] == 30 and not rep["wrapped"]
    ring = np.load(out / D.RING_FILE)
    assert ring.dtype == D.TICK_LOG_DTYPE and list(ring["tick"]) == list(range(100, 130))
    assert (ring["status"] == D.ST_OK).all() and np.all(np.diff(ring["t_recv_ns"]) > 0)
    r = subprocess.run(NICE + [PY, str(X280D), "dump", "--out", str(out)], cwd=RT, env=_env(), capture_output=True,
                       text=True, timeout=30)
    assert r.returncode == 2 and "needs --via" in r.stderr


def test_dump_of_a_wrapped_ring_is_oldest_first(monkeypatch, hold, tmp_path):
    monkeypatch.setattr(D, "RING", 8)
    s = D.Session("sim", ledger_path=hold)
    srv = D.Server(s, _free_port(), log_dir=tmp_path)
    try:
        s.boot()
        with s.batch(3):
            s.load_blob(3)
        srv.health(force=True)
        for i in range(20):
            assert "error" not in srv.gear_wbc_action(BAL, np.zeros((1, 516), np.float32), tick=i, seq=i)
        rep = srv.admin("dump")
        assert rep["ring"] == str(tmp_path / D.RING_FILE) and rep["rows"] == 8 and rep["calls"] == 20 and rep["wrapped"]
        assert list(np.load(tmp_path / D.RING_FILE)["tick"]) == list(range(12, 20))
        assert srv.admin("dump", out=str(tmp_path / "t37.npy"))["ring"] == str(tmp_path / "t37.npy")
        with pytest.raises(D.X280Error, match="neither a directory nor a .npy"):
            srv.dump_ring(str(tmp_path / "ring.txt"))
    finally:
        srv.close()
        s.close()


# --- verify B3: PLL4 walk readiness is part of the start interlocks ------------------------------------------------


def test_probe_and_stage0_refuse_a_pll4_that_is_not_walk_ready(monkeypatch, hold, capsys):
    """PLL_LOCK bit 4 clear: probe is dirty (exit 3) with the reason and E0 lists it; boot_stage0 (E1F) refuses at
    its start as boot does, before its GP-scratch / vector stores."""
    rec = {"l2cpu_reset": 0x0F, "enabled_l2cpu": 0xF, "noc_translation": 1, "timer_heartbeat": 1000, "pll_lock": 0x0F,
           "pll4": {"cntl_0": 0x12, "cntl_1": regs.PLL4_CNTL_1_DEFAULT, "cntl_5": regs.PLL4_CNTL_5_DEFAULT,
                    "use_postdiv": 0xF}}
    _recorded_chip(monkeypatch, rec)
    assert D.main(["probe", "--json", "--dry-run", "--ledger", str(hold)]) == 3
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rep["dirty"] and not rep["pll4_walk_ready"] and rep["pll4_default"]
    assert len(rep["reasons"]) == 1 and rep["reasons"][0].startswith("PLL4 not walk-ready: CNTL_0 0x12")
    assert "PLL_LOCK 0xf" in rep["reasons"][0]
    monkeypatch.setenv("X280_E1F_APPROVED", "1")  # sim only
    s = D.Session("sim", ledger_path=hold)
    try:
        with pytest.raises(D.Dirty, match="walk-ready"):
            s.boot_stage0()
        assert not [o for o in s.k.oplog if o.kind in ("W32", "WC64")] and s.chip.hart(3).state == "reset"
    finally:
        s.close()


# --- S1d: lim-ecc on the FakeEcc model (jobs/s1d.sh's measurement, both ways) ---------------------------------------


def _no_release(hold, ecc, **kw):
    s = D.Session("sim", ledger_path=hold, hart=False, sim_ecc=ecc, **kw)
    s.open()
    assert not s.interlocks()["dirty"]
    s.mode = "no_release"
    return s


@pytest.mark.parametrize("ecc,clean", [("poison", []), ("clean", ["a64", "a64nf", "wc2x32", "uc8", "a64+uc8"]),
                                       ("granule", ["uc8", "a64+uc8"])])
def test_lim_ecc_on_the_ecc_model(hold, ecc, clean):
    s = _no_release(hold, ecc)
    try:
        n0 = len(s.k.oplog)
        r = D.run_admin(s, "lim_ecc", lines=128)
        assert r["pass"] and r["integrity"] and r["clean_variants"] == clean and r["mismatches"] == 0
        assert [v["name"] for v in r["variants"]] == D.S1D_PLAN.split(",")
        span = 128 * 64
        for i, v in enumerate(r["variants"]):  # one fresh region per item, from regs.S1D_BASE on
            assert (int(v["lo"], 16), int(v["hi"], 16)) == (regs.S1D_BASE + i * span, regs.S1D_BASE + (i + 1) * span)
            last = v["steps"][-1]
            assert last["partial_writes"] == D.S1D_PARTIAL and set(last["phases"]) == {"write", "read1", "partial",
                                                                                       "read2"}
            assert len(last["probe_write"]) == len(last["probe_read"]) == D.S1D_PROBE_LINES
            assert len(last["probe_partial"]) == D.S1D_PARTIAL and all(x["ok"] for x in last["probe_partial"])
            pf = last["derived"]["partial_fail_events"]
            assert pf["pairs"] == D.S1D_PARTIAL and pf["writes"] + pf["read_backs"] == last["phases"]["partial"][
                "dat_fail_count"]
            if last["clean"]:
                assert pf["writes"] == pf["read_backs"] == 0
            assert all(x["settled"] for st in v["steps"] for x in st["probe_write"])
            assert all(x["event_addr_is_the_read"] for st in v["steps"] for x in st["probe_read"])
        a64 = r["variants"][0]["steps"][0]
        if ecc == "clean":
            assert a64["derived"]["write_events_per_line"] == {"fix": 0.0, "fail": 0.0}
        else:  # S1c: two 32 B accesses per 64 B NT line store, events at +0 / +0x20, fail-only re-reads
            assert a64["derived"]["accesses_per_line_min"] == 2 and set(a64["derived"]["probe_event_offsets"]) <= {0, 32}
            assert a64["derived"]["read1_events_per_word"]["fail"] == pytest.approx(0.935, abs=0.05)
            assert a64["derived"]["read1_events_per_word"]["fix"] == 0.0
            wc = r["variants"][2]["steps"][0]["derived"]  # UMD's 2 x 32 B: one 64 B access per line
            assert wc["accesses_per_line_min"] == 1 and wc["probe_event_offsets"] == [0]
            heal = r["variants"][4]["steps"]
            assert heal[0]["clean"] is False and heal[1]["clean"] is (ecc == "granule")  # 8 B stores heal only there
        # every write: the S1d regions only, ctx lim_init, an allowed width, after the region's init stores
        writes = [o for o in s.k.oplog[n0:] if o.kind in ("W32", "W64", "WC64")]
        lo, hi = regs.S1D_BASE, regs.S1D_BASE + len(r["variants"]) * span
        assert writes and all(lo <= o.addr < hi for o in writes)
        assert all("lim_init" in o.detail for o in writes if o.kind != "W32")
        assert s.k.txn_summary()["lim_partial_before_init"] == 0 and "REFUSED" not in s.k.oplog_text()
        assert s.chip.peek32(regs.ARC_XY, regs.L2CPU_RESET) == 0x0F
        with pytest.raises(D.X280Error, match="already written"):  # S1d measures never-written lines only
            D.run_admin(s, "lim_ecc", lines=128)
        r2 = D.run_admin(s, "lim_ecc", plan="uc8", lines=16, probe_lines=2, partial=4, base=hi)
        assert r2["pass"] and r2["clean_variants"] == (["uc8"] if ecc != "poison" else [])
    finally:
        s.close()


@pytest.mark.parametrize("kw,match", [({"plan": "a64,nt128"}, "plan item"), ({"plan": "a64++"}, None),
                                      ({"plan": ""}, "plan"), ({"base": regs.MAILBOX_BASE}, "weights area"),
                                      ({"lines": 20000}, "weights area"), ({"probe_lines": 5, "lines": 4}, "probe"),
                                      ({"base": regs.S1D_BASE + 4}, "aligned")])
def test_lim_ecc_refuses_bad_arguments_before_any_write(hold, kw, match):
    s = _no_release(hold, "poison")
    try:
        n0 = len(s.k.oplog)
        if match is None:  # 'a64++' is the item a64 (empty '+' parts dropped): runs
            assert D.run_admin(s, "lim_ecc", lines=8, **kw)["variants"][0]["name"] == "a64"
            return
        with pytest.raises((D.X280Error, ValueError), match=match):
            D.run_admin(s, "lim_ecc", **kw)
        assert not [o for o in s.k.oplog[n0:] if o.kind in ("W32", "W64", "WC64")]
    finally:
        s.close()


def test_lim_ecc_refused_with_a_live_or_released_hart(hold):
    s = D.Session("sim", ledger_path=hold)
    try:
        s.boot()
        with pytest.raises(D.X280Error, match="no-release"):
            D.run_admin(s, "lim_ecc")
    finally:
        s.close()
    chip = MS.SimChip.create((), ecc="poison")
    chip.poke32(regs.ARC_XY, regs.L2CPU_RESET, 0x8F)
    s = D.Session("sim", ledger_path=hold, hart=False, chip=chip)
    try:
        s.open()
        s.mode = "no_release"
        with pytest.raises(K.PolicyError, match="released=True"):
            s.lim_ecc(3, lines=8)
        assert not [o for o in s.k.oplog if o.kind in ("W32", "W64", "WC64")]
    finally:
        s.close()
        chip.shutdown()


def test_lim_check_and_lim_init_report_ecc_with_the_gate_unchanged(hold):
    """verify D: lim_check names each part of its (unchanged) gate and the failed ones, and reports the ECC deltas,
    events per word read and where the last event was; lim_init reports its own events. On 'poison' the S1c pattern
    (E1a fails on the ECC fail delta alone, last fail at L38 after the cleanup check, or at a partial write); on
    'clean' and without a model it passes as before."""
    for ecc in ("poison", "clean", None):
        s = _no_release(hold, ecc)
        try:
            lo, hi = regs.MAILBOX_BASE, regs.MAILBOX_BASE + 512 * 64  # L38 inside: the cleanup check writes it
            init = D.run_admin(s, "lim_init", method="A", lo=lo, hi=hi, seed=1)
            c = D.run_admin(s, "lim_check", lo=lo, hi=hi, seed=1, partial=20, cleanup_check=True)
            assert set(c["checks"]) == {"data_read_back", "partial_writes_read_back", "ecc_fail_counts_unchanged",
                                        "l2cpu_reset_unchanged", "wayenable_unchanged_0", "cleanup_wrote_gone"}
            assert c["mismatches"] == 0 and c["cleanup_ok"] and c["lim_accesses"]["word_reads"] == 512 * 16
            if ecc == "poison":
                assert init["ecc_events_per_line"]["fail"] == pytest.approx(1.87, abs=0.1) and init["store"] == "a64"
                assert init["ecc_delta"]["dat_fail_at"]["where"] == "inside the range"
                assert c["pass"] is False and c["failed_checks"] == ["ecc_fail_counts_unchanged"]
                assert c["ecc_events_per_word_read"]["fail"] == pytest.approx(0.935, abs=0.05)
                assert c["ecc_events_per_word_read"]["fix"] == 0 and c["ecc_delta"]["dat_fail_count"] > 0
                at = c["ecc_delta"]["dat_fail_at"]  # the last event: the cleanup's L38 accesses come last
                l38 = s.chip.ecc.tiles[3]["state"][(regs.MB_HOST_GONE - regs.LIM_BASE) // 8] == K.FakeEcc.POISON
                if l38:
                    assert at["addr"] == hex(regs.MB_HOST_GONE) and at["where"] == "inside the range"
                assert D.lim_addr_info(regs.MB_HOST_GONE)["where"] == ("mailbox L38 (MB_HOST_GONE: the "
                                                                       "SET_NOC_CLEANUP word)")
            else:
                assert init["ecc_delta"]["dat_fail_count"] == init["ecc_delta"]["dat_fix_count"] == 0
                assert c["pass"] and c["failed_checks"] == [] and c["ecc_delta"]["dat_fail_at"] is None
        finally:
            s.close()


def test_lim_init_store_option_and_sim_ecc_cli(hold, capsys, tmp_path):
    s = _no_release(hold, "granule")
    try:
        r = D.run_admin(s, "lim_init", method="A", lo=regs.S1D_BASE, hi=regs.S1D_BASE + 64 * 64, seed=3, store="uc8")
        assert r["store"] == "uc8" and r["lines"] == 64 and r["ecc_delta"]["dat_fail_count"] == 0
        c = D.run_admin(s, "lim_check", lo=regs.S1D_BASE, hi=regs.S1D_BASE + 64 * 64, seed=3, partial=8)
        assert c["pass"], c["failed_checks"]
        with pytest.raises(ValueError, match="method A"):
            s.lim_init(3, method="B", store="uc8")
    finally:
        s.close()
    assert D.main(["lim-ecc", "--backend", "sim", "--sim-ecc", "poison", "--lines", "32", "--plan", "a64,uc8",
                   "--ledger", str(hold)]) == 0
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rep["pass"] and rep["clean_variants"] == [] and [v["name"] for v in rep["variants"]] == ["a64", "uc8"]
    assert D.main(["lim-ecc", "--sim-ecc", "poison", "--ledger", str(hold)]) == 2  # device backend: refused
    for argv in (["boot", "--dry-run"], ["serve", "--backend", "sim", "--port", "0"]):  # a sim hart: not modelled
        assert D.main([*argv, "--sim-ecc", "poison", "--ledger", str(hold)]) == 2, argv
    with pytest.raises(ValueError, match="without a sim hart"):
        D.Session("sim", ledger_path=hold, sim_ecc="poison")
    assert D.main(["lim-init", "--backend", "sim", "--store", "wc2x32", "--lo", hex(regs.S1D_BASE), "--hi",
                   hex(regs.S1D_BASE + 4096), "--ledger", str(hold)]) == 0
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rep["store"] == "wc2x32" and rep["lines"] == 64


def test_serve_spin_answers_like_blocking_and_reports_it(daemon):
    """serve --spin (T-3.5 2026-10-25: client RTT p99 3.25 ms against x280d's 0.63 ms) busy-polls the REP socket and
    yields per miss; the replies are the same, info says so, and health/status still run between requests."""
    d = daemon("--spin")
    info = d.rpc("info")
    assert info["spin"] is True and info["mode"] == "booted"
    ref = _ref(D.BLOB_FP16)
    rng = np.random.default_rng(7)
    for i, obs in enumerate(_rng_obs(rng, 50)):
        rep = d.act((BAL, WALK)[i & 1], obs, tick=i, seq=i)
        assert "error" not in rep, rep
        np.testing.assert_array_equal(rep["action"][0], ref(i & 1, 2, obs))
    assert d.admin("status")["counts"]["ok"] == 50
    assert d.rpc("info")["spin"] is True


# --- harts 1-3: --harts 4, harts adopt, bench --mh (the 4-hart bench, S3b) --------------------------------------------


@pytest.fixture
def mh_env(monkeypatch, hold):
    monkeypatch.setenv("X280_E1F_APPROVED", "1")  # sim only: the chip is a FakeDevice
    monkeypatch.setenv("X280_MH_APPROVED", "1")
    return hold


def _mh_session(hold, harts=4, **kw):
    s = D.Session("sim", ledger_path=hold, harts=harts, sim_cpus=[_sim_cpu()], **kw)
    s.boot_stage0()
    with s.batch(3):
        s.load_blob(3)
    return s


def test_harts_flag_refusals(monkeypatch, hold, capsys):
    """--harts 4 only with --stage0 and X280_MH_APPROVED=1 (exit 2 before any Session), only for L2CPU3; Session-level
    checks repeat it (method A refuses, boot_stage0 refuses without the approval)."""
    monkeypatch.setenv("X280_E1F_APPROVED", "1")
    monkeypatch.delenv("X280_MH_APPROVED", raising=False)
    assert D.main(["boot", "--tile", "3", "--stage0", "--harts", "4", "--dry-run"]) == 2
    assert "X280_MH_APPROVED" in capsys.readouterr().err
    monkeypatch.setenv("X280_MH_APPROVED", "1")
    assert D.main(["boot", "--tile", "3", "--harts", "4", "--dry-run"]) == 2
    assert "--stage0" in capsys.readouterr().err
    port = _free_port()
    for extra, env in ((["--harts", "4"], {}), (["--stage0", "--harts", "4"], {"X280_MH_APPROVED": None})):
        e = {k: v for k, v in _env().items() if k not in env}
        r = subprocess.run(NICE + [PY, str(X280D), "serve", "--backend", "sim", "--port", str(port), *extra],
                           cwd=RT, env=e, capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and ("--stage0" in r.stderr or "X280_MH_APPROVED" in r.stderr), r.stderr
    with pytest.raises(ValueError, match="L2CPU3"):
        D.Session("sim", tiles=(2,), harts=4, ledger_path=hold)
    s = D.Session("sim", ledger_path=hold, harts=4)
    try:
        with pytest.raises(D.X280Error, match="stage-0"):
            s.boot()
        monkeypatch.delenv("X280_MH_APPROVED")
        with pytest.raises(D.X280Error, match="X280_MH_APPROVED"):
            s.boot_stage0()
        assert not [o for o in s.k.oplog if o.kind in ("W32", "WC64")]
    finally:
        s.close()


def _dry(args, env) -> dict:
    r = subprocess.run(NICE + [PY, str(X280D), *args], cwd=RT, env={**_env(), **env}, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return parse_dry_run(r.stdout)


def test_stage0_dry_run_with_harts4_differs_only_in_the_secondary_vectors():
    """The flag-off boot is the standard one; with --harts 4 the only difference in what is written is harts 1-3's RNMI
    handler (0x2001_0100 -> 0x0800_0000) and exception handler (0x2001_0130 -> 0x0800_0200); hart 0's, the reset
    vectors, the GP scratch, the release, PLL4, LIM and the mailbox writes are the same ops in the same order."""
    env = {"X280_E1F_APPROVED": "1", "X280_MH_APPROVED": "1"}
    base = _dry(["boot", "--tile", "3", "--stage0", "--dry-run"], env)
    mh = _dry(["boot", "--tile", "3", "--stage0", "--harts", "4", "--dry-run"], env)
    ws = (lambda d: [(o["kind"], o["xy"], o["addr"], o["value"]) for o in d["ops"] if o["kind"] in ("W32", "WC64")])
    a, b = ws(base), ws(mh)
    assert len(a) == len(b) and [x[:3] for x in a] == [x[:3] for x in b]
    diff = {x[2]: (x[3], y[3]) for x, y in zip(a, b, strict=True) if x != y}
    want = {}
    for h in (1, 2, 3):
        want[regs.rnmi_handler(h)] = (regs.PARK_STUB, regs.RESET_PC)
        want[regs.rnmi_exc_handler(h)] = (regs.RNMI_EXC_STUB, regs.MH_SEC_RNMI_EXC)
    assert diff == want
    assert [s for s in base["steps"]] == [s.split("; harts 1-3")[0] for s in mh["steps"]]
    assert not [o for o in mh["ops"] if o["kind"] == "REFUSED"]


def test_harts_adopt_bench_mh_and_status_on_the_sim(mh_env):
    s = _mh_session(mh_env)
    try:
        with s.batch(3):
            assert s.mb[3].l36()["mh"] == ml.MH_CAP | 1 and not any(d["valid"] for d in s.mh_lines(3)["harts"].values())
        a = s.harts_adopt(3)
        assert a["pass"] and (a["adopted"], a["lost"]) == (0xE, 0), a
        for h in ("1", "2", "3"):
            d = a["harts"][h]
            assert (d["state"], d["entries"], d["mnepc"]) == (ml.MH_STATE["IDLE"], 1, regs.STAGE0_SEC_WFI) and d["heartbeat_moving"]
            assert d["beu"]["accrued"] == 0
        assert a["ping"]["done"] == 0xE and a["ping"]["l36_mh"] == ml.MH_CAP | 0xF
        trig = [o.value for o in s.k.oplog if o.kind == "W32" and o.addr == regs.RNMI_TRIGGER]
        assert trig == [0xE, 0]                                            # one pulse, bit 0 never with them
        with s.batch(3):
            st = D.run_admin(s, "status")
        assert st["tiles"]["3"]["mh"]["mh"]["adopted"] == 0xE
        b = s.bench_mh(3, plan="quick", repeats=1, bench_iters=20)
        assert b["pass"] and b["abort"] is None, b.get("abort")
        assert [c["name"] for c in b["configs"]] == [c["name"] for c in D.Session.mh_plan("quick")]
        assert all(c["runs"][0]["done"] == c["mask"] for c in b["configs"])
        dv = b["derived"]
        assert set(dv) >= {"scaling", "S4_K9", "compute_bound", "split_worth_it", "verdict", "rule"}
        assert dv["S4_K9"] is not None and dv["meaningful"] is False and "K9/P0" in dv["scaling"]
        assert b["interference"]["background"]["dispatched"] == 0xE and b["parity"]["pass"]
        assert all(v["heartbeat_moving"] for v in b["secondaries_after"].values())
        assert set(b["events_before"]["beu_accrued"]) == {"0", "1", "2", "3"}
        with pytest.raises(D.X280Error, match="mh_bench payload"):
            s.bench(3, 0, 1100, out_addr=D.HOST_BENCH_BASE)                # would run into the payload now
        with s.batch(3):
            s.load_blob(3)                                                 # quiesce saw the secondaries idle
        assert s.check_parity()["pass"]
    finally:
        s.close()


def test_adopt_refused_without_the_secondary_vectors_and_the_sim_strands_them(mh_env):
    """A standard stage-0 session: harts adopt is refused before any RNMI write (kmd: no --harts 4 vectors). Forcing the
    trigger bits on that chip anyway (the sim's hardware) strands harts 1-3: the stale stage-0 line in their L1I sends
    them back to wfi with NMIE = 0, which is what the guard prevents."""
    s = _mh_session(mh_env, harts=1)
    try:
        a = s.harts_adopt(3)
        assert not a["pass"] and "--harts 4" in a["why"]
        assert not [o for o in s.k.oplog if o.kind == "W32" and o.addr == regs.RNMI_TRIGGER]
        with s.batch(3):
            with pytest.raises(K.PolicyError, match="secondary RNMI vectors"):
                s.k.rnmi_set(3, 0b0010)
        s.chip.poke32(regs.L2CPU_XY[3], regs.RNMI_TRIGGER, 0b1110)
        time.sleep(0.05)
        s.chip.poke32(regs.L2CPU_XY[3], regs.RNMI_TRIGGER, 0)
        time.sleep(0.05)
        assert {h: v["state"] for h, v in s.chip.hart(3).status()["harts"].items()} == {"1": "stranded", "2": "stranded",
                                                                                          "3": "stranded"}
        assert s.check_parity()["pass"]                                    # hart 0 unaffected
    finally:
        s.close()


def test_hung_secondary_recovered_by_its_rnmi_bit(mh_env):
    s = _mh_session(mh_env)
    try:
        assert s.harts_adopt(3)["pass"]
        s.chip.hart(3).inject_secondary_hang(2)
        time.sleep(0.01)
        with s.batch(3):
            p = s.mh_ping(3, timeout_us=500)
        assert p["done"] == 0xA                                            # hart 2 dispatched, never answered
        r = s.harts_adopt(3, mask=0x4)                                     # its own RNMI bit: crt0 -> re-adoption
        assert r["pass"] and r["harts"]["2"]["entries"] == 2, r
        with s.batch(3):
            assert s.mh_ping(3)["done"] == 0xE
        lost = s.harts_adopt(3, mask=0x2)
        assert lost["pass"]                                                # an adopted hart may be re-adopted
    finally:
        s.close()


def test_serve_harts4_adopt_and_bench_mh_through_the_admin_endpoint(daemon, monkeypatch):
    """`serve --stage0 --harts 4` on the sim, then the S3b steps as jobs/s3b_mh.sh sends them (--via)."""
    monkeypatch.setenv("X280_E1F_APPROVED", "1")
    monkeypatch.setenv("X280_MH_APPROVED", "1")
    d = daemon("--stage0", "--harts", "4")

    def via(*args, timeout=300):
        r = subprocess.run(NICE + [PY, str(X280D), *args, "--via", d.url], cwd=RT, env=_env(), capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, json.loads(r.stdout.strip().splitlines()[-1])
    rc, st = via("status")
    assert rc == 0 and st["tiles"]["3"]["mh"]["l36_mh"] & ml.MH_CAP
    rc, a = via("harts", "adopt")
    assert rc == 0 and a["pass"] and a["adopted"] == 0xE, a
    rc, b = via("bench", "--mh", "--mh-plan", "quick", "--mh-repeats", "1", "--mh-bench-iters", "20")
    assert rc == 0 and b["pass"] and b["derived"]["verdict"], b.get("abort")
    rc, p = via("harts", "ping")
    assert rc == 0 and p["done"] == 0xE
    act = d.act(BAL, np.zeros(516, np.float32))
    assert "error" not in act                                              # still serving on hart 0
