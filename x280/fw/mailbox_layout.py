"""Python mirror of the X280 LIM mailbox layout v2 (x280/fw/mailbox.h, PLAN.md §1.4.4).

Why: x280d, mailbox_sim.py, the spike harness and the tests all pack and parse the same 64 B lines; one
mirror keeps them consistent, and tests/x280/test_mailbox_x86.py checks every offset here against the
compiled C layout (mon_x86_layout_*()). Pure stdlib (struct, zlib): importable from any venv.
"""
from __future__ import annotations

import struct
import zlib

LIM_BASE = 0x0800_0000
LIM_SIZE = 0x001E_0000
LIM_END = LIM_BASE + LIM_SIZE
LINE = 64
IMAGE_MAX = 48 * 1024
CODE_END = 0x0801_0000
MB_BASE = 0x0801_0000
PAYLOAD_BASE = 0x0801_1000

MAGIC = 0x32434257            # bytes "WBC2"
LAYOUT_VERSION = 2
FW_REV = 1
FW_VER = (LAYOUT_VERSION << 16) | FW_REV
HOST_GONE = 0x474F4E45
CRC_RETRY_US = 50
L0_CRC_BYTES = 44
L37_CRC_BYTES = 60
OBS_LINES = 33
OBS_DIM = 516
ACT_DIM = 15

CMD = dict(NOP=0, INFER=1, LOAD=2, FENCE_I=3, CALL=4, PARK=5, SELFTEST=6, BENCH=7, MH=8)
STATE = dict(BOOT=0, IDLE=1, BUSY=2, FAULT=3, REENTERED=4, HOST_GONE=5, PARKED=6)
ERR = dict(CRC=0xC0C, MAGIC=0xC0D, CMD=0xC0E, ARG=0xC0F, KERNEL=0xC10, BLOB=0xC11, LOAD_CRC=0xC12,
           NOCODE=0xC13, DESC=0xC14)
ARG = dict(RANGE=1, ALIGN=2, POLICY=3, KIND=4, LEN=5, OVERLAP=6)
LOAD_KIND = dict(CODE=0, BLOB=1, DATA=2)
L39_KIND = dict(SELFTEST=1, BENCH=2, CALL=3, ENTRY=4, MH=5)
ENTRY_FLAG = dict(WARM=1, RELOADED=2, STUB_KERNEL=4)
FLAG_HOST_ALIVE = 1 << 0
FLAG_VARIANT_SHIFT = 8
FLAG_VARIANT_MASK = 0xF
FLAG_VARIANT_SET = 1 << 12
MH_OP = dict(PING=0)          # MB_CMD_MH L0.policy
MH_TIMEOUT_US = 1000
MH_CAP = 1 << 8               # L36.mh bit: the image has the secondary path, mh.c and MB_CMD_MH
MH_READY_MASK = 0xF
MON_API_VERSION = 2


def line_addr(n: int) -> int:
    return MB_BASE + LINE * n


LINES = dict(L0=0, L1=1, L2=2, L35=35, L36=36, L37=37, L38=38, L39=39, L40=40, L42=42)
ADDR = {k: line_addr(v) for k, v in LINES.items()}

# struct formats, little-endian, each exactly one line
L0_FMT = "<IIIIqQQIII8xI"        # magic cmd policy flags tick arg_addr arg_entry arg_len crc32 obs_crc32 . seq_req
L0_FIELDS = ("magic", "cmd", "policy", "flags", "tick", "arg_addr", "arg_entry", "arg_len", "crc32",
             "obs_crc32", "seq_req")
L1_FMT = "<QIIQQQIIIIII"
L1_FIELDS = ("heartbeat", "seq_done", "state", "mcause", "mepc", "mtval", "cycles_last", "cycles_max",
             "fw_ver", "mhartid", "vlenb", "fw_magic")
L35_FMT = "<15fI"
L36_FMT = "<Q14I"
L36_FIELDS = ("misa", "mtime_hz", "build_id", "kernel_variant", "n_runs", "n_overrun", "n_reentry",
              "beu_accrued", "n_traps", "n_crc_fail", "variants_built", "n_host_gone", "entry_flags",
              "n_crc_retry", "mh")
L37_FMT = "<QIIII36xI"
L37_FIELDS = ("dst", "len", "crc32", "kind", "load_seq", "desc_crc32")
L39_HDR_FMT = "<IIQQ"
L39_FIELDS = ("kind", "seq", "ret", "cycles")
L39_ENTRY_FMT = "<IIQQQQ"
L39_ENTRY_FIELDS = ("entry_kind", "mnstatus", "mncause", "mnepc", "mcycle", "mtime")
L39_SELFTEST_FMT = "<IIIIQIIII"
L39_SELFTEST_FIELDS = ("lines", "cyc_per_mtime_x1000", "beu_cause", "beu_accrued_before", "beu_value",
                       "beu_accrued_after", "misa_v", "vlenb", "mtime_ticks")
L39_BENCH_FMT = "<IIIIQIiII"
L39_BENCH_FIELDS = ("iters", "cyc_min", "cyc_max", "nondet", "cyc_sum", "act_crc32", "rc", "variant", "policy")
LB_RESULT_FMT = "<IIQQQQ"       # lim_bench result in L39 bytes 24..63
LB_RESULT_FIELDS = ("test", "status", "iters", "bytes", "cycles", "ticks")
L39_MH_FMT = "<IIII3IIII"       # MB_CMD_MH ping record in L39 bytes 24..63 (mailbox.h struct mb_l39_mh)
L39_MH_FIELDS = ("op", "mask", "done", "gen", "wake_ticks", "total_ticks", "ready", "trapped")

# --- multi-hart region (x280/fw/mh.h, asm.h; lim.ld .mh NOLOAD at a fixed address; X280 and x86 geometry) ---------
MH_BASE = 0x0800_A000
MH_SIZE = 0x2000
MH_END = MH_BASE + MH_SIZE
MH_NHARTS = 4
MH_SEC_MASK = 0xE
MH_SEC_RNMI_EXC = 0x0800_0200   # RNMI exception trap handler address of harts 1-3 under --harts 4 (crt0.S .org)
MH_CTL_MAGIC = 0x4C54434D       # "MCTL"
MH_STAT_MAGIC = 0x5441544D      # "MSTA"
MH_STATE = dict(OFF=0, STARTING=1, IDLE=2, BUSY=3, TRAPPED=4)
MH_MODE = dict(WFI=1, POLL=2)
MIP_MSIP = 1 << 3


def mh_stat_addr(h: int) -> int:
    return MH_BASE + 0x80 * h


def mh_aux_addr(h: int) -> int:
    return MH_BASE + 0x80 * h + 0x40


MH_CTL = MH_BASE
MH_TRAP_TOP = {h: MH_BASE + 0x400 + 0x100 * h for h in (1, 2, 3)}
MH_PRIV = {h: MH_BASE + 0x800 + 0x800 * (h - 1) for h in (1, 2, 3)}
MH_STACK_TOP = {h: MH_BASE + 0x800 + 0x800 * h for h in (1, 2, 3)}
MH_CTL_FMT = "<IIIIQQQI20x"
MH_CTL_FIELDS = ("magic", "epoch", "gen", "mask", "fn", "arg", "t_dispatch", "n_dispatch")
MH_STAT_FMT = "<IIIIQIIIIQQII"
MH_STAT_FIELDS = ("magic", "hartid", "epoch", "state", "heartbeat", "entries", "done_gen", "trap_gen", "traps",
                  "mncause", "mnepc", "mode", "mip")
MH_AUX_FMT = "<QQQQQIIIII4x"
MH_AUX_FIELDS = ("mcause", "mepc", "mtval", "result", "t_seen", "gen_seen", "wakes", "beu_accrued", "wfi_ticks",
                 "fence_i")

# --- mh_bench payload (x280/fw/mh_bench.c, mh_bench.ld; mh.h MHB_*) ---------------------------------------------
MHB_BASE = 0x081D_F400
MHB_DATA = 0x081D_FE40          # job line (hart 0)
MHB_END = 0x081D_FF80           # = the E1F entry-word line, never reached
MHB_MAX_BYTES = MHB_END - MHB_BASE


def mhb_res_addr(h: int) -> int:
    return MHB_DATA + 0x40 + 0x40 * h


MHB_K = dict(LD8X8=1, LD8=2, CHAIN=3, LW16=4, VLE8_M1=5, VLE8_M2=6, VLE8_M4=7, VLE8_M8=8, MLP=9, MLP_ALU=10, ALU=11,
             MLP_LD=12, TRAP=15)
MHB_N_KERNELS = 12
MHB_P = dict(PRIVATE=0, SAME_BANK=1, OWN_BANK=2, SHARED=3, SWEEP=4)
MHB_MODE = dict(RUN=0, START=1, COLLECT=2)
MHB_STATUS = dict(OK=0, EKERNEL=1, ERANGE=2, EPATTERN=3, EAPI=4, EWINDOW=5, ENOVECTOR=6, ENOJOB=7, EMODE=8)
MHB_WINDOW_US = 500
MHB_LEAD_US = 50
MHB_WS_DEFAULT_BASE = 0x0801_2000
MHB_WS_DEFAULT_LEN = 0x001C_0000
MHB_WS_ALIGN = 0x1_0000
# per kernel: unit footprint U, bytes loaded per access, accesses per ~4 KiB chunk (mh_bench.c KUNIT/KBPA, TIMED)
MHB_UNIT = {1: 64, 2: 64, 3: 64, 4: 64, 5: 64, 6: 128, 7: 256, 8: 512, 9: 128, 10: 128, 11: 64, 12: 128}
MHB_BPA = {1: 64, 2: 8, 3: 8, 4: 64, 5: 64, 6: 128, 7: 256, 8: 512, 9: 128, 10: 0, 11: 0, 12: 128}
MHB_CHUNK = {1: 64, 2: 64, 3: 64, 4: 64, 5: 64, 6: 32, 7: 16, 8: 8, 9: 32, 10: 32, 11: 64, 12: 32}
MHB_JOB_FMT = "<IIIIIIQQQQII"
MHB_JOB_FIELDS = ("tag", "kernel", "kernel0", "pattern", "kbit", "mask", "ws_base", "ws_len", "t_start", "t_end",
                  "mode", "dispatched")
MHB_RES_FMT = "<IIIIQQQQIIQ"
MHB_RES_FIELDS = ("hart", "status", "tag", "kernel", "bytes", "accesses", "cycles", "t0", "dt", "beu_accrued", "sink")
MHB_SUMMARY_FMT = "<IIIIQIIII"
MHB_SUMMARY_FIELDS = ("kernel", "status", "mask", "done", "bytes", "span", "start_skew", "end_skew", "tag")

for _fmt in (L0_FMT, L1_FMT, L35_FMT, L36_FMT, L37_FMT):
    assert struct.calcsize(_fmt) == LINE, _fmt
for _fmt in (L39_ENTRY_FMT, L39_SELFTEST_FMT, L39_BENCH_FMT, LB_RESULT_FMT, L39_MH_FMT, MHB_SUMMARY_FMT):
    assert struct.calcsize(_fmt) == 40, _fmt
for _fmt in (MH_CTL_FMT, MH_STAT_FMT, MH_AUX_FMT, MHB_JOB_FMT, MHB_RES_FMT):
    assert struct.calcsize(_fmt) == LINE, _fmt
assert struct.calcsize(L39_HDR_FMT) == 24

# Byte offsets within each line; the names match mon_x86_layout_name() in plat_x86.c.
OFFSETS = {
    "l0.magic": 0, "l0.cmd": 4, "l0.policy": 8, "l0.flags": 12, "l0.tick": 16, "l0.arg_addr": 24,
    "l0.arg_entry": 32, "l0.arg_len": 40, "l0.crc32": 44, "l0.obs_crc32": 48, "l0.seq_req": 60,
    "l1.heartbeat": 0, "l1.seq_done": 8, "l1.state": 12, "l1.mcause": 16, "l1.mepc": 24, "l1.mtval": 32,
    "l1.cycles_last": 40, "l1.cycles_max": 44, "l1.fw_ver": 48, "l1.mhartid": 52, "l1.vlenb": 56,
    "l1.fw_magic": 60,
    "l35.action": 0, "l35.seq_echo": 60,
    "l36.misa": 0, "l36.mtime_hz": 8, "l36.build_id": 12, "l36.kernel_variant": 16, "l36.n_runs": 20,
    "l36.n_overrun": 24, "l36.n_reentry": 28, "l36.beu_accrued": 32, "l36.n_traps": 36,
    "l36.n_crc_fail": 40, "l36.variants_built": 44, "l36.n_host_gone": 48, "l36.entry_flags": 52,
    "l36.n_crc_retry": 56,
    "l37.dst": 0, "l37.len": 8, "l37.crc32": 12, "l37.kind": 16, "l37.load_seq": 20, "l37.desc_crc32": 60,
    "l38.host_gone": 0,
    "l39.kind": 0, "l39.seq": 4, "l39.ret": 8, "l39.cycles": 16, "l39.data": 24,
    "call.policy": 0, "call.arg_addr": 16, "call.arg_len": 32, "call.seq": 36,
    "l36.mh": 60, "api.mh_ready": 64, "api.wapi": 88,
    "mh_ctl.epoch": 4, "mh_ctl.gen": 8, "mh_ctl.mask": 12, "mh_ctl.fn": 16, "mh_ctl.t_dispatch": 32,
    "mh_ctl.n_dispatch": 40,
    "mh_stat.state": 12, "mh_stat.heartbeat": 16, "mh_stat.entries": 24, "mh_stat.done_gen": 28,
    "mh_stat.trap_gen": 32, "mh_stat.traps": 36, "mh_stat.mncause": 40, "mh_stat.mnepc": 48, "mh_stat.mode": 56,
    "mh_stat.mip": 60,
    "mh_aux.result": 24, "mh_aux.t_seen": 32, "mh_aux.gen_seen": 40, "mh_aux.wakes": 44, "mh_aux.beu_accrued": 48,
    "mh_aux.fence_i": 56,
    "mhb_job.t_start": 40, "mhb_job.dispatched": 60, "mhb_res.tag": 8, "mhb_res.cycles": 32, "mhb_res.dt": 48,
    "mhb_res.sink": 56, "l39mh.wake_ticks": 16, "l39mh.trapped": 36,
}
CONSTANTS = {
    "MB_LIM_BASE": LIM_BASE, "MB_LIM_SIZE": LIM_SIZE, "MB_BASE": MB_BASE, "MB_PAYLOAD_BASE": PAYLOAD_BASE,
    "MB_CODE_END": CODE_END, "MB_IMAGE_MAX": IMAGE_MAX, "MB_MAGIC": MAGIC, "MB_FW_VER": FW_VER,
    "MB_HOST_GONE": HOST_GONE, "MB_CRC_RETRY_US": CRC_RETRY_US, "MB_L0_CRC_BYTES": L0_CRC_BYTES,
    "MB_L37_CRC_BYTES": L37_CRC_BYTES, "MB_OBS_LINES": OBS_LINES,
    "MB_FLAG_VARIANT_SHIFT": FLAG_VARIANT_SHIFT, "MB_FLAG_VARIANT_SET": FLAG_VARIANT_SET,
    **{f"line.{k}": v for k, v in ADDR.items()},
    "MB_CMD_MH": CMD["MH"], "MB_L39_MH": L39_KIND["MH"], "MB_MH_CAP": MH_CAP, "MB_MH_TIMEOUT_US": MH_TIMEOUT_US,
    "MON_API_VERSION": MON_API_VERSION,
    "MH_BASE": MH_BASE, "MH_SIZE": MH_SIZE, "MH_SEC_RNMI_EXC": MH_SEC_RNMI_EXC, "MH_CTL_MAGIC": MH_CTL_MAGIC,
    "MH_STAT_MAGIC": MH_STAT_MAGIC, "mh.ctl": MH_CTL, "mh.stat1": mh_stat_addr(1), "mh.aux1": mh_aux_addr(1),
    "mh.stat3": mh_stat_addr(3), "mh.trap_top1": MH_TRAP_TOP[1], "mh.priv1": MH_PRIV[1],
    "mh.stack_top1": MH_STACK_TOP[1], "mh.stack_top3": MH_STACK_TOP[3],
    "MH_ST_IDLE": MH_STATE["IDLE"], "MH_ST_BUSY": MH_STATE["BUSY"], "MH_MODE_WFI": MH_MODE["WFI"],
    "MH_MODE_POLL": MH_MODE["POLL"],
    "MHB_BASE": MHB_BASE, "MHB_DATA": MHB_DATA, "MHB_END": MHB_END, "mhb.res0": mhb_res_addr(0),
    "mhb.res3": mhb_res_addr(3), "MHB_WS_DEFAULT_BASE": MHB_WS_DEFAULT_BASE, "MHB_WS_DEFAULT_LEN": MHB_WS_DEFAULT_LEN,
    "MHB_WS_ALIGN": MHB_WS_ALIGN, "MHB_WINDOW_US": MHB_WINDOW_US, "MHB_LEAD_US": MHB_LEAD_US,
    "MHB_K_MLP": MHB_K["MLP"], "MHB_K_TRAP": MHB_K["TRAP"], "MHB_N_KERNELS": MHB_N_KERNELS,
}


def crc32(data: bytes, crc: int = 0) -> int:
    """zlib CRC-32, as the monitor computes it (reflected 0xEDB88320, init/final xor ~0)."""
    return zlib.crc32(data, crc) & 0xFFFFFFFF


def variant_flags(variant: int | None) -> int:
    """L0.flags bits selecting a wbc_infer variant (None = the image's default)."""
    if variant is None:
        return 0
    return FLAG_VARIANT_SET | ((variant & FLAG_VARIANT_MASK) << FLAG_VARIANT_SHIFT)


def pack_obs(obs) -> bytes:
    """L2-L34: 516 little-endian f32 plus 48 zero bytes (2,112 B, what obs_crc32 covers)."""
    if hasattr(obs, "astype"):
        raw = obs.astype("<f4").reshape(-1).tobytes()
    else:
        raw = struct.pack("<516f", *obs)
    assert len(raw) == OBS_DIM * 4, len(raw)
    return raw + bytes(OBS_LINES * LINE - len(raw))


def pack_l0(cmd: int, seq_req: int, policy: int = 0, flags: int = 0, tick: int = 0, arg_addr: int = 0,
            arg_entry: int = 0, arg_len: int = 0, obs_crc32: int = 0, magic: int = MAGIC,
            crc_override: int | None = None) -> bytes:
    """Full 64 B L0 with crc32 over bytes 0..43. The host writes bytes 0..59 as one line, then seq_req
    as its own 4 B write (§1.4.4 steps 1-2); seq_req is included here only so a full-line write keeps it."""
    head = struct.pack("<IIIIqQQI", magic, cmd, policy, flags, tick, arg_addr, arg_entry, arg_len)
    assert len(head) == L0_CRC_BYTES
    c = crc32(head) if crc_override is None else crc_override
    return struct.pack(L0_FMT, magic, cmd, policy, flags, tick, arg_addr, arg_entry, arg_len, c,
                       obs_crc32, seq_req)


def pack_l37(dst: int, length: int, data_crc32: int, kind: int, load_seq: int,
             desc_crc_override: int | None = None) -> bytes:
    body = struct.pack("<QIIII36x", dst, length, data_crc32, kind, load_seq)
    assert len(body) == L37_CRC_BYTES
    c = crc32(body) if desc_crc_override is None else desc_crc_override
    return body + struct.pack("<I", c)


def _unpack(fmt: str, fields, raw: bytes) -> dict:
    return dict(zip(fields, struct.unpack(fmt, bytes(raw))))


def unpack_l0(raw: bytes) -> dict:
    return _unpack(L0_FMT, L0_FIELDS, raw)


def unpack_l1(raw: bytes) -> dict:
    return _unpack(L1_FMT, L1_FIELDS, raw)


def unpack_l35(raw: bytes) -> dict:
    v = struct.unpack(L35_FMT, bytes(raw))
    return {"action": list(v[:15]), "seq_echo": v[15]}


def unpack_l36(raw: bytes) -> dict:
    return _unpack(L36_FMT, L36_FIELDS, raw)


def unpack_l37(raw: bytes) -> dict:
    return _unpack(L37_FMT, L37_FIELDS, raw)


def unpack_l39(raw: bytes) -> dict:
    raw = bytes(raw)
    d = _unpack(L39_HDR_FMT, L39_FIELDS, raw[:24])
    data = raw[24:64]
    d["data"] = data
    kind = d["kind"]
    if kind == L39_KIND["ENTRY"]:
        d.update(_unpack(L39_ENTRY_FMT, L39_ENTRY_FIELDS, data))
    elif kind == L39_KIND["SELFTEST"]:
        d.update(_unpack(L39_SELFTEST_FMT, L39_SELFTEST_FIELDS, data))
    elif kind == L39_KIND["BENCH"]:
        d.update(_unpack(L39_BENCH_FMT, L39_BENCH_FIELDS, data))
    elif kind == L39_KIND["MH"]:
        d.update(unpack_l39_mh(data))
    return d


def unpack_lb_result(data: bytes) -> dict:
    return _unpack(LB_RESULT_FMT, LB_RESULT_FIELDS, bytes(data)[:40])


def unpack_l39_mh(data: bytes) -> dict:
    v = struct.unpack(L39_MH_FMT, bytes(data)[:40])
    return {"op": v[0], "mask": v[1], "done": v[2], "gen": v[3], "wake_ticks": list(v[4:7]), "total_ticks": v[7],
            "ready": v[8], "trapped": v[9]}


def unpack_mh_ctl(raw: bytes) -> dict:
    return _unpack(MH_CTL_FMT, MH_CTL_FIELDS, raw)


def unpack_mh_stat(raw: bytes) -> dict:
    return _unpack(MH_STAT_FMT, MH_STAT_FIELDS, raw)


def unpack_mh_aux(raw: bytes) -> dict:
    return _unpack(MH_AUX_FMT, MH_AUX_FIELDS, raw)


def unpack_mhb_job(raw: bytes) -> dict:
    return _unpack(MHB_JOB_FMT, MHB_JOB_FIELDS, raw)


def unpack_mhb_res(raw: bytes) -> dict:
    return _unpack(MHB_RES_FMT, MHB_RES_FIELDS, raw)


def unpack_mhb_summary(data: bytes) -> dict:
    return _unpack(MHB_SUMMARY_FMT, MHB_SUMMARY_FIELDS, bytes(data)[:40])


def mhb_flags(mask: int, pattern: int = 0, kbit: int = 0, mode: int = 0) -> int:
    """CALL flags of mh_bench: hart mask [3:0] | pattern [7:4] | k [11:8] | mode [13:12]."""
    return (mask & 0xF) | (pattern & 0xF) << 4 | (kbit & 0xF) << 8 | (mode & 3) << 12


def mhb_stream(kernel: int, pattern: int, kbit: int, hart: int, base: int, length: int) -> tuple:
    """(lo, hi, stride) of hart `hart`'s unit stream (mh_bench.c stream()): addresses lo, lo + st, ... wrapping to lo
    at hi."""
    q, u = length // 4, MHB_UNIT[kernel]
    if kernel in (MHB_K["MLP"], MHB_K["MLP_ALU"], MHB_K["MLP_LD"]):
        return base + 128 * (0 if pattern == MHB_P["SHARED"] else hart), base + length, 512
    if pattern == MHB_P["PRIVATE"]:
        return base + hart * q, base + (hart + 1) * q, u
    if pattern == MHB_P["SAME_BANK"]:
        return base + hart * q, base + (hart + 1) * q, 4 * u
    if pattern == MHB_P["OWN_BANK"]:
        return base + hart * q + hart * u, base + (hart + 1) * q, 4 * u
    if pattern == MHB_P["SHARED"]:
        return base, base + q, u
    return base + (hart << kbit), base + length, 4 << kbit


def mhb_sink(kernel: int, pattern: int, kbit: int, hart: int, base: int, length: int, accesses: int, mem) -> int:
    """The sink mh_bench reports for `accesses` units of hart `hart` (mh_bench.c header): K1 the sum of the 8 words of
    each line, K2/K3 word 0, K4 the 16 u32 words, K5-K8 the sum over lanes of each lane's byte sum mod 256, else 0.
    `mem(addr, n)` returns n bytes of LIM at addr (what the working set held)."""
    if kernel not in (1, 2, 3, 4, 5, 6, 7, 8):
        return 0
    lo, hi, st = mhb_stream(kernel, pattern, kbit, hart, base, length)
    n_units = (hi - lo + st - 1) // st
    counts = [0] * n_units          # how often each unit of the stream was visited
    full, rest = divmod(accesses, n_units)
    for i in range(n_units):
        counts[i] = full + (1 if i < rest else 0)
    u = MHB_UNIT[kernel]
    if kernel in (5, 6, 7, 8):
        lanes = [0] * u
        for i, c in enumerate(counts):
            if c:
                b = mem(lo + i * st, u)
                for j in range(u):
                    lanes[j] += c * b[j]
        return sum(x % 256 for x in lanes)
    total = 0
    for i, c in enumerate(counts):
        if not c:
            continue
        b = mem(lo + i * st, 64)
        if kernel == 1:
            v = sum(struct.unpack("<8Q", b))
        elif kernel == 4:
            v = sum(struct.unpack("<16I", b))
        else:
            v = struct.unpack_from("<Q", b)[0]
        total += c * v
    return total & 0xFFFF_FFFF_FFFF_FFFF
