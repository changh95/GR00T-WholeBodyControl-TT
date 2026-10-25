# GR00T-WholeBodyControl-TT (rtloop)

A VLA and an RL whole-body policy running together on one Tenstorrent Blackhole p150a, driving a Unitree G1 in
MuJoCo (GR00T-WholeBodyControl, apple-to-plate). GR00T N1.6 runs on the Tensix grid; the gear_wbc lower-body MLP
runs every 50 Hz tick on the chip's on-die SiFive X280 RISC-V cores. The control loop is shaped like a real robot's:
the WBC runs every tick while the VLA produces 30-row action chunks asynchronously in the background.

## Demo

[![G1 apple-to-plate: GR00T N1.6 on Blackhole Tensix, gear_wbc WBC on the on-die X280](docs/media/demo_seed2000.gif)](https://github.com/changh95/GR00T-WholeBodyControl-TT/releases/download/demo-videos/g1_tensix_vla_x280_wbc_seed2000.mp4)

One Blackhole p150a runs the whole stack: GR00T N1.6 on the Tensix grid, the gear_wbc whole-body MLP on the chip's
X280 (L2CPU3 hart 0, 800 MHz, v3 kernel, 0.278 ms per call), the 50 Hz loop on the host. Left: the MuJoCo viewer;
right: the live HUD (processor map, Blackhole panel with the Tensix megakernel layout and the X280 status: serving rate,
heartbeat rate, per-call time; ego view of the active chunk). Full-resolution recordings (1920x1080):
[seed 2000, success at tick 854](https://github.com/changh95/GR00T-WholeBodyControl-TT/releases/download/demo-videos/g1_tensix_vla_x280_wbc_seed2000.mp4),
[seed 2002, tick 667](https://github.com/changh95/GR00T-WholeBodyControl-TT/releases/download/demo-videos/g1_tensix_vla_x280_wbc_seed2002.mp4),
[seed 2009, tick 1283](https://github.com/changh95/GR00T-WholeBodyControl-TT/releases/download/demo-videos/g1_tensix_vla_x280_wbc_seed2009.mp4).
The checkpoint succeeds in about 20-30% of episodes in this env on any backend (M3.md); these seeds succeed every run.

| Stage | Status |
|---|---|
| M3: VLA on Tensix, WBC on the X280 (L2CPU3) | done: closed loop 21/21 (T-3.7), success rate no lower than the baseline's, fault recovery (T-3.8), 107 min soak with 0 WBC fallbacks (T-3.9) |
| next: the WBC on 4 X280 tiles x up to 4 harts | single-hart kernel done (v3, 0.278 ms); S5a: all four L2CPU tiles probe clean; tensor-parallel firmware not yet written |

The simulator (MuJoCo physics and its camera renders) runs on the host; no policy runs there. The baselines the
M3 numbers are compared against ran the VLA and the WBC on a GPU (M1.md, M2.md); their code is at the git tag
`gpu-baselines`. Results:
`docs/plan/vla-wbc-x280/results/`. Design and test plan: `docs/plan/vla-wbc-x280/PLAN.md`.
Module contracts: `INTERFACES.md`. Test suites and how to run them: `TESTING.md`.

## Layout

- `rtloop/` the 50 Hz loop (splicer, VLA/WBC clients, telemetry, viewer process); `rtserve/` the mock VLA used by the
  tests and the numpy gear_wbc reference
- `rttt/` the TT VLA server (`server_g1.py`) on the tt-metal megakernel GR00T port
- `x280/` the X280 side: firmware (`fw/`: crt0, monitor, kernels, stage-0 LIM initialiser, multi-hart runtime),
  host layer (`host/kmd.py`, `host/wcio.c`), register map and write allowlist (`regs.py`), daemon (`x280d.py`), simulators
- `scripts/` demo, HUD, chip sessions, analysis; `jobs/` device-session jobs; `tests/` CPU, sim, spike and fake-device suites
- `third_party_refs/SHAS.txt`, `x280/refs/SOURCES.txt`: the upstream sources the code cites (fetch them; not vendored)
- `docs/plan/vla-wbc-x280/`: plan, results (M1-M3), research; `docs/media/`: the README GIF (the MP4s are on the release)
- not tracked: `runs/` (every session's logs, ticks and recordings), `snapshots/` (frozen worktrees for device sessions)
- the day-1 prototypes (`prototypes/`: first experiments and their captured outputs) are in git history at `c1d5b90`

## Setup

Clone to `~/experiments/GR00T-WholeBodyControl-TT`. `source env.sh` defines the interpreter wrappers (`simpy`,
`srvpy`, `ttpy`, `xpy`), ports and paths; `RT` is the tree `env.sh` lives in. The external inputs (Isaac-GR00T N1.6
and its sim venv, the tt-metal megakernel snapshot, goldens, the device guard) stay under `~/experiments/gr00t/` and are
named in `env.sh`. Device access goes only through the guarded wrappers `$DEV` / `$DEVX` (`chip_session.sh`); every X280
hold resets the chip after. Device sessions run from a frozen worktree: `scripts/snapshot.sh NAME`, then
`source snapshots/NAME/env.sh` and `$DEVX $RT/scripts/chip_session.sh ...`.

## Findings worth knowing

- Host NoC writes cannot initialise X280 LIM: the CCACHE pre-reads uninitialised SRAM and poisons the ECC, and the
  poison survives `tt-smi -r`. A hart-side stage-0 zero-fill (E1F) fixes it; it is the standard boot.
- One X280 hart streams LIM at ~6.4 GB/s, the tile's whole read port, so more harts on one tile do not help (4 harts:
  1.94x on a bandwidth proxy); the v3 MLP is limited by the hart's vector pipe, not by LIM.
- The GP scratch the RNMI handler runs from is cached in the hart's L1I, which only the hart's own `fence.i` clears:
  without one after the stage-0 boot, the first RNMI re-ran stage 0 and wiped LIM (T-3.8, fixed in crt0 and x280d).
- The tt-metal snapshot's `ttnn` must run with `TT_METAL_RUNTIME_ROOT` pinned, or the device compiles another tree's kernels.
