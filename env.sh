# Source this file: `source ~/experiments/gr00t/rtloop/env.sh`.
#
# Paths, ports and the interpreter wrappers used by every rtloop command (PLAN.md §0 "Interpreters
# and wrappers"). The sim side (simpy) must never create a CUDA context: doing so corrupts its EGL
# renders, so CUDA is hidden there and EGL is pointed at GPU 0 explicitly (robosuite's EGL init
# crashes on an empty CUDA_VISIBLE_DEVICES without MUJOCO_EGL_DEVICE_ID).

export RT=$HOME/experiments/gr00t/rtloop
export R=$HOME/experiments/gr00t/Isaac-GR00T-n1.6
export W=$R/external_dependencies/GR00T-WholeBodyControl
export DEV=$HOME/experiments/gr00t/bin/with-device.sh
export DEVX="env WITH_DEVICE_RESET_AFTER=1 $DEV"
export ONNX=$W/gr00t_wbc/sim2mujoco/resources/robots/g1/policy

export SIM_PYTHON=$R/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python
export SRV_PYTHON=$R/.venv/bin/python

export VLA_GPU=tcp://127.0.0.1:5565
export VLA_TT=tcp://127.0.0.1:5557
export VLA_MOCK=tcp://127.0.0.1:5559
export WBC_GPU=tcp://127.0.0.1:5556
export WBC_X280=tcp://127.0.0.1:5558
export TELEMETRY_PUB=tcp://127.0.0.1:5570
export CONTROL_PULL=tcp://127.0.0.1:5571
# Out-of-band device status PUBs read by the HUD chip panel (PLAN.md §4.3 item 7); both off unless --status-pub.
export X280_STATUS=tcp://127.0.0.1:5573
export TT_STATUS=tcp://127.0.0.1:5574

# Default demo checkpoint (PLAN.md D11 / DP1).
export VLA_CHECKPOINT=cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate

simpy() {
    (cd "$R" && env -u GR00T_WBC_SERVER MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES= \
        PYTHONPATH="$RT:$R${PYTHONPATH:+:$PYTHONPATH}" "$SIM_PYTHON" "$@")
}

srvpy() {
    (cd "$R" && env PYTHONPATH="$RT:$R${PYTHONPATH:+:$PYTHONPATH}" "$SRV_PYTHON" "$@")
}

# TT side (PLAN.md D10/DP6): the frozen megakernel snapshot MK, its tt-metal python_env, plus RT.
# Opening the device is only allowed under $DEV/$DEVX.
# TT_METAL_RUNTIME_ROOT=$MK unless already set (S1, 2026-10-25): without it ttnn takes its root from the resolved
# ttnn symlink, i.e. the live clone, and compiles the live tree's kernels against MK's host code. With it every MK
# process run through ttpy compiles MK's own kernels; rttt/g1_n16.pin_runtime_root accepts this root (and refuses
# any other preset one), and the worktree-mk-env.sh it sources does not set it.
export MK=$HOME/experiments/gr00t/publish/worktree-mk
export TT=$HOME/experiments/gr00t/tt-metal/models/experimental/gr00t
ttpy() {
    (source "$HOME/experiments/gr00t/publish/worktree-mk-env.sh" && cd "$RT" && \
        export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$MK}" && \
        PYTHONPATH="$PYTHONPATH:$RT" python "$@")
}

# RISC-V emulation before silicon (WP-3.2b): spike built in user space by x280/sim/build_spike.sh. run_spike.py
# finds spike relative to itself, so these are conveniences for job scripts.
export SPIKE=$RT/x280/sim/spike-install/bin/spike
export RUN_SPIKE=$RT/x280/sim/run_spike.py

# X280 daemon side (PLAN.md D12): numpy + pyzmq + msgpack only, no pyluwen.
xpy() {
    (cd "$RT" && PYTHONPATH="$RT" "$RT/.venv-x280/bin/python" "$@")
}
