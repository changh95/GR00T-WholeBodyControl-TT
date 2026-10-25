import numpy as np, sys
rng = np.random.default_rng(0)
dims = dict(left_leg=6, right_leg=6, waist=3, left_arm=7, right_arm=7, left_hand=7, right_hand=7)
# smooth-ish synthetic frame (random noise is a worst case for resampling parity; add a gradient frame too)
img = rng.integers(0, 255, (1, 480, 640, 3), dtype=np.uint8)
yy, xx = np.mgrid[0:480, 0:640]
grad = np.stack([(xx * 255 // 639), (yy * 255 // 479), ((xx + yy) * 255 // 1118)], -1).astype(np.uint8)[None]
st = {k: rng.uniform(-0.8, 0.8, (1, d)).astype(np.float32) for k, d in dims.items()}
pred = rng.uniform(-1, 1, (1, 50, 128)).astype(np.float32)
np.savez(sys.argv[1], img_noise=img, img_grad=grad, pred=pred, **{"state." + k: v for k, v in st.items()})
