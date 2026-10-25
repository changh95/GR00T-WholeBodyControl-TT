"""Synthetic left-drag across the MuJoCo viewer (XTest), restoring the pointer afterwards."""

import time

from Xlib import X, display
from Xlib.ext import xtest

d = display.Display()
root = d.screen().root
clients = root.get_full_property(d.intern_atom("_NET_CLIENT_LIST"), X.AnyPropertyType).value
win = next(
    w
    for w in (d.create_resource_object("window", i) for i in clients)
    if (w.get_wm_name() or "").startswith("MuJoCo")
)
geom = win.get_geometry()
pos = win.translate_coords(root, 0, 0)
x0, y0 = -pos.x + geom.width // 2, -pos.y + geom.height // 2
print(f"window {hex(win.id)} {geom.width}x{geom.height} center ({x0},{y0})")

saved = root.query_pointer()
xtest.fake_input(d, X.MotionNotify, x=x0 - 300, y=y0)
d.sync()
time.sleep(0.2)
xtest.fake_input(d, X.ButtonPress, 1)
d.sync()
for i in range(31):
    xtest.fake_input(d, X.MotionNotify, x=x0 - 300 + 20 * i, y=y0)
    d.sync()
    time.sleep(0.02)
xtest.fake_input(d, X.ButtonRelease, 1)
xtest.fake_input(d, X.MotionNotify, x=saved.root_x, y=saved.root_y)
d.sync()
print("drag done")
