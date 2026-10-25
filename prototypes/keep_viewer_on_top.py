"""Marks every new MuJoCo viewer window always-on-top (without taking keyboard focus).

robosuite destroys and relaunches the viewer on each episode reset, and the new window opens
behind the focused terminal, so this keeps watching for new ones.
"""

import time

from Xlib import X, display
from Xlib.protocol import event

d = display.Display()
root = d.screen().root
NET_CLIENT_LIST = d.intern_atom("_NET_CLIENT_LIST")
NET_WM_STATE = d.intern_atom("_NET_WM_STATE")
NET_WM_STATE_ABOVE = d.intern_atom("_NET_WM_STATE_ABOVE")
NET_WM_STATE_ADD = 1
SOURCE_PAGER = 2


def mujoco_windows():
    prop = root.get_full_property(NET_CLIENT_LIST, X.AnyPropertyType)
    for wid in prop.value if prop else []:
        w = d.create_resource_object("window", wid)
        try:
            name = w.get_wm_name()
        except Exception:
            continue
        if name and name.startswith("MuJoCo"):
            yield w


seen = set()
while True:
    for w in mujoco_windows():
        if w.id not in seen:
            seen.add(w.id)
            data = [NET_WM_STATE_ADD, NET_WM_STATE_ABOVE, 0, SOURCE_PAGER, 0]
            ev = event.ClientMessage(window=w, client_type=NET_WM_STATE, data=(32, data))
            root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
            d.flush()
            print(f"on top: {hex(w.id)}", flush=True)
    time.sleep(0.5)
