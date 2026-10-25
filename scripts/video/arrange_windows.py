"""Wait for the MuJoCo viewer and the HUD, move the HUD below the viewer and bring both to the front.

GNOME ignores a plain XRaiseWindow from another client (focus-stealing prevention), so the windows are activated
with an EWMH _NET_ACTIVE_WINDOW message marked as coming from a pager. The screen recorder only sees what is on
top, so run this before a recorded demo:  DISPLAY=:1 python3 scripts/video/arrange_windows.py [timeout_s]
"""
import ctypes, ctypes.util, re, subprocess, sys, time

def windows():
    out = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True).stdout
    found = {}
    for name, key in (("MuJoCo : base", "viewer"), ("rtloop HUD", "hud")):
        m = re.search(r"(0x[0-9a-f]+) \"" + re.escape(name) + r"\"", out)
        if m:
            found[key] = int(m.group(1), 16)
    return found

t0 = time.time()
w = {}
while time.time() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 300) and len(w) < 2:
    w = windows(); time.sleep(1)
if len(w) < 2:
    sys.exit(f"windows not found: {w}")
x = ctypes.cdll.LoadLibrary(ctypes.util.find_library("X11") or "libX11.so.6")
x.XOpenDisplay.restype = ctypes.c_void_p
d = x.XOpenDisplay(None)
x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]; x.XDefaultRootWindow.restype = ctypes.c_ulong
x.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]; x.XInternAtom.restype = ctypes.c_ulong
x.XMoveWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
class CM(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int), ("display", ctypes.c_void_p),
                ("window", ctypes.c_ulong), ("message_type", ctypes.c_ulong), ("format", ctypes.c_int), ("l", ctypes.c_long * 5)]
class EV(ctypes.Union):
    _fields_ = [("xclient", CM), ("pad", ctypes.c_long * 24)]
x.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long, ctypes.POINTER(EV)]
x.XFlush.argtypes = [ctypes.c_void_p]
time.sleep(2)
x.XMoveWindow(d, w["hud"], 120, 2100)
root, act = x.XDefaultRootWindow(d), x.XInternAtom(d, b"_NET_ACTIVE_WINDOW", 0)
for win in (w["hud"], w["viewer"]):
    ev = EV(); ev.xclient.type = 33; ev.xclient.send_event = 1; ev.xclient.display = d; ev.xclient.window = win
    ev.xclient.message_type = act; ev.xclient.format = 32; ev.xclient.l[0] = 2
    x.XSendEvent(d, root, 0, (1 << 20) | (1 << 19), ctypes.byref(ev))
x.XFlush(d)
print(f"arranged at {time.time():.0f}: {w}", flush=True)
