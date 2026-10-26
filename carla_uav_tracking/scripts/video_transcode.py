"""Host-side: <seed>.raw.mp4 (mp4v, not browser-playable) -> <seed>.mp4 (vp09). Loops until --once or Ctrl-C."""
import glob, os, sys, time
import cv2
d = sys.argv[1]
while True:
    for raw in sorted(glob.glob(os.path.join(d, "*.raw.mp4"))):
        out, part = raw.replace(".raw.mp4", ".mp4"), raw.replace(".raw.mp4", ".part.vp9.mp4")
        work = raw + ".work"
        try:
            os.rename(raw, work)            # atomic claim: several workers may run in parallel
        except OSError:
            continue
        cap = cv2.VideoCapture(work); w = None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if w is None:
                w = cv2.VideoWriter(part, cv2.VideoWriter_fourcc(*"vp09"), 10, (f.shape[1], f.shape[0]))
            w.write(f)
        if w is not None:
            w.release(); os.replace(part, out); os.chmod(out, 0o644); os.remove(work); print("done", out, flush=True)
    time.sleep(10)
