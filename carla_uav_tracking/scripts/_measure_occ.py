import glob, sys, h5py, numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else "/data/occ_val"
print("ep      frames  pitch  occ_biased  occ_struct>0.3  off_screen%  meanOcc")
frac = []
for fp in sorted(glob.glob(root + "/*.h5")):
    with h5py.File(fp, "r") as f:
        n = f["rgb"].shape[0]
        a = f.attrs
        os_ = f["annotation/occ_structural"][:] if "annotation/occ_structural" in f else np.zeros(n)
        off = f["annotation/off_screen"][:] if "annotation/off_screen" in f else np.zeros(n)
        occ = f["annotation/occlusion"][:] if "annotation/occlusion" in f else np.zeros(n)
        fr = float((os_ > 0.3).mean())
        pitch = float(a.get("camera_pitch", 0))
        biased = bool(a.get("occlusion_biased", 0))
        ep = fp.split("_")[-1][:6]
        if n > 200:
            frac.append(fr)
        print("%s  %6d  %5.0f  %9s  %12.1f%%  %9.1f%%  %.3f"
              % (ep, n, pitch, biased, 100 * fr, 100 * off.mean(), occ.mean()))
if frac:
    print("\nMEAN occ_structural>0.3 (eps that ran >200 frames) = %.1f%%   vs MVP baseline 4.3%%"
          % (100 * np.mean(frac)))
