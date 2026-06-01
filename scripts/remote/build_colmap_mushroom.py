"""Build a COLMAP sparse/0 model for PGSR from a MuSHRoom iPhone long_capture.

Input: transformations_colmap.json (nerfstudio-style transforms, OpenGL camera
convention, c2w, already in a COLMAP-consistent metric world frame -- it carries
`orientation_model`, NOT `applied_transform`, so NO axis swap is applied; only the
OpenGL->OpenCV camera-axis flip c2w[:3,1:3] *= -1).

Writes cameras.bin / images.bin / points3D.bin (PINHOLE) directly.
points3D are sensor-depth back-projections (uint16 mm) so PGSR has a real seed.

Usage:
  python build_colmap_mushroom.py <long_capture_dir> <out_dir>
"""
import os, sys, json, struct, glob
from pathlib import Path
import numpy as np
from PIL import Image

LC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
DEPTH_STEP = 3          # back-project every Nth frame for the seed cloud
PIX_PER_FRAME = 1200
DEPTH_SCALE = 1000.0    # uint16 millimetres -> metres
MAX_POINTS = 150000

meta = json.load(open(LC / "transformations_colmap.json"))
W = int(meta["w"]); H = int(meta["h"])
FX = float(meta["fl_x"]); FY = float(meta["fl_y"])
CX = float(meta["cx"]); CY = float(meta["cy"])
frames = meta["frames"]
# stable order by filename
frames = sorted(frames, key=lambda f: f["file_path"])
print(f"intrinsics WxH={W}x{H} fx={FX:.2f} fy={FY:.2f} cx={CX:.2f} cy={CY:.2f}")
print(f"frames in transforms: {len(frames)}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "images").mkdir(exist_ok=True)
SP = OUT / "sparse" / "0"
SP.mkdir(parents=True, exist_ok=True)


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    w, v = np.linalg.eigh(K)
    q = v[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0:
        q *= -1
    return q  # w,x,y,z


def c2w_opencv(frame):
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    # OpenGL (nerfstudio) camera -> OpenCV camera: flip Y,Z axes
    c2w[:3, 1:3] *= -1
    return c2w


# ---- symlink images + collect poses ----
img_records = []  # (image_id, qvec, tvec, camera_id, name)
poses_c2w = []
for k, frame in enumerate(frames, start=1):
    name = Path(frame["file_path"]).name
    src = (LC / frame["file_path"]).resolve()
    dst = OUT / "images" / name
    if not dst.exists():
        try:
            os.symlink(str(src), dst)
        except FileExistsError:
            pass
    c2w = c2w_opencv(frame)
    poses_c2w.append(c2w)
    w2c = np.linalg.inv(c2w)
    q = rotmat2qvec(w2c[:3, :3])
    t = w2c[:3, 3]
    img_records.append((k, q, t, 1, name))

# ---- sparse points from sensor depth ----
ii, jj = np.meshgrid(np.arange(W), np.arange(H))
ii = ii.ravel(); jj = jj.ravel()
all_pts, all_rgb = [], []
rng = np.random.default_rng(0)
for idx in range(0, len(frames), DEPTH_STEP):
    frame = frames[idx]
    dpath = LC / frame["depth_file_path"]
    ipath = LC / frame["file_path"]
    if not dpath.exists():
        continue
    d = np.asarray(Image.open(dpath)).astype(np.float32) / DEPTH_SCALE
    if d.shape != (H, W):
        d = np.asarray(Image.open(dpath).resize((W, H), Image.NEAREST)).astype(np.float32) / DEPTH_SCALE
    rgb = np.asarray(Image.open(ipath).convert("RGB").resize((W, H))).astype(np.float32) / 255.0
    dflat = d.ravel()
    sel = np.nonzero((dflat > 0.1) & (dflat < 8.0))[0]
    if len(sel) == 0:
        continue
    take = rng.choice(sel, size=min(PIX_PER_FRAME, len(sel)), replace=False)
    u = ii[take].astype(np.float32); v = jj[take].astype(np.float32)
    z = dflat[take]
    x = (u - CX) / FX * z
    y = (v - CY) / FY * z
    cam = np.stack([x, y, z, np.ones_like(z)], 1)
    world = (poses_c2w[idx] @ cam.T).T[:, :3]
    all_pts.append(world)
    all_rgb.append(rgb.reshape(-1, 3)[take])

pts = np.concatenate(all_pts, 0)
cols = (np.concatenate(all_rgb, 0) * 255).astype(np.uint8)
if len(pts) > MAX_POINTS:
    keep = rng.choice(len(pts), MAX_POINTS, replace=False)
    pts, cols = pts[keep], cols[keep]
print("sparse points", pts.shape, "bbox", np.round(pts.max(0) - pts.min(0), 2))


def w_cameras(path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 1))      # camera_id
        f.write(struct.pack("<i", 1))      # model_id 1 = PINHOLE
        f.write(struct.pack("<Q", W))
        f.write(struct.pack("<Q", H))
        for p in [FX, FY, CX, CY]:
            f.write(struct.pack("<d", p))


def w_images(path, recs):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(recs)))
        for (iid, q, t, cid, name) in recs:
            f.write(struct.pack("<I", iid))
            f.write(struct.pack("<dddd", q[0], q[1], q[2], q[3]))
            f.write(struct.pack("<ddd", t[0], t[1], t[2]))
            f.write(struct.pack("<I", cid))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))


def w_points3D(path, pts, cols, n_images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for n in range(len(pts)):
            f.write(struct.pack("<Q", n + 1))
            f.write(struct.pack("<ddd", *pts[n].tolist()))
            f.write(struct.pack("<BBB", int(cols[n][0]), int(cols[n][1]), int(cols[n][2])))
            f.write(struct.pack("<d", 0.5))
            f.write(struct.pack("<Q", 3))
            for tk in range(3):
                img_id = (n + tk) % n_images + 1
                f.write(struct.pack("<ii", img_id, 0))


w_cameras(SP / "cameras.bin")
w_images(SP / "images.bin", img_records)
w_points3D(SP / "points3D.bin", pts, cols, len(img_records))
print("WROTE", sorted(os.listdir(SP)))
print("images:", len(img_records), "points:", len(pts))
print("BUILD_COLMAP_DONE")
