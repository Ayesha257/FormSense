"""Simulate how the form is captured: flatbed scan or phone photo. Returns image + homography (template -> image)."""
from __future__ import annotations
import cv2, numpy as np
from PIL import Image

# rot: degrees, persp: corner jitter as share of width, pad: border around the page, blur: gaussian sigma, noise: sigma
PROFILES = {
    "scan_clean": dict(w=.30, rot=.4, persp=0.0, pad=0.0, blur=0.0, noise=2, shadow=0.0, bright=(.98, 1.02), jpeg=95, bg=(255, 255, 255), motion=0),
    "scan_noisy": dict(w=.20, rot=1.8, persp=0.004, pad=0.0, blur=0.8, noise=7, shadow=0.10, bright=(.85, 1.05), jpeg=60, bg=(235, 235, 235), motion=0),
    "phone_good": dict(w=.30, rot=3.0, persp=0.03, pad=0.08, blur=0.6, noise=4, shadow=0.20, bright=(.9, 1.05), jpeg=85, bg=(120, 95, 70), motion=0),
    "phone_bad":  dict(w=.20, rot=7.0, persp=0.07, pad=0.12, blur=1.8, noise=9, shadow=0.45, bright=(.55, .9), jpeg=45, bg=(70, 70, 75), motion=9),
}


def pick_profile(rng):
    names = list(PROFILES); return rng.choices(names, [PROFILES[n]["w"] for n in names])[0]


def capture(pil, profile, rng):
    p = PROFILES[profile]; img = np.asarray(pil.convert("RGB"))[:, :, ::-1].copy(); h, w = img.shape[:2]
    pad = int(p["pad"] * w); cw, ch = w + 2 * pad, h + 2 * pad
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    ang = np.deg2rad(rng.uniform(-p["rot"], p["rot"])); R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    c = np.array([w / 2, h / 2]); dst = ((src - c) @ R.T + c + pad).astype(np.float32)
    dst += np.float32([[rng.uniform(-p["persp"], p["persp"]) * w, rng.uniform(-p["persp"], p["persp"]) * w] for _ in range(4)])
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, H, (cw, ch), borderMode=cv2.BORDER_CONSTANT, borderValue=p["bg"][::-1])
    if p["shadow"]:  # uneven lighting: linear gradient in a random direction
        th = rng.uniform(0, 2 * np.pi); yy, xx = np.mgrid[0:ch, 0:cw]; g = (xx * np.cos(th) + yy * np.sin(th)); g = (g - g.min()) / (np.ptp(g) + 1e-6)
        out = (out * (1 - p["shadow"] * g[..., None])).astype(np.float32)
    out = out.astype(np.float32) * rng.uniform(*p["bright"])
    if p["motion"]:
        k = np.zeros((p["motion"], p["motion"]), np.float32); k[p["motion"] // 2, :] = 1; k /= k.sum()
        M = cv2.getRotationMatrix2D((p["motion"] / 2, p["motion"] / 2), rng.uniform(0, 180), 1); k = cv2.warpAffine(k, M, k.shape[::-1]); out = cv2.filter2D(out, -1, k / max(k.sum(), 1e-6))
    if p["blur"]: out = cv2.GaussianBlur(out, (0, 0), p["blur"])
    out += np.random.default_rng(rng.randint(0, 2 ** 31)).normal(0, p["noise"], out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]]); out = cv2.imdecode(buf, 1)
    return Image.fromarray(out[:, :, ::-1]), H.tolist()
