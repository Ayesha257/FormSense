"""Align a filled-form photo/scan onto the blank template.

ORB keypoints on both images -> match -> RANSAC homography -> warp the page
into the template's coordinate frame, so every field bbox from the template
JSON lands on the right spot of the filled form.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_SIDE = 1600  # images are matched at this size for speed, then warped at full size


@dataclass
class AlignResult:
    image: np.ndarray        # page warped to template size (grayscale)
    ok: bool                 # False = alignment failed, fell back to a plain resize
    matches: int             # good keypoint matches
    inliers: int             # matches consistent with the homography
    homography: np.ndarray | None


def _scaled(gray: np.ndarray) -> tuple[np.ndarray, float]:
    s = min(1.0, MAX_SIDE / max(gray.shape))
    if s < 1.0:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return gray, s


def align_to_template(page: np.ndarray, blank: np.ndarray,
                      n_features: int = 5000, ratio: float = 0.75,
                      min_inliers: int = 25, refine: bool = True) -> AlignResult:
    """Warp `page` (gray or BGR) onto `blank` (gray)."""
    if page.ndim == 3:
        page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    h, w = blank.shape[:2]

    p_small, sp = _scaled(page)
    b_small, sb = _scaled(blank)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp_p, des_p = orb.detectAndCompute(p_small, None)
    kp_b, des_b = orb.detectAndCompute(b_small, None)

    fallback = AlignResult(cv2.resize(page, (w, h)), False, 0, 0, None)
    if des_p is None or des_b is None or len(kp_p) < 10 or len(kp_b) < 10:
        return fallback

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(des_p, des_b, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    if len(good) < min_inliers:
        fallback.matches = len(good)
        return fallback

    src = np.float32([kp_p[m.queryIdx].pt for m in good]) / sp   # full-res page coords
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]) / sb   # full-res template coords
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < min_inliers:
        fallback.matches, fallback.inliers = len(good), inliers
        return fallback

    if refine:
        H = refine_ecc(page, blank, H)
    warped = cv2.warpPerspective(page, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return AlignResult(warped, True, len(good), inliers, H)


def _ecc_image(gray: np.ndarray, scale: float) -> np.ndarray:
    """Lighting-independent, slightly blurred 'darkness' image for ECC."""
    from formsense.vision.fields import darkness
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(darkness(small), (0, 0), 1.5).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def refine_ecc(page: np.ndarray, blank: np.ndarray, H: np.ndarray, scale: float = 0.5,
               iterations: int = 100) -> np.ndarray:
    """Second alignment stage: ECC (enhanced correlation coefficient) refinement.

    ORB keypoints get the page roughly right; on blurry or bent photos the fit can
    still be 10-20 px off in places. ECC fine-tunes the homography by maximising
    the pixel-level correlation with the blank template. The result is kept only
    if it really matches the template better."""
    h, w = blank.shape[:2]
    S = np.diag([scale, scale, 1.0])
    rough = cv2.warpPerspective(page, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    tmpl, moving = _ecc_image(blank, scale), _ecc_image(rough, scale)
    before = _corr(tmpl, moving)
    W = np.eye(3, dtype=np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-5)
        _, W = cv2.findTransformECC(tmpl, moving, W, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    except cv2.error:
        return H
    # W maps template pixels to rough-aligned pixels (at `scale`); fold it into H
    W_full = np.linalg.inv(S) @ W.astype(np.float64) @ S
    H_new = np.linalg.inv(W_full) @ H
    refined = cv2.warpPerspective(moving, W, (moving.shape[1], moving.shape[0]),
                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return H_new if _corr(tmpl, refined) > before + 1e-3 else H
