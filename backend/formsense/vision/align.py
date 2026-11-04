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
                      min_inliers: int = 25) -> AlignResult:
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

    warped = cv2.warpPerspective(page, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return AlignResult(warped, True, len(good), inliers, H)
