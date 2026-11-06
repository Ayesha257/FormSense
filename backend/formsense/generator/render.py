"""Draw the blank form from the template JSON, then fill it with handwriting-like content."""
from __future__ import annotations
import pathlib, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = pathlib.Path(__file__).resolve().parents[2]
FONTS = ROOT / "data" / "fonts"
HAND_SCALE = {"hand_caveat": 1.15, "hand_indie": 0.95, "hand_reenie": 1.35, "hand_shadows": 1.05, "hand_patrick": 1.0, "hand_gochi": 0.95}
INK = [(18, 28, 110), (10, 10, 10), (25, 45, 150), (60, 60, 70)]   # blue pen, black pen, dark blue, pencil
INK_THRESHOLD = 200   # gray level below this counts as ink


def hand_fonts():
    return [n for n in HAND_SCALE if (FONTS / f"{n}.ttf").exists()] or ["__default__"]


def pfont(size):
    p = FONTS / "print_main.ttf"
    try:
        return ImageFont.truetype(str(p), size)
    except Exception:
        return ImageFont.load_default(size)


def hfont(name, size):
    try:
        return ImageFont.truetype(str(FONTS / f"{name}.ttf"), size)
    except Exception:
        return ImageFont.load_default(size)


# ------------------------------------------------------------------ blank form
def blank_form(tpl, rng=None, variants=False):
    """Return (PIL image, {field_id: label printed}). With variants=True, labels are randomly swapped for synonyms
    (e.g. 'D.O.B' instead of 'Date of Birth') to test label to field matching."""
    img = Image.new("RGB", (tpl["width"], tpl["height"]), "white"); d = ImageDraw.Draw(img)
    g = lambda v: (v, v, v)
    for e in tpl["static"]:
        k = e["k"]
        if k == "border": x, y, w, h = e["box"]; d.rectangle([x, y, x + w, y + h], outline=g(60), width=3)
        elif k == "mark":
            cx, cy = e["c"]; d.line([cx - 20, cy, cx + 20, cy], fill=g(30), width=3); d.line([cx, cy - 20, cx, cy + 20], fill=g(30), width=3)
        elif k == "logo":
            cx, cy = e["c"]; r = e["r"]
            for rr in (r, r - 10, r - 22): d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=g(50), width=3)
            d.polygon([(cx, cy - 28), (cx + 26, cy + 20), (cx - 26, cy + 20)], fill=g(70))
            d.text((cx, cy + 4), "NIC", font=pfont(20), fill="white", anchor="mm")
        elif k == "photo":
            x, y, w, h = e["box"]; d.rectangle([x, y, x + w, y + h], outline=g(90), width=2)
            d.text((x + w / 2, y + h / 2), "Passport size\nphotograph", font=pfont(16), fill=g(110), anchor="mm", align="center")
        elif k == "text":
            d.text((e["x"], e["y"]), e["t"], font=pfont(e["size"]), fill=g(e["color"]), stroke_width=1 if e["bold"] else 0, stroke_fill=g(e["color"]))
        elif k == "bar":
            d.rectangle([80, e["y"], 1160, e["y"] + 38], fill=g(222)); d.text((92, e["y"] + 6), e["t"], font=pfont(22), fill=g(30), stroke_width=1, stroke_fill=g(30))
        elif k == "cell":
            x, y, w, h = e["box"]; d.rectangle([x, y, x + w, y + h], fill=g(238), outline=g(110), width=2); d.text((x + w / 2, y + h / 2), e["t"], font=pfont(18), fill=g(30), anchor="mm")
    used = {}
    for f in tpl["fields"]:
        x, y, w, h = f["bbox"]
        if f["type"] == "checkbox":
            d.rectangle([x, y, x + w, y + h], outline=g(50), width=3)
            d.text((x + w + 14, y + h / 2), f["label"], font=pfont(24), fill=g(30), anchor="lm"); used[f["name"]] = f["label"]; continue
        d.rectangle([x, y, x + w, y + h], outline=g(70), width=2)
        if f["type"] == "signature": continue
        if f["draw_label"]:
            lab = f["label"]
            if variants and rng and f["variants"]:
                lab = rng.choice([lab] + f["variants"])
            lab_txt = lab + (" *" if f["required"] else "")
            if d.textlength(lab_txt, font=pfont(22)) > 285: lab_txt, lab = f["label"] + " *", f["label"]
            d.text((80, y + h / 2 if f["kind"] != "multiline" else y + 28), lab_txt, font=pfont(22), fill=g(30), anchor="lm"); used[f["name"]] = lab
        if f.get("hint"):
            d.text((x + w + 14, y + h / 2), f["hint"], font=pfont(16), fill=g(130), anchor="lm")
    return img, used


# ------------------------------------------------------------------ handwriting
def _wrap(d, text, font, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]


def _put(d, x, base, text, font, ink, rng):
    """draw char by char with small baseline jitter -> less 'typed' look"""
    for i, ch in enumerate(text):
        cx = x + d.textlength(text[:i], font=font)
        d.text((cx + rng.uniform(-.6, .6), base + rng.uniform(-2, 2)), ch, font=font, fill=ink + (rng.randint(215, 255),), anchor="ls")


def text_layer(w, h, text, fname, ink, rng, multiline=False, struck=None, illegible=False, size_mul=1.0):
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    sc = HAND_SCALE.get(fname, 1.0) * size_mul
    size = int((h * (0.36 if multiline else 0.6)) * sc)
    maxw = w - 26
    if multiline:
        while True:
            f = hfont(fname, size); lines = _wrap(d, text, f, maxw)
            if len(lines) <= 2 or size < 16: break
            size -= 3
        for i, ln in enumerate(lines[:3]): _put(d, 12, h * (0.40 + 0.45 * i), ln, f, ink, rng)
    else:
        x0 = 12
        if struck:
            fs = hfont(fname, int(size * .6)); _put(d, x0, h * .62, struck, fs, ink, rng)
            sw = d.textlength(struck, font=fs); y = h * .62 - fs.size * .3
            d.line([x0 - 4, y + rng.uniform(-2, 2), x0 + sw + 4, y + rng.uniform(-2, 2)], fill=ink + (255,), width=3)
            x0 += sw + 22; maxw -= sw + 22
        f = hfont(fname, size)
        while d.textlength(text, font=f) > maxw and size > 14: size -= 2; f = hfont(fname, size)
        _put(d, x0, h * .72, text, f, ink, rng)
    if illegible:  # smear, scribble, blur: ink is there, but nobody can read it
        a = layer.copy()
        for dx, dy in [(5, 2), (-4, 3), (9, -3)]: layer.paste(a, (dx, dy), a)
        d = ImageDraw.Draw(layer)
        for _ in range(rng.randint(3, 5)):
            pts = [(rng.uniform(0, w), rng.uniform(h * .2, h * .9)) for _ in range(rng.randint(4, 7))]
            d.line(pts, fill=ink + (230,), width=rng.randint(6, 10), joint="curve")
        layer = layer.filter(ImageFilter.GaussianBlur(rng.uniform(2.2, 3.4)))
    return layer.rotate(rng.uniform(-1.4, 1.4), resample=Image.BICUBIC, center=(w // 2, h // 2))


def _catmull(pts, n=10):
    P = np.array([pts[0]] + list(pts) + [pts[-1]], float); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return [tuple(p) for p in out]


def signature_layer(w, h, ink, rng, mode):
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    if mode == "speck":
        for _ in range(rng.randint(1, 3)):
            x, y = rng.uniform(20, w - 20), rng.uniform(20, h - 20); r = rng.uniform(1.5, 3); d.ellipse([x - r, y - r, x + r, y + r], fill=ink + (255,))
        return layer
    col = (140, 150, 175, 255) if mode == "faint" else ink + (255,)
    n = rng.randint(7, 11); xs = np.linspace(w * .08, w * .88, n)
    pts = [(x + rng.uniform(-10, 10), h * .5 + rng.uniform(-h * .32, h * .32)) for x in xs]
    d.line(_catmull(pts), fill=col, width=2 if mode == "faint" else rng.randint(3, 5), joint="curve")
    y = h * rng.uniform(.65, .85); d.line(_catmull([(w * .1, y), (w * .45, y + rng.uniform(-8, 8)), (w * .9, y - rng.uniform(0, 12))]), fill=col, width=3 if mode != "faint" else 2)
    return layer


def draw_checkbox(img, box, style, ink, rng):
    x, y, w, h = box; d = ImageDraw.Draw(img); c = ink; j = lambda: rng.uniform(-2, 2)
    if style == "tick":
        d.line([(x + 8 + j(), y + h * .5 + j()), (x + w * .4 + j(), y + h * .85 + j()), (x + w * 1.02 + j(), y - 6 + j())], fill=c, width=4, joint="curve")
    elif style == "cross":
        d.line([(x + 6 + j(), y + 6 + j()), (x + w - 6 + j(), y + h - 6 + j())], fill=c, width=4); d.line([(x + w - 6 + j(), y + 6 + j()), (x + 6 + j(), y + h - 6 + j())], fill=c, width=4)
    else:  # scribbled fill
        for i in range(6, h - 4, 7): d.line([(x + 5 + j(), y + i), (x + w - 5 + j(), y + i + j())], fill=c, width=4)


def fill_form(img, tpl, state, rng):
    """Draw handwriting into the image in place. state: generator.errors.FormState (+ .style with font, ink, tick)."""
    st = state.style; fname, ink = st["font"], tuple(st["ink"])
    sig_font_ink = tuple(st["sig_ink"])
    for f in tpl["fields"]:
        x, y, w, h = f["bbox"]; fid = f["name"]
        if f["type"] == "checkbox":
            if state.ticks.get(fid): draw_checkbox(img, f["bbox"], st["tick"], ink, rng)
        elif f["type"] == "signature":
            if state.signature != "none":
                l = signature_layer(w, h, sig_font_ink, rng, state.signature); img.paste(l, (x, y), l)
        else:
            txt = state.text.get(fid, ""); df = state.defects.get(fid)
            if not txt and df != "illegible": continue
            struck = df[1] if isinstance(df, tuple) else None
            l = text_layer(w, h, txt, fname, ink, rng, f["kind"] == "multiline", struck, df == "illegible", st["size_mul"])
            img.paste(l, (x, y), l)


def ink_ratio(img, bbox, shrink=5):
    x, y, w, h = bbox; a = np.asarray(img.convert("L"))[y + shrink:y + h - shrink, x + shrink:x + w - shrink]
    return float((a < INK_THRESHOLD).mean())
