#!/usr/bin/env python3
"""Draw every image on github.com/shvuuuu, once for each GitHub theme.

    pip install -r scripts/requirements.txt
    python3 scripts/build.py              # writes assets/dark/*.svg and assets/light/*.svg
    python3 scripts/build.py hero rules   # only some of them

Everything is drawn in dots. The type is Geist and Geist Mono (SIL OFL 1.1, see
scripts/fonts/OFL.txt), subset and embedded in each SVG so it renders the same on
every machine. The hero is redrawn from "Golden Fog, San Francisco" by runner310,
CC BY 2.0; that credit travels inside hero.svg and in scripts/source/CREDITS.md.
"""
from __future__ import annotations

import base64
import io
import math
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from PIL import Image, ImageFilter
import logging
logging.getLogger("fontTools").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "scripts" / "fonts"
SRC = ROOT / "scripts" / "source"
OUT = ROOT / "assets"

# --------------------------------------------------------------------------- theme

THEMES = {
    "dark": dict(
        bg="#0d1117", surface="#151b23", surface2="#1c232d", line="#262d36",
        ink="#f0f6fc", muted="#9198a1", faint="#5c6570",
        orange="#ff5a1f", orange_text="#ff7b47", lamp="#ffc56b", air="#ff3b30",
        fog="#a7b6c8", fog_op=0.58, wisp_op=0.42,
        hero_fog="#a7b6c8", hero_fog_op=0.58, hero_wisp_op=0.42,
        ok="#3fb950", bad="#f85149",
        chip="#21262d", chip_line="#3d444d", chip_shadow=("#010409", 0.55),
    ),
    "light": dict(
        bg="#ffffff", surface="#f6f8fa", surface2="#eef1f4", line="#e3e7ec",
        ink="#1f2328", muted="#59636e", faint="#8c959f",
        orange="#e5470b", orange_text="#c2410c", lamp="#d0830b", air="#e5383b",
        fog="#8d9aab", fog_op=0.50, wisp_op=0.36,
        hero_fog="#6f7f95", hero_fog_op=0.74, hero_wisp_op=0.58,
        ok="#1a7f37", bad="#cf222e",
        chip="#ffffff", chip_line="#d1d9e0", chip_shadow=("#1f2328", 0.10),
    ),
}

# family, weight, size, tracking (em)
ROLES = {
    "display": ("Geist", 600, 76, -0.035),
    "h2": ("Geist", 600, 54, -0.032),
    "title": ("Geist", 600, 38, -0.026),
    "name": ("Geist", 600, 27, -0.012),
    "lede": ("Geist", 400, 25, -0.005),
    "body": ("Geist", 400, 22, -0.002),
    "small": ("Geist", 400, 18, 0.0),
    "chip": ("Geist", 500, 17, -0.006),
    "mono": ("GeistMono", 400, 17, 0.0),
    "monob": ("GeistMono", 500, 17, 0.0),
}
FALLBACK = {
    "Geist": "Geist,-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif",
    "GeistMono": "'Geist Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
}

# --------------------------------------------------------------------------- fonts


@lru_cache(maxsize=None)
def static_font_bytes(family: str, weight: int) -> bytes:
    vf = TTFont(FONTS / f"{family}-VF.ttf")
    font = instancer.instantiateVariableFont(vf, {"wght": weight})
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


@lru_cache(maxsize=None)
def metrics(family: str, weight: int):
    font = TTFont(io.BytesIO(static_font_bytes(family, weight)))
    return font.getBestCmap(), font["hmtx"].metrics, font["head"].unitsPerEm


def measure(text: str, role: str, size: float | None = None) -> float:
    family, weight, rsize, tracking = ROLES[role]
    size = size or rsize
    cmap, hmtx, upm = metrics(family, weight)
    adv = sum(hmtx[cmap.get(ord(c), ".notdef")][0] for c in text)
    return adv * size / upm + tracking * size * len(text)


def woff2_b64(family: str, weight: int, chars: str) -> str:
    font = TTFont(io.BytesIO(static_font_bytes(family, weight)))
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["kern", "liga", "calt", "ccmp", "locl", "mark", "mkmk", "tnum"]
    opts.name_IDs = []
    opts.hinting = False
    opts.notdef_outline = True
    sub = subset.Subsetter(opts)
    sub.populate(text=chars + " ")
    sub.subset(font)
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    return base64.b64encode(buf.getvalue()).decode()


def wrap_balanced(text: str, role: str, width: float) -> list[str]:
    """Two even lines when one won't fit, so no word is left alone on the second line."""
    if measure(text, role) <= width:
        return [text]
    words = text.split(" ")
    best = None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        wa, wb = measure(a, role), measure(b, role)
        if wa <= width and wb <= width and (best is None or max(wa, wb) < best[0]):
            best = (max(wa, wb), [a, b])
    return best[1] if best else wrap(text, role, width)


def wrap(text: str, role: str, width: float, size: float | None = None) -> list[str]:
    lines, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}".strip()
        if cur and measure(trial, role, size) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- svg


class Svg:
    """A small SVG writer that records which glyphs each font needs."""

    def __init__(self, w: int, h: int, theme: str, title: str):
        self.w, self.h, self.theme, self.title = w, h, theme, title
        self.desc = ""
        self.c = THEMES[theme]
        self.defs: list[str] = []
        self.css: list[str] = []
        self.body: list[str] = []
        self.glyphs: dict[tuple[str, int], set[str]] = defaultdict(set)

    def add(self, s: str):
        self.body.append(s)

    def text(self, x, y, s, role="body", fill=None, anchor="start", size=None, opacity=None, extra=""):
        family, weight, rsize, tracking = ROLES[role]
        size = size or rsize
        self.glyphs[(family, weight)].update(s)
        fam = f"'{family}{weight}',{FALLBACK[family]}"
        attrs = [
            f'x="{x:g}"', f'y="{y:g}"', f'font-family="{fam}"', f'font-size="{size:g}"',
            f'font-weight="{weight}"', f'fill="{fill or self.c["ink"]}"',
        ]
        if tracking:
            attrs.append(f'letter-spacing="{tracking * size:.2f}"')
        if anchor != "start":
            attrs.append(f'text-anchor="{anchor}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        self.add(f"<text {' '.join(attrs)} {extra}>{escape(s)}</text>")

    def runs(self, x, y, parts, role="body", size=None, anchor="start"):
        """One line made of differently coloured runs: parts = [(text, fill), ...]."""
        family, weight, rsize, tracking = ROLES[role]
        size = size or rsize
        fam = f"'{family}{weight}',{FALLBACK[family]}"
        spans = []
        for s, fill in parts:
            self.glyphs[(family, weight)].update(s)
            spans.append(f'<tspan fill="{fill}">{escape(s)}</tspan>')
        ls = f' letter-spacing="{tracking * size:.2f}"' if tracking else ""
        ta = f' text-anchor="{anchor}"' if anchor != "start" else ""
        self.add(
            f'<text x="{x:g}" y="{y:g}" font-family="{fam}" font-size="{size:g}" '
            f'font-weight="{weight}"{ls}{ta} xml:space="preserve">{"".join(spans)}</text>'
        )

    def paragraph(self, x, y, s, role, width, leading, fill=None, size=None):
        lines = wrap(s, role, width, size)
        for i, line in enumerate(lines):
            self.text(x, y + i * leading, line, role, fill=fill, size=size)
        return y + (len(lines) - 1) * leading

    def dots(self, pts, r, fill, opacity=1.0, cls=""):
        """Many dots of one radius as a single path of zero-length round-capped segments."""
        if not pts:
            return
        d = "".join(f"M{x:g} {y:g}h0" for x, y in pts)
        c = f' class="{cls}"' if cls else ""
        op = f' opacity="{opacity:g}"' if opacity < 1 else ""
        self.add(
            f'<path{c} d="{d}" fill="none" stroke="{fill}" stroke-width="{2 * r:g}" '
            f'stroke-linecap="round"{op}/>'
        )

    def render(self) -> str:
        faces = []
        for (family, weight), chars in sorted(self.glyphs.items()):
            data = woff2_b64(family, weight, "".join(sorted(chars)))
            faces.append(
                f"@font-face{{font-family:'{family}{weight}';font-weight:{weight};"
                f"src:url(data:font/woff2;base64,{data}) format('woff2')}}"
            )
        style = "".join(faces + self.css)
        motion = "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"
        desc = f"<desc>{escape(self.desc)}</desc>\n" if self.desc else ""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" role="img" '
            f'aria-label="{escape(self.title)}">\n<title>{escape(self.title)}</title>\n{desc}'
            f"<style>{style}{motion}</style>\n"
            f'<defs>{"".join(self.defs)}</defs>\n' + "\n".join(self.body) + "\n</svg>\n"
        )

    def save(self, name: str):
        path = OUT / self.theme / f"{name}.svg"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.render()
        path.write_text(data)
        print(f"  {self.theme:5s} {name:18s} {len(data) / 1024:6.1f} KB")


# --------------------------------------------------------------------------- helpers


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fbm(x, y, seed, tile_w, octaves=((3, 1.0, 1.0), (7, 2.3, 0.5), (13, 4.1, 0.26), (29, 7.7, 0.13))):
    """Wavy fog-like noise in 0..1, periodic in x with period tile_w."""
    rnd = random.Random(seed)
    v = np.zeros_like(x, dtype=float)
    total = 0.0
    for fx, fy, amp in octaves:
        p1, p2, p3 = (rnd.uniform(0, 2 * math.pi) for _ in range(3))
        kx = 2 * math.pi * fx / tile_w
        ky = 2 * math.pi * fy / 420
        v += amp * np.sin(kx * x + p1 + 1.1 * np.sin(ky * y + p2) + 0.4 * np.sin(2 * math.pi * x / tile_w + p3))
        total += amp
    return 0.5 + 0.5 * v / total


def grid(x0, y0, w, h, pitch):
    xs = x0 + (np.arange(int(w // pitch)) + 0.5) * pitch
    ys = y0 + (np.arange(int(h // pitch)) + 0.5) * pitch
    return np.meshgrid(xs, ys)


def quantize(pts_r, step=0.25):
    groups = defaultdict(list)
    for x, y, r in pts_r:
        q = round(r / step) * step
        if q >= 0.5:
            groups[q].append((round(float(x), 1), round(float(y), 1)))
    return groups


# --------------------------------------------------------------------------- hero

PITCH = 8


def bridge_cells(crop, scale, cols, rows):
    """Where the bridge is, per grid cell, read from the photo: tower, cables, deck, lamps."""
    im = Image.open(SRC / "golden-fog-san-francisco.jpg").convert("RGB").crop(crop)
    a = np.asarray(im).astype(np.float32) / 255
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    v, s = mx, (mx - mn) / (mx + 1e-6)
    d = mx - mn + 1e-6
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    bg = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9))) / 255
    steel = ((hue >= 255) | (hue <= 42)) & (s >= 0.17) & (v >= 0.09) & ~((v > 0.86) & (g > 0.5))
    lines = ((bg - lum) > 0.06) & (bg > 0.42)
    lamps = (v > 0.86) & (r > 0.82) & (g > 0.5) & ((hue <= 60) | (hue >= 345)) & (s > 0.22)

    def frac(mask):
        img = Image.fromarray((mask * 255).astype(np.uint8)).resize((cols, rows), Image.BOX)
        return np.asarray(img).astype(np.float32) / 255

    return frac(steel | lines), frac(lamps)


def hero(theme: str):
    c = THEMES[theme]
    W, Y0, IH = 1280, 288, 568
    H = Y0 + IH
    crop_l, crop_t, crop_r = 0, 60, 1728
    scale = W / (crop_r - crop_l)
    crop = (crop_l, crop_t, crop_r, crop_t + round(IH / scale))
    cols, rows = W // PITCH, IH // PITCH

    svg = Svg(W, H, theme, "Vibes don’t ship. Evidence does. AI agents built for production, not demos. "
              "The Golden Gate Bridge drawn in dots, its far tower disappearing into the fog.")
    svg.desc = ("Bridge drawn in dots after “Golden Fog, San Francisco” by runner310, CC BY 2.0, "
                "https://commons.wikimedia.org/wiki/File:Golden_Fog,_San_Francisco.jpg "
                "(licence: https://creativecommons.org/licenses/by/2.0/). Fog redrawn.")
    svg.css.append(
        "@keyframes drift{to{transform:translateX(-1280px)}}"
        ".drift{animation:drift 104s steps(160) infinite}"
        "@keyframes blink{0%,46%{opacity:1}50%,96%{opacity:.12}100%{opacity:1}}"
        ".air{animation:blink 2.6s infinite}.air2{animation-delay:1.3s}"
        "@keyframes tw{0%,100%{opacity:1}50%{opacity:.35}}"
        ".tw{animation:tw 3.7s ease-in-out infinite}.tw2{animation-delay:-1.4s}.tw3{animation-delay:-2.6s}"
    )

    # -- headline and the one sentence under it
    x = 0
    svg.text(x, 118, "Vibes don’t ship.", "display")
    svg.text(x, 200, "Evidence does.", "display", fill=c["muted"])
    svg.text(x, 254, "AI agents built for production, not demos.", "lede")

    # -- the bridge, read from the photograph
    steel, lamps = bridge_cells(crop, scale, cols, rows)
    X, Y = grid(0, Y0, W, rows * PITCH, PITCH)
    rmax = PITCH * 0.46
    edge = smooth(0, 64, X) * smooth(0, 64, W - X) * smooth(0, 130, H - Y)
    fog_top = 594 + 7 * np.sin(X / 71) + 5 * np.sin(X / 23 + 1.3)

    lamp_pts, steel_pts, fog_pts = [], [], []
    is_lamp = lamps > 0.03
    is_steel = (steel > 0.12) & ~is_lamp
    r_steel = rmax * np.clip(0.5 + 0.75 * steel, 0, 1) * edge
    band = smooth(fog_top - 8, fog_top + 46, Y) * (1 - 0.38 * smooth(fog_top + 70, fog_top + 270, Y))
    n = fbm(X, Y, 7, W)
    fog = band * (0.28 + 0.72 * n) * edge
    r_fog = rmax * np.clip(fog, 0, 1) ** 1.05
    for j in range(rows):
        for i in range(cols):
            xx, yy = X[j, i], Y[j, i]
            if is_lamp[j, i]:
                lamp_pts.append((xx, yy, rmax * 0.95 * edge[j, i]))
            elif is_steel[j, i]:
                steel_pts.append((xx, yy, r_steel[j, i]))
            elif r_fog[j, i] > 0.5:
                fog_pts.append((xx, yy, r_fog[j, i]))

    for r, pts in sorted(quantize(fog_pts).items()):
        svg.dots(pts, r, c["hero_fog"], c["hero_fog_op"])
    for r, pts in sorted(quantize(steel_pts).items()):
        svg.dots(pts, r, c["orange"])
    lamps_q = [(x_, y_, r_) for x_, y_, r_ in lamp_pts]
    rnd = random.Random(3)
    buckets = {"tw": [], "tw tw2": [], "tw tw3": [], "": []}
    for x_, y_, r_ in lamps_q:
        buckets[rnd.choice(list(buckets))].append((x_, y_, r_))
    for cls, pts in buckets.items():
        for r, ps in quantize(pts).items():
            svg.dots(ps, r, c["lamp"], cls=cls)

    # -- the far tower's legs, going down into the fog
    for lx in (972, 996):
        for k, rr in enumerate((2.7, 2.3, 1.9, 1.4, 1.0, 0.7)):
            svg.add(f'<circle cx="{lx}" cy="{612 + 8 * k}" r="{rr}" fill="{c["orange"]}" opacity="{1 - 0.12 * k:.2f}"/>')

    # -- aviation lights on the tower tops
    for (px_, py_), cls in (((383, 92), "air"), ((492, 82), "air air2")):
        hx = (math.floor((px_ - crop_l) * scale / PITCH) + 0.5) * PITCH
        hy = Y0 + (math.floor((py_ - crop_t) * scale / PITCH) + 0.5) * PITCH
        svg.add(f'<circle class="{cls}" cx="{hx:g}" cy="{hy:g}" r="3.4" fill="{c["air"]}"/>')

    # -- fog that drifts in front of everything, one dot-step at a time
    wx, wy = grid(0, Y0, 2 * W, rows * PITCH, PITCH)
    # whole-number frequencies of the loop width, so the band edge matches itself at the seam
    ftop = 594 + 7 * np.sin(2 * math.pi * 18 * wx / W) + 5 * np.sin(2 * math.pi * 56 * wx / W + 1.3)
    wband = smooth(ftop - 40, ftop + 30, wy) * (1 - smooth(H - 150, H - 10, wy))
    wn = fbm(wx, wy, 21, W, octaves=((2, 0.8, 1.0), (5, 1.9, 0.55), (11, 3.3, 0.3)))
    wisp = wband * np.clip((wn - 0.52) * 2.6, 0, 1)
    wpts = [(wx[j, i], wy[j, i], rmax * 0.9 * wisp[j, i])
            for j in range(wx.shape[0]) for i in range(wx.shape[1]) if wisp[j, i] > 0.1]
    svg.add('<g mask="url(#heromask)"><g class="drift">')
    for r, pts in sorted(quantize(wpts).items()):
        svg.dots(pts, r, c["hero_fog"], c["hero_wisp_op"])
    svg.add("</g></g>")
    svg.defs.append(
        '<linearGradient id="fadex" x1="0" x2="1" y1="0" y2="0">'
        '<stop offset="0" stop-color="#000"/><stop offset=".05" stop-color="#fff"/>'
        '<stop offset=".95" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>'
        f'<mask id="heromask" maskUnits="userSpaceOnUse" x="0" y="{Y0}" width="{W}" height="{IH}">'
        f'<rect x="0" y="{Y0}" width="{W}" height="{IH}" fill="url(#fadex)"/></mask>'
    )
    svg.save("hero")


# --------------------------------------------------------------------------- shared pieces


def drift_css(name: str, width: int, seconds: float) -> str:
    steps = width // PITCH
    return (f"@keyframes {name}{{to{{transform:translateX(-{width}px)}}}}"
            f".{name}{{animation:{name} {seconds}s steps({steps}) infinite}}")


def fog_wisps(svg: Svg, x0, y0, w, h, seed, density=0.52, top_fade=60, bottom_fade=60, op=None,
              seconds=110, mask_id="wispmask"):
    """A band of fog dots that scrolls left one dot at a time, seamlessly, masked to its box."""
    c = svg.c
    cls = f"{mask_id}d"
    svg.css.append(drift_css(cls, int(w), seconds))
    gx, gy = grid(x0, y0, 2 * w, h, PITCH)
    lx = gx - x0
    n = fbm(lx, gy, seed, w, octaves=((2, 0.8, 1.0), (5, 1.9, 0.55), (11, 3.3, 0.3)))
    band = smooth(y0, y0 + top_fade, gy) * (1 - smooth(y0 + h - bottom_fade, y0 + h, gy))
    k = band * np.clip((n - density) * 2.6, 0, 1)
    pts = [(gx[j, i], gy[j, i], PITCH * 0.42 * k[j, i])
           for j in range(gx.shape[0]) for i in range(gx.shape[1]) if k[j, i] > 0.1]
    svg.defs.append(
        f'<linearGradient id="{mask_id}g" x1="0" x2="1" y1="0" y2="0">'
        '<stop offset="0" stop-color="#000"/><stop offset=".08" stop-color="#fff"/>'
        '<stop offset=".92" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>'
        f'<mask id="{mask_id}" maskUnits="userSpaceOnUse" x="{x0}" y="{y0}" width="{w}" height="{h}">'
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="url(#{mask_id}g)"/></mask>'
    )
    svg.add(f'<g mask="url(#{mask_id})"><g class="{cls}">')
    for r, ps in sorted(quantize(pts).items()):
        svg.dots(ps, r, c["fog"], op if op is not None else c["wisp_op"])
    svg.add("</g></g>")


def heading(name: str, line1: str, line2: str):
    def build(theme):
        svg = Svg(1280, 170, theme, f"{line1} {line2}")
        svg.text(0, 82, line1, "h2")
        svg.text(0, 146, line2, "h2", fill=svg.c["muted"])
        svg.save(name)
    return build


def image_b64(path: Path, width: int, crop=None, quality=84, alpha=False) -> tuple[str, int, int]:
    im = Image.open(path).convert("RGBA" if alpha else "RGB")
    if crop:
        im = im.crop(crop)
    h = round(im.height * width / im.width)
    im = im.resize((width, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=6)
    return base64.b64encode(buf.getvalue()).decode(), width, h


@lru_cache(maxsize=None)
def logo_b64(name: str, px: int) -> str:
    """A raster logo cropped to what it draws and stored as a small PNG."""
    im = Image.open(SRC / name).convert("RGBA")
    im = im.crop(im.getchannel("A").getbbox())
    im = im.resize((px, round(im.height * px / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# marks, each drawn in a box of `size` at (x, y)

def mark_hosh(svg: Svg, x, y, size):
    s = size / 100
    tile, ink = ("#FFFFFF", "#141414") if svg.theme == "dark" else ("#141414", "#FFFFFF")
    svg.add(
        f'<g transform="translate({x:g} {y:g}) scale({s:g})">'
        f'<rect width="100" height="100" rx="23" fill="{tile}"/>'
        f'<rect x="11.5" y="34.5" width="77" height="31" rx="15.5" fill="none" stroke="{ink}" stroke-width="7"/>'
        '<rect x="19" y="42" width="11" height="16" rx="5.5" fill="#F2555A"/>'
        '<rect x="70" y="42" width="11" height="16" rx="5.5" fill="#2FBF71"/>'
        f'<ellipse cx="50" cy="50" rx="10" ry="7" fill="{ink}"/></g>'
    )
    return size


def mark_exhibit(svg: Svg, x, y, size):
    s = size / 100
    tile, ink = ("#FFFFFF", "#0b0b0b") if svg.theme == "dark" else ("#0b0b0b", "#FFFFFF")
    svg.add(
        f'<g transform="translate({x:g} {y:g}) scale({s:g})">'
        f'<rect width="100" height="100" rx="22" fill="{tile}"/>'
        f'<path d="M22 22H80V36H36V64H80V78H22Z" fill="{ink}"/>'
        '<rect x="44" y="44" width="34" height="12" fill="#E8A33D"/></g>'
    )
    return size


def sphere_dots(cx, cy, radius, pitch, light=(-0.45, -0.55)):
    """A dot sphere lit from the top left, like the DataSphere and 9to5code marks."""
    pts = []
    n = int(radius // pitch) + 1
    lx, ly = light
    lz = math.sqrt(max(0.0, 1 - lx * lx - ly * ly))
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            x, y = i * pitch, j * pitch
            d = math.hypot(x, y) / radius
            if d > 1:
                continue
            nz = math.sqrt(1 - d * d)
            shade = max(0.0, (x / radius) * lx + (y / radius) * ly + nz * lz)
            r = pitch * 0.46 * (0.25 + 0.75 * shade) * (0.55 + 0.45 * nz)
            pts.append((cx + x, cy + y, r))
    return pts


def mark_datasphere(svg: Svg, x, y, size):
    blue = "#4C8DFF" if svg.theme == "dark" else "#2563EB"
    pts = sphere_dots(x + size / 2, y + size / 2, size / 2, size / 6.5)
    for r, ps in quantize(pts, 0.1).items():
        svg.dots(ps, r, blue)
    return size


# KaDeep Studio's logo is KS in white dots on a black tile, and it stays black in both themes.
# Its units are those of kadeep-mark.svg (dot pitch 100, radius 39); the letters span w x h of them.
KS = dict(w=1228.0, h=778.0)
KS_TILE = "#0A0A0A"


@lru_cache(maxsize=None)
def ks_points() -> tuple:
    import re
    src = (SRC / "kadeep-mark.svg").read_text()
    return tuple((float(a), float(b)) for a, b in re.findall(r'cx="([\d.]+)" cy="([\d.]+)"', src))


def mark_ks_tile(svg: Svg, x, y, size, fill=0.72):
    """KaDeep Studio as a small app tile, the letters enlarged to stay legible."""
    s = fill * size / KS["w"]
    ox = x + (size - KS["w"] * s) / 2 - 11 * s
    oy = y + (size - KS["h"] * s) / 2 - 11 * s
    svg.add(f'<rect x="{x:g}" y="{y:g}" width="{size:g}" height="{size:g}" rx="{0.224 * size:.2f}" fill="{KS_TILE}"/>')
    svg.dots([(ox + u * s, oy + v * s) for u, v in ks_points()], 43 * s, "#FFFFFF")
    return size


def mark_kadeep_ai(svg: Svg, x, y, size):
    """KaDeep AI's folded blue K, from kadeep.ai."""
    svg.add(f'<image x="{x:g}" y="{y:g}" width="{size:g}" height="{size:g}" preserveAspectRatio="xMidYMid meet" '
            f'href="data:image/png;base64,{logo_b64("kadeep-ai-mark.png", round(size * 3))}"/>')
    return size


def mark_glance(svg: Svg, x, y, size):
    """Glance's Frost icon, the arctic fox, as on glance.9to5code.com."""
    svg.add(f'<image x="{x:g}" y="{y:g}" width="{size:g}" height="{size:g}" '
            f'href="data:image/png;base64,{logo_b64("glance-icon-frost.png", round(size * 3))}"/>')
    return size


def tm_text(svg: Svg, x, y, head, tail, role, size=None, fill=None) -> float:
    """`head`, a small superscript ™, then `tail`, as one line of text. Returns its width."""
    family, weight, rsize, tracking = ROLES[role]
    size = size or rsize
    for s in (head, tail, "™"):
        svg.glyphs[(family, weight)].update(s)
    fam = f"'{family}{weight}',{FALLBACK[family]}"
    ls = f' letter-spacing="{tracking * size:.2f}"' if tracking else ""
    svg.add(f'<text x="{x:g}" y="{y:g}" font-family="{fam}" font-size="{size:g}" font-weight="{weight}"{ls} '
            f'fill="{fill or svg.c["ink"]}" xml:space="preserve">{escape(head)}'
            f'<tspan font-size="{size * 0.42:.1f}" dy="{-size * 0.46:.1f}">™</tspan>'
            f'<tspan dy="{size * 0.46:.1f}">{escape(tail)}</tspan></text>')
    return measure(head, role, size) + measure("™", role, size * 0.42) + measure(tail, role, size)


def lockup(svg: Svg, x, y, mark, size, name, sub, right_text=""):
    """A product lockup: the logo, the name on one line and a quieter line under it.
    `name` is a string, or (head, tail) for a small superscript ™ after head. `sub` holds strings,
    marks and (head, tail) pairs, e.g. ["a product of", mark_kadeep_ai, ("KaDeep", " AI")]."""
    c = svg.c
    mark(svg, x, y, size)
    tx = x + size + 16
    base1, base2 = y + size * 0.50 + 1, y + size * 0.92
    if isinstance(name, tuple):
        tm_text(svg, tx, base1, *name, "name", size=31)
    else:
        svg.text(tx, base1, name, "name", size=31)
    sx = tx
    for k, part in enumerate(sub):
        if callable(part):
            sx += part(svg, sx, base2 - 15, 19) + 6
            continue
        ink = c["muted"] if k == 0 else c["ink"]
        if isinstance(part, tuple):
            sx += tm_text(svg, sx, base2, *part, "small", fill=ink) + 7
        else:
            svg.text(sx, base2, part, "small", fill=ink)
            sx += measure(part, "small") + 7
    if right_text:
        svg.text(svg.w - 56, base1, right_text, "small", fill=c["muted"], anchor="end")


# a 5x7 LED face, for the signs KaDeep Studio's dot logo belongs with
LED_FONT = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    ".": [".", ".", ".", ".", ".", ".", "#"],
    " ": ["..", "..", "..", "..", "..", "..", ".."],
}


def led_cells(text: str):
    """Lit cells (column, row) of one line of LED text, and the line's width in columns."""
    cells, col = [], 0
    for ch in text:
        glyph = LED_FONT[ch]
        cells += [(col + i, r) for r, row in enumerate(glyph) for i, v in enumerate(row) if v == "#"]
        col += len(glyph[0]) + 1
    return cells, col - 1


def led_sign(svg: Svg, x0, y0, w, h, lines, pitch, colors, lead="EVERY ", cycle=6.6):
    """Lines of LED text on a field of unlit LEDs. The word after `lead` blinks into its own colour,
    one line at a time from the top, and the sequence repeats."""
    gap_rows = 4
    widths = [led_cells(t)[1] for t in lines]
    block_w, block_h = max(widths) * pitch, (7 * len(lines) + gap_rows * (len(lines) - 1)) * pitch
    tx = x0 + (w - block_w) / 2
    ty = y0 + (h - block_h) / 2
    ox, oy = tx - math.floor((tx - x0) / pitch) * pitch, ty - math.floor((ty - y0) / pitch) * pitch
    svg.defs.append(
        f'<pattern id="led" patternUnits="userSpaceOnUse" x="{ox:.2f}" y="{oy:.2f}" width="{pitch:g}" '
        f'height="{pitch:g}"><circle cx="{pitch / 2:g}" cy="{pitch / 2:g}" r="{pitch * 0.36:.2f}" '
        f'fill="#FFFFFF" fill-opacity="0.085"/></pattern>')
    svg.add(f'<rect x="{x0:g}" y="{y0:g}" width="{w:g}" height="{h:g}" fill="url(#led)"/>')
    svg.css.append(
        "@keyframes lit{0%{opacity:0}5%{opacity:1}29%{opacity:1}35%{opacity:0}100%{opacity:0}}"
        f".lit{{opacity:0;animation:lit {cycle}s ease-in-out infinite}}")
    lead_cols = led_cells(lead)[1] + 1
    for k, (text, color) in enumerate(zip(lines, colors)):
        cells, _ = led_cells(text)
        pos = lambda cs: [(tx + (i + 0.5) * pitch, ty + (k * (7 + gap_rows) + r + 0.5) * pitch) for i, r in cs]
        word = pos([cr for cr in cells if cr[0] >= lead_cols])
        svg.dots(pos([cr for cr in cells if cr[0] < lead_cols]), pitch * 0.40, "#FFFFFF", 0.45)
        svg.dots(word, pitch * 0.40, "#FFFFFF")
        svg.add(f'<g class="lit" style="animation-delay:{k * cycle / len(lines):.2f}s">')
        svg.dots(word, pitch * 0.42, color)
        svg.add("</g>")


NINE_TO_FIVE = (  # from 9to5code's logo: a clock face lit from nine round to five
    [(15.36, 48.00, 5.38), (19.73, 31.68, 5.64), (31.68, 19.73, 5.90), (48.00, 15.36, 6.17),
     (64.32, 19.73, 6.43), (76.27, 31.68, 6.70), (80.64, 48.00, 6.96), (76.27, 64.32, 7.22)],
    (64.32, 76.27, 7.49),
    [(48.00, 80.64, 4.57), (31.68, 76.27, 4.57), (19.73, 64.32, 4.57)],
)


def mark_9to5(svg: Svg, x, y, size):
    lit, amber, dim = NINE_TO_FIVE
    s = size / 78
    ox, oy = x - 10 * s, y - 9.4 * s
    dimc = "#3d444d" if svg.theme == "dark" else "#c8d1da"
    out = []
    for (cx, cy, r), fill in [(p, svg.c["ink"]) for p in lit] + [(amber, "#E8A33D")] + [(p, dimc) for p in dim]:
        out.append(f'<circle cx="{ox + cx * s:.2f}" cy="{oy + cy * s:.2f}" r="{r * s:.2f}" fill="{fill}"/>')
    svg.add("".join(out))
    return size


def mark_x(svg: Svg, x, y, size):
    s = size / 24
    svg.add(f'<path transform="translate({x:g} {y:g}) scale({s:g})" fill="{svg.c["ink"]}" d="M18.901 1.153h3.68'
            'l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932Z'
            'M17.61 20.644h2.039L6.486 3.24H4.298Z"/>')
    return size


def mark_linkedin(svg: Svg, x, y, size):
    s = size / 24
    svg.add(f'<g transform="translate({x:g} {y:g}) scale({s:g})">'
            f'<rect width="24" height="24" rx="3.2" fill="{svg.c["ink"]}"/>'
            f'<g fill="{svg.c["chip"]}"><circle cx="5.34" cy="5.37" r="2.06"/>'
            '<rect x="3.56" y="9" width="3.56" height="11.45"/>'
            '<path d="M9.35 9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28h-3.56v-5.57'
            'c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35Z"/></g></g>')
    return size


def card_header(svg: Svg, x, y, mark, name, domain="", right=0):
    w = mark(svg, x, y - 25, 30)
    svg.text(x + w + 13, y, name, "name")
    if domain:
        svg.text(right, y, domain, "small", fill=svg.c["muted"], anchor="end")


# --------------------------------------------------------------------------- cards

CARD_R = 22


KADEEP = dict(   # as on kadeep.ai/studio and the flyers
    name=("KaDeep", " Studio Pro"), sub=["a product of", mark_kadeep_ai, ("KaDeep", " AI")],
    title=("Know when your product", "is ready to ship."),
    body=("The AI quality operation for enterprise teams.", "Faster releases, lower cost-to-quality, less risk."),
    meta="ISO/IEC 27001 and ISO 9001 certified. Enterprise-ready from day one.",
    sign=("EVERY SURFACE.", "EVERY MARKET.", "EVERY RELEASE."),
    sign_colors=("#3D8BFF", "#FF6A2B", "#2FE07A"),     # surface blue, market orange, release green
    domain="kadeep.ai/studio ↗",
)


def card_kadeep(theme: str):
    W, H, seam = 1280, 500, 660
    k = KADEEP
    svg = Svg(W, H, theme, f"KaDeep Studio Pro, a product of KaDeep AI. {' '.join(k['title'])} "
              f"{' '.join(k['body'])} {k['meta']} Beside it, an LED sign reads every surface, every market, "
              "every release, each word blinking in turn: blue, orange, then green.")
    c = svg.c
    svg.defs.append(f'<clipPath id="card"><rect width="{W}" height="{H}" rx="{CARD_R}"/></clipPath>')
    svg.add(f'<g clip-path="url(#card)"><rect width="{W}" height="{H}" fill="{c["surface"]}"/>'
            f'<rect x="{seam}" width="{W - seam}" height="{H}" fill="{KS_TILE}"/>')
    led_sign(svg, seam, 0, W - seam, H, k["sign"], 7.3, k["sign_colors"])
    svg.add("</g>")
    svg.text(W - 40, H - 34, k["domain"], "small", fill="#8b949e", anchor="end")
    x = 56
    lockup(svg, x, 44, mark_ks_tile, 58, k["name"], k["sub"])
    svg.text(x, 206, k["title"][0], "h2", size=48)
    svg.text(x, 262, k["title"][1], "h2", size=48, fill=c["muted"])
    for i, line in enumerate(k["body"]):
        svg.text(x, 330 + i * 35, line, "body")
    svg.text(x, H - 58, k["meta"], "small", fill=c["muted"])
    svg.save("card-kadeep")


GLANCE = dict(   # in the words of glance.9to5code.com
    name="Glance", sub=["for macOS 26 and later"],
    title=("Your Mac and every agent on it.", "At a glance."),
    body="A native Mac app that shows, in plain words, what each coding agent is doing and how your Mac "
         "is holding up.",
    meta="Runs entirely on your Mac. No account, no cloud.",
    domain="glance.9to5code.com ↗",
)


def card_glance(theme: str):
    W, H = 1280, 748
    g = GLANCE
    svg = Svg(W, H, theme, f"Glance, for macOS. {' '.join(g['title'])} {g['body']} {g['meta']} Below, Glance's "
              "Agents window: four coding agents, one of them asking permission to run a command.")
    c = svg.c
    svg.defs.append(f'<clipPath id="card"><rect width="{W}" height="{H}" rx="{CARD_R}"/></clipPath>')
    data, iw, ih = image_b64(SRC / f"glance-agents-{theme}.png", 1300, quality=84, alpha=True)
    sw = W - 112
    svg.add(f'<g clip-path="url(#card)"><rect width="{W}" height="{H}" fill="{c["surface"]}"/>'
            f'<image x="56" y="344" width="{sw}" height="{sw * ih / iw:.1f}" '
            f'href="data:image/webp;base64,{data}"/></g>')
    x = 56
    lockup(svg, x, 44, mark_glance, 58, g["name"], g["sub"], right_text=g["domain"])
    svg.text(x, 200, g["title"][0], "title")
    svg.text(x, 246, g["title"][1], "title", fill=c["muted"])
    bx = max(712, x + measure(g["title"][0], "title") + 56)
    end = svg.paragraph(bx, 200, g["body"], "body", W - 56 - bx, 35)
    svg.text(bx, end + 52, g["meta"], "small", fill=c["muted"])
    svg.save("card-glance")


def chip(name: str, label: str, mark, mark_w: float, mark_h: float = 24):
    """A link with its logo: a pill with a hairline edge and a soft shadow, sized to its label."""
    def build(theme):
        h, ph, pad, gap, after = 54, 46, 17, 10, 12
        w = pad + mark_w + gap + measure(label, "chip") + pad + 2
        svg = Svg(math.ceil(w + 2 + after), h, theme, label)
        c = svg.c
        flood, op = c["chip_shadow"]
        svg.defs.append(f'<filter id="sh" x="-10%" y="-30%" width="120%" height="170%">'
                        f'<feDropShadow dx="0" dy="1.2" stdDeviation="1.4" flood-color="{flood}" flood-opacity="{op}"/></filter>')
        top = (h - ph) / 2 - 1
        svg.add(f'<rect x="1" y="{top:g}" width="{w:.1f}" height="{ph}" rx="{ph / 2:g}" fill="{c["chip"]}" '
                f'stroke="{c["chip_line"]}" stroke-width="1" filter="url(#sh)"/>')
        mark(svg, 1 + pad, top + (ph - mark_h) / 2, mark_h)
        svg.text(1 + pad + mark_w + gap, top + ph / 2 + 6, label, "chip")
        svg.save(f"link-{name}")
    return build


# the art at the top of the Hosh and Exhibit cards, each drawn into a box (ox, 0, w, h)

def art_exhibit(svg: Svg, ox, w, h):
    """Exhibit's homepage hero: the real case page, its evidence graph at 88% Fight, on Harnett's letter rack."""
    data, iw, ih = image_b64(SRC / "exhibit-home.jpg", 1260, quality=86)
    svg.add(f'<image x="{ox}" y="0" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" '
            f'href="data:image/webp;base64,{data}"/>')


def art_hosh(svg: Svg, ox, w, h):
    """Hosh's level, keeping still under a market that never does. Up moves are green, down moves red."""
    dark = svg.theme == "dark"
    bg, ink = ("#0E0F11", "#F0F6FC") if dark else ("#F7F7F5", "#1F2328")
    flat, area_c, label_c = ("#6E7681", "#262B32", "#8B949E") if dark else ("#8C959F", "#DADFE4", "#6E7781")
    red, green = "#F2555A", "#2FBF71"
    P = PITCH
    svg.add(f'<rect x="{ox}" y="0" width="{w}" height="{h}" fill="{bg}"/>')
    # whole-number waves of the box width, so the drift loops without a seam
    rnd = random.Random(9)
    waves = [(k, a, rnd.uniform(0, 2 * math.pi)) for k, a in ((1, 1.0), (2, .66), (3, .5), (5, .4), (8, .3),
                                                               (13, .22), (21, .15), (34, .1))]
    cols = int(w // P)
    def row(i):
        x = (i + 0.5) / cols
        v = sum(a * math.sin(2 * math.pi * k * x + p) for k, a, p in waves) / 3.3
        return round((h * 0.33 + v * h * 0.24) / P - 0.5)
    rows = [row(i % cols) for i in range(2 * cols + 1)]
    ups, downs, flats, area = [], [], [], []
    for i in range(2 * cols):
        r0, r1 = rows[i], rows[i + 1]
        x = ox + (i + 0.5) * P
        run = range(min(r0, r1), max(r0, r1) + 1) if abs(r1 - r0) > 1 else [r0]
        bucket = ups if r1 < r0 else downs if r1 > r0 else flats
        bucket += [(x, (r + 0.5) * P) for r in run]
        area += [(x, (r + 0.5) * P) for r in range(max(r0, r1) + 2, int(h // P)) if (r - r0) % 2 == 0]
    svg.css.append(drift_css("mkt", int(cols * P), 52))
    svg.add('<g class="mkt">')
    svg.dots(area, P * 0.15, area_c)
    svg.dots(flats, P * 0.36, flat)
    svg.dots(ups, P * 0.36, green)
    svg.dots(downs, P * 0.36, red)
    svg.add("</g>")
    # the level: a glass vial, fear on the left, greed on the right, and a bubble that stays centred
    cx, cy, vw, vh = ox + w / 2, h * 0.68, 404, 116
    svg.add(f'<rect x="{cx - vw / 2 - 16:g}" y="{cy - vh / 2 - 16:g}" width="{vw + 32}" height="{vh + 32}" '
            f'rx="{vh / 2 + 16:g}" fill="{bg}"/>')
    ring, r_ = [], vh / 2
    straight = vw - 2 * r_
    perim = 2 * straight + 2 * math.pi * r_
    n = int(perim // 9)
    for k in range(n):
        d = k * perim / n
        if d < straight:
            ring.append((cx - straight / 2 + d, cy - r_))
        elif d < straight + math.pi * r_:
            a = (d - straight) / r_ - math.pi / 2
            ring.append((cx + straight / 2 + r_ * math.cos(a), cy + r_ * math.sin(a)))
        elif d < 2 * straight + math.pi * r_:
            ring.append((cx + straight / 2 - (d - straight - math.pi * r_), cy + r_))
        else:
            a = (d - 2 * straight - math.pi * r_) / r_ + math.pi / 2
            ring.append((cx - straight / 2 + r_ * math.cos(a), cy + r_ * math.sin(a)))
    svg.dots(ring, 3.3, ink)
    def pill(px, color):
        svg.dots([(px + i * P, cy + j * P) for i in (-1.5, -0.5, 0.5, 1.5) for j in (-3, -2, -1, 0, 1, 2, 3)
                  if not (abs(i) == 1.5 and abs(j) == 3)], 3.2, color)
    pill(cx - vw / 2 + 50, red)
    pill(cx + vw / 2 - 50, green)
    bubble = [(cx + i * P, cy + j * P) for i in range(-6, 7) for j in range(-4, 5)
              if (i * P / 50) ** 2 + (j * P / 30) ** 2 <= 1]
    svg.css.append("@keyframes keep{0%,100%{transform:translateX(0)}22%{transform:translateX(-7px)}"
                   "44%{transform:translateX(5px)}62%{transform:translateX(-2px)}78%{transform:translateX(1px)}}"
                   ".keep{animation:keep 7s ease-in-out infinite}")
    svg.add('<g class="keep">')
    svg.dots(bubble, 3.2, ink)
    svg.add("</g>")
    for px, word in ((cx - vw / 2 + 50, "fear"), (cx + vw / 2 - 50, "greed")):
        svg.text(px, cy + vh / 2 + 40, word, "small", size=16, fill=label_c, anchor="middle")


HALF = dict(W=640, IMG_H=464)
HALF_COPY = {
    "hosh": dict(
        mark=mark_hosh, name="Hosh", domain="hoshhq.com ↗", art=art_hosh,
        title="Keep your head when markets don’t.",
        body="An AI trading copilot for Indian equities and Bitcoin. Code proposes every trade, "
             "AI can only veto or shrink it, and hard risk limits sit under both.",
        meta="Private alpha. Paper trading by default.",
        alt="Hosh. Keep your head when markets don't. An AI trading copilot for Indian equities and Bitcoin. "
            "Above it, Hosh's spirit level drawn in dots, fear on the left and greed on the right, its bubble "
            "holding still under a red and green market line that keeps moving.",
    ),
    "exhibit": dict(
        mark=mark_exhibit, name="Exhibit", domain="exhibit.9to5code.com ↗", art=art_exhibit,
        title="Disputes, answered with evidence.",
        body="An AI dispute agent for PayPal merchants. It investigates every case, cites each claim "
             "to a real record, and refunds when the buyer is right.",
        meta="With Someet Shende, for the PayPal AI Hackathon 2026.",
        alt="Exhibit. Disputes, answered with evidence. An AI dispute agent for PayPal merchants. Above it, "
            "Exhibit's case page, its evidence graph at 88 percent likely to win, on Harnett's painting "
            "The Artist's Letter Rack.",
    ),
}


def half_layout():
    """Both half cards share one height so they sit level side by side."""
    W, IMG_H = HALF["W"], HALF["IMG_H"]
    col = W - 12 - 2 * 56
    heights = []
    for copy in HALF_COPY.values():
        t = len(wrap_balanced(copy["title"], "title", col))
        b = len(wrap(copy["body"], "body", col))
        heights.append((t, b))
    t_max = max(t for t, _ in heights)
    b_max = max(b for _, b in heights)
    title_y = IMG_H + 150
    body_y = title_y + (t_max - 1) * 46 + 56
    meta_y = body_y + (b_max - 1) * 35 + 70
    return dict(H=meta_y + 52, title_y=title_y, body_y=body_y, meta_y=meta_y, col=col)


def half_card(key: str):
    def build(theme):
        copy = HALF_COPY[key]
        L = half_layout()
        W, H, IMG_H = HALF["W"], L["H"], HALF["IMG_H"]
        svg = Svg(W, H, theme, copy["alt"])
        c = svg.c
        gut = 12
        ox = 0 if key == "hosh" else gut          # the gutter sits on the inner side of each card
        cw = W - gut
        svg.defs.append(f'<clipPath id="card"><rect x="{ox}" width="{cw}" height="{H}" rx="{CARD_R}"/></clipPath>')
        svg.add(f'<g clip-path="url(#card)"><rect x="{ox}" width="{cw}" height="{H}" fill="{c["surface"]}"/>')
        copy["art"](svg, ox, cw, IMG_H)
        svg.add("</g>")
        x = ox + 56
        card_header(svg, x, IMG_H + 74, copy["mark"], copy["name"], copy["domain"], ox + cw - 56)
        for i, line in enumerate(wrap_balanced(copy["title"], "title", L["col"])):
            svg.text(x, L["title_y"] + i * 46, line, "title")
        svg.paragraph(x, L["body_y"], copy["body"], "body", L["col"], 35)
        svg.text(x, L["meta_y"], copy["meta"], "small", fill=c["muted"])
        svg.save(f"card-{key}")
    return build


def card_datasphere(theme: str):
    W, H = 1280, 430
    svg = Svg(W, H, theme, "DataSphere. Still in the fog. Converging since 2023. A dot sphere sits behind "
              "drifting fog.")
    c = svg.c
    blue = "#4C8DFF" if theme == "dark" else "#2563EB"
    svg.defs.append(f'<clipPath id="card"><rect width="{W}" height="{H}" rx="{CARD_R}"/></clipPath>')
    svg.add(f'<g clip-path="url(#card)"><rect width="{W}" height="{H}" fill="{c["surface"]}"/>')
    # the sphere, then fog over it
    pts = sphere_dots(884, 252, 148, PITCH)
    for r, ps in quantize(pts, 0.25).items():
        svg.dots(ps, r, blue, 0.95)
    fx, fy = grid(560, 0, W - 560, H, PITCH)
    n = fbm(fx, fy, 11, W)
    fband = smooth(150, 330, fy) * smooth(560, 720, fx)
    k = fband * (0.25 + 0.75 * n)
    fpts = [(fx[j, i], fy[j, i], PITCH * 0.44 * k[j, i]) for j in range(fx.shape[0])
            for i in range(fx.shape[1]) if k[j, i] > 0.14]
    for r, ps in sorted(quantize(fpts).items()):
        svg.dots(ps, r, c["fog"], c["fog_op"] * 0.95)
    svg.add("</g>")
    fog_wisps(svg, 560, 144, W - 560, H - 144, seed=5, density=0.47, top_fade=80, bottom_fade=10,
              op=c["wisp_op"] + 0.12, seconds=90)
    x = 56
    card_header(svg, x, 96, mark_datasphere, "DataSphere")
    svg.text(W - 56, 96, "datasphere.9to5code.com ↗", "small", fill=c["muted"], anchor="end")
    svg.text(x, 178, "Still in the fog.", "title")
    svg.paragraph(x, 232, "Converging since 2023. Every source lies a little, and it does the math. "
                  "That is as much as I can say here.", "body", 520, 35)
    svg.text(x, H - 58, "Its page has a passphrase, if you read slowly.", "small", fill=c["muted"])
    svg.save("card-datasphere")


# --------------------------------------------------------------------------- the bet


def bet(theme: str):
    """Making got cheap, trust didn't: two curves since 2019 and the orange gap between them."""
    W, X1, YT, YB = 1280, 930, 262, 668
    H = YB + 64
    svg = Svg(W, H, theme, "The bet. Making got cheap. Trust didn’t. A chart drawn in dots: since 2019, "
              "what AI can make has climbed steeply while what teams can trust has barely moved. The gap "
              "between them is filled in orange: the gap I build in.")
    c = svg.c
    svg.css.append("@keyframes now{0%,100%{opacity:1}50%{opacity:.2}}.now{animation:now 1.8s ease-in-out infinite}")
    svg.text(0, 82, "The bet.", "h2")
    svg.text(0, 146, "Making got cheap. Trust didn’t.", "h2", fill=c["muted"])
    P = PITCH
    floor = YB - P / 2
    gx, gy = grid(0, YT - 2 * P, X1, floor - YT + 3 * P, P)
    t = (gx - P / 2) / (X1 - P)
    make = floor - (floor - YT) * t ** 2.1
    check = floor - (floor - YT) * 0.22 * t ** 1.15
    gap, made_edge, checked = [], [], []
    for j in range(gx.shape[0]):
        for i in range(gx.shape[1]):
            x, y, mk, ck = gx[j, i], gy[j, i], make[j, i], check[j, i]
            if y > floor + 0.1:
                continue
            if y >= ck - P / 2:
                checked.append((x, y))
            elif y >= mk - P / 2:
                depth = min(1.0, max(0.0, (y - mk) / max(ck - mk, 1)))
                gap.append((x, y, P * 0.47 * (1 - 0.6 * depth ** 0.8)))
            if mk - P / 2 <= y < mk + P / 2 and y < ck - P / 2:
                made_edge.append((x, y))
    svg.dots(checked, P * 0.24, c["muted"], 0.55)
    for r, ps in sorted(quantize(gap).items()):
        svg.dots(ps, r, c["orange"])
    svg.dots(made_edge, P * 0.47, c["ink"])
    # the end of each curve, labelled
    lx = X1 + 30
    y_make, y_check = make[0, -1], check[0, -1]
    svg.text(lx, y_make + 9, "What AI can make", "lede")
    svg.text(lx, (y_make + y_check) / 2 + 9, "The gap I build in.", "lede", fill=c["orange_text"])
    svg.text(lx, y_check + 9, "What teams can trust", "lede", fill=c["muted"])
    svg.add(f'<circle class="now" cx="{gx[0, -1]:g}" cy="{y_make:.1f}" r="{P * 0.9:g}" fill="{c["orange"]}"/>')
    svg.text(0, H - 14, "2019", "small", fill=c["muted"])
    svg.text(X1, H - 14, "Now", "small", fill=c["muted"], anchor="end")
    svg.save("bet")


# --------------------------------------------------------------------------- footer


def footer(theme: str):
    from PIL import ImageDraw, ImageFont
    W, H = 1280, 360
    svg = Svg(W, H, theme, "Shivansh, drawn in dots, with a blinking cursor. Built with agents. Shipped with evidence.")
    c = svg.c
    svg.css.append("@keyframes cur{0%,49%{opacity:1}50%,100%{opacity:0}}.cur{animation:cur 1.1s steps(1) infinite}")
    word = "Shivansh"
    size = 300
    while True:                      # as large as fits beside the cursor
        font = ImageFont.truetype(io.BytesIO(static_font_bytes("Geist", 600)), size)
        box = font.getbbox(word)
        if box[2] - box[0] + 4 + 22 + 5 * PITCH <= W:
            break
        size -= 4
    pitch = PITCH
    box = font.getbbox(word)
    tw, th = box[2] - box[0], box[3] - box[1]
    img = Image.new("L", (tw + 4, th + 4), 0)
    ImageDraw.Draw(img).text((2 - box[0], 2 - box[1]), word, font=font, fill=255)
    cols, rows = img.width // pitch, img.height // pitch
    cov = np.asarray(img.resize((cols, rows), Image.BOX)).astype(np.float32) / 255
    x0, y0 = 0, 40
    pts = []
    for j in range(rows):
        for i in range(cols):
            if cov[j, i] > 0.12:
                pts.append((x0 + (i + 0.5) * pitch, y0 + (j + 0.5) * pitch, pitch * 0.45 * min(1, 0.35 + cov[j, i])))
    band_y = y0 + PITCH * int(rows * 0.45)
    fog_wisps(svg, 0, band_y, W, PITCH * int(rows * 0.62), seed=17, density=0.5,
              top_fade=70, bottom_fade=30, seconds=96)
    ink = c["ink"]
    for r, ps in sorted(quantize(pts).items()):
        svg.dots(ps, r, ink, 0.9)
    # cursor: a block of orange dots after the last letter
    cx0 = x0 + cols * pitch + 22
    base = y0 + rows * pitch
    cur = [(cx0 + (i + 0.5) * pitch, base - (j + 0.5) * pitch) for i in range(5) for j in range(int(rows * 0.62))]
    svg.add('<g class="cur">')
    svg.dots(cur, pitch * 0.45, c["orange"])
    svg.add("</g>")
    svg.text(0, H - 30, "Built with agents. Shipped with evidence.", "small", fill=c["muted"])
    svg.save("footer")


# --------------------------------------------------------------------------- main

BUILDERS = {
    "hero": hero,
    "link-kadeep": chip("kadeep", "KaDeep AI", mark_kadeep_ai, 24),
    "link-9to5code": chip("9to5code", "9to5code", mark_9to5, 24),
    "link-x": chip("x", "@shvuuuu", mark_x, 20, 20),
    "link-linkedin": chip("linkedin", "LinkedIn", mark_linkedin, 22, 22),
    "h-building": heading("h-building", "What I’m building.", "Products for the agent era."),
    "card-kadeep": card_kadeep,
    "card-glance": card_glance,
    "card-hosh": half_card("hosh"),
    "card-exhibit": half_card("exhibit"),
    "card-datasphere": card_datasphere,
    "bet": bet,
    "h-hi": heading("h-hi", "Say hi.", "Bring the job you’d never hand to an AI."),
    "footer": footer,
}


def main(argv):
    names = argv or list(BUILDERS)
    for theme in THEMES:
        for name in names:
            BUILDERS[name](theme)


if __name__ == "__main__":
    main(sys.argv[1:])
