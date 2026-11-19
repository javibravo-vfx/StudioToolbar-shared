"""
studio_lookdev.library
======================

Maintenance of the shared HDRI library, DCC-agnostic.

For every HDRI of a library it can generate, once:
  <cache>/<name>.thumb.jpg  512x256 preview, auto-exposed, sRGB
  <cache>/<name>.bg.exr     blurred low-res copy, used as camera background
  <name>.tx                 Arnold mipmapped texture next to the source (only if maketx is found)
<cache> is shared/lookdev/hdri/_cache for the studio library and
shared/lookdev/_library_cache/<id> for any other folder — both on the studio drive, so every
toolbar and every machine uses them, and a collection on the NAS is never written to
(except .tx, only when asked).

Files are regenerated when the source HDRI is newer than the cached file.

A DCC only provides image I/O:
    read_linear(path, w, h) -> flat list of linear floats [r, g, b, r, g, b, ...]
    write_jpg(path, rgb8_bytes, w, h)
    write_exr(path, flat_linear_floats, w, h)
and calls refresh(); all the decisions and the pixel math live here.
"""

from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from . import core

THUMB_W, THUMB_H = 512, 256
BG_W, BG_H = 512, 256
BG_BLOCK = 16                 # blur strength: average over BG_BLOCK x BG_BLOCK pixels


# ---------------------------------------------------------------------------
# What is missing
# ---------------------------------------------------------------------------

def _stale(target: str, source: str) -> bool:
    if not os.path.isfile(target):
        return True
    try:
        return os.path.getmtime(target) < os.path.getmtime(source)
    except OSError:
        return True


@dataclass
class Job:
    hdri: core.HDRI
    thumb: bool = False
    bg: bool = False
    tx: bool = False

    @property
    def any(self) -> bool:
        return self.thumb or self.bg or self.tx


def pending_jobs(folder: str = core.HDRI_DIR, want_tx: bool = False,
                 force: bool = False) -> List[Job]:
    """HDRIs with something to (re)generate. Hand-made thumbnails are never overwritten."""
    jobs = []
    for h in core.list_hdris(folder):
        cached_thumb = core.thumb_cache_path(h.path, h.root)
        handmade = h.thumb is not None and core._norm_path(h.thumb) != core._norm_path(cached_thumb)
        j = Job(h,
                thumb=not handmade and (force or _stale(cached_thumb, h.path)),
                bg=force or _stale(core.bg_cache_path(h.path, h.root), h.path),
                tx=want_tx and (force or _stale(core.tx_path(h.path), h.path)))
        if j.any:
            jobs.append(j)
    return jobs


# ---------------------------------------------------------------------------
# Pixel math (pure Python, small images only)
# ---------------------------------------------------------------------------

def _lum(r: float, g: float, b: float) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def auto_exposure(rgb: Sequence[float], key: float = 0.18) -> float:
    """Scale that maps the log-average luminance to `key` (photographic auto-exposure)."""
    n = len(rgb) // 3
    if not n:
        return 1.0
    acc = 0.0
    for i in range(0, n * 3, 3):
        acc += math.log(1e-4 + max(0.0, _lum(rgb[i], rgb[i + 1], rgb[i + 2])))
    avg = math.exp(acc / n)
    return key / max(avg, 1e-6)


def _srgb_oetf(v: float) -> float:
    return 12.92 * v if v <= 0.0031308 else 1.055 * (v ** (1.0 / 2.4)) - 0.055


def tonemap_srgb8(rgb: Sequence[float], exposure: Optional[float] = None,
                  white: float = 6.0) -> bytes:
    """
    Linear HDR floats -> 8-bit sRGB bytes (RGB888) for thumbnails.
    Auto-exposure, then extended Reinhard on luminance (keeps hue), then sRGB.
    """
    scale = auto_exposure(rgb) if exposure is None else exposure
    w2 = white * white
    out = bytearray(len(rgb))
    for i in range(0, len(rgb), 3):
        r, g, b = (max(0.0, rgb[i] * scale), max(0.0, rgb[i + 1] * scale),
                   max(0.0, rgb[i + 2] * scale))
        l = _lum(r, g, b)
        if l > 0.0:
            ld = l * (1.0 + l / w2) / (1.0 + l)
            k = ld / l
            r, g, b = r * k, g * k, b * k
        for j, v in enumerate((r, g, b)):
            out[i + j] = max(0, min(255, int(round(_srgb_oetf(min(1.0, v)) * 255.0))))
    return bytes(out)


def blur_latlong(rgb: Sequence[float], w: int, h: int, block: int = BG_BLOCK) -> List[float]:
    """
    Soft background from a lat-long image: block average, then bilinear upsample back to
    w x h. Wraps horizontally so the seam at +-180° stays invisible.
    """
    bw, bh = max(1, w // block), max(1, h // block)
    small = [0.0] * (bw * bh * 3)
    for by in range(bh):
        for bx in range(bw):
            acc = [0.0, 0.0, 0.0]
            cnt = 0
            for y in range(by * block, min(h, (by + 1) * block)):
                row = y * w
                for x in range(bx * block, min(w, (bx + 1) * block)):
                    i = (row + x) * 3
                    acc[0] += rgb[i]; acc[1] += rgb[i + 1]; acc[2] += rgb[i + 2]
                    cnt += 1
            o = (by * bw + bx) * 3
            small[o:o + 3] = [a / max(cnt, 1) for a in acc]

    out = [0.0] * (w * h * 3)
    for y in range(h):
        fy = min(max((y + 0.5) / h * bh - 0.5, 0.0), bh - 1.0)
        y0 = int(fy); y1 = min(y0 + 1, bh - 1); ty = fy - y0
        for x in range(w):
            fx = (x + 0.5) / w * bw - 0.5
            x0 = int(math.floor(fx)); tx = fx - x0
            xa, xb = x0 % bw, (x0 + 1) % bw
            o = (y * w + x) * 3
            for c in range(3):
                a = small[(y0 * bw + xa) * 3 + c] * (1 - tx) + small[(y0 * bw + xb) * 3 + c] * tx
                b = small[(y1 * bw + xa) * 3 + c] * (1 - tx) + small[(y1 * bw + xb) * 3 + c] * tx
                out[o + c] = a * (1 - ty) + b * ty
    return out


# ---------------------------------------------------------------------------
# Arnold maketx
# ---------------------------------------------------------------------------

def find_maketx(extra_dirs: Sequence[str] = ()) -> Optional[str]:
    """maketx from PATH, $ARNOLD_PATH/bin or the given folders (e.g. MAXtoA plugin dirs)."""
    exe = "maketx.exe" if os.name == "nt" else "maketx"
    found = shutil.which("maketx")
    if found:
        return found
    dirs = list(extra_dirs)
    if os.environ.get("ARNOLD_PATH"):
        dirs.append(os.path.join(os.environ["ARNOLD_PATH"], "bin"))
    for d in dirs:
        for cand in glob.glob(os.path.join(d, exe)) + glob.glob(os.path.join(d, "*", exe)):
            if os.path.isfile(cand):
                return cand
    return None


def make_tx(maketx: str, src: str, dst: str) -> None:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run([maketx, "-v", "--oiio", "--filter", "lanczos3", src, "-o", dst],
                   check=True, capture_output=True, creationflags=flags)


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

@dataclass
class Report:
    done: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def refresh(read_linear: Callable[[str, int, int], List[float]],
            write_jpg: Callable[[str, bytes, int, int], None],
            write_exr: Callable[[str, List[float], int, int], None],
            folder: str = core.HDRI_DIR, maketx: Optional[str] = None, force: bool = False,
            progress: Optional[Callable[[int, int, str], bool]] = None) -> Report:
    """
    Generates what is missing. `progress(i, total, label)` may return False to cancel.
    """
    jobs = pending_jobs(folder, want_tx=bool(maketx), force=force)
    rep = Report()
    for i, job in enumerate(jobs):
        h = job.hdri
        if progress and progress(i, len(jobs), h.label) is False:
            break
        try:
            if job.thumb or job.bg:
                thumb, bg = core.thumb_cache_path(h.path, h.root), core.bg_cache_path(h.path, h.root)
                os.makedirs(os.path.dirname(thumb), exist_ok=True)
                px = read_linear(h.path, THUMB_W, THUMB_H)
                if job.thumb:
                    write_jpg(thumb, tonemap_srgb8(px), THUMB_W, THUMB_H)
                if job.bg:
                    write_exr(bg, blur_latlong(px, BG_W, BG_H), BG_W, BG_H)
            if job.tx:
                make_tx(maketx, h.path, core.tx_path(h.path))
            rep.done.append(h.name)
        except Exception as exc:  # noqa: BLE001
            rep.errors.append("%s: %s" % (h.name, exc))
    if progress:
        progress(len(jobs), len(jobs), "")
    return rep
