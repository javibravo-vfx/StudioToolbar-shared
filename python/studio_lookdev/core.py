"""
studio_lookdev.core
===================

DCC-agnostic logic of the Studio Toolbar Lookdev Rig.

This module imports nothing from 3ds Max, Houdini, Blender or Maya. It only:
  - Resolves the shared library paths (HDRIs, presets, references).
  - Reads presets and HDRI sidecars.
  - Converts the ColorChecker to the working space (ACEScg / lin_srgb).
  - Computes the camera framing from the asset bounding box (new camera or an existing one).
  - Computes where the balls and chart go, in camera space.
  - Computes the ground (shadow catcher) plane and the turntable timing.

Camera-space convention (same in Max, Houdini, Maya and Blender):
  +X right, +Y up, the camera looks down -Z.

Each DCC only implements node / material creation and reads everything else from here,
so the same preset gives the same result in every package.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.environ.get("STUDIO_TOOLBAR_ROOT", r"T:\.studio-toolbar")
LOOKDEV_DIR = os.environ.get("STUDIO_LOOKDEV_DIR", os.path.join(ROOT, "shared", "lookdev"))
HDRI_DIR = os.path.join(LOOKDEV_DIR, "hdri")
PRESET_DIR = os.path.join(LOOKDEV_DIR, "presets")
REFERENCE_DIR = os.path.join(LOOKDEV_DIR, "reference")

HDRI_EXTS = (".exr", ".hdr")
THUMB_EXTS = (".jpg", ".jpeg", ".png")
CACHE_DIRNAME = "_cache"     # generated files: <name>.thumb.jpg, <name>.bg.exr

RIG_PREFIX = "LKD_"          # prefix of every node the rig creates
RIG_LAYER = "LOOKDEV_RIG"    # layer / collection / group holding the rig
USER_PROP_HDRI = "lookdev_hdri"
USER_PROP_PRESET = "lookdev_preset"
DEFAULT_CHART = "colorchecker24"   # chart definition in lookdev/reference/


# ---------------------------------------------------------------------------
# JSON / presets
# ---------------------------------------------------------------------------

DEFAULT_PRESET: Dict = {
    "working_space": "ACEScg",
    "camera": {
        "focal_mm": 50.0,
        "film_width_mm": 36.0,
        "framing_margin": 1.15,
        "elevation_deg": 8.0,
    },
    "refs": {
        "items": ["chrome", "grey18", "colorchecker"],
        "corner": "bottom_right",     # bottom_right / bottom_left / top_right / top_left
        "chart": DEFAULT_CHART,       # chart definition in reference/
        "depth_factor": 0.5,
        "ball_size_screen": 0.09,
        "chart_height_factor": 1.0,
        "gap": 0.18,
        "padding_screen": 0.035,
        "cast_shadows": False,
    },
    "ground": {
        "shadow_catcher": True,
        "size_factor": 12.0,     # side of the ground plane = bounding radius * size_factor
        "offset": 0.0,           # push the plane down (fraction of the radius) if it z-fights
    },
    "clay": {
        "enabled": False,        # start with the clay override on
        "albedo": 0.18,
    },
    "turntable": {
        "enabled": True,
        "start_frame": 1001,
        "object_frames": 120,
        "hdri_frames": 120,
        "object_spin": "ccw",
    },
    "light": {"intensity": 1.0},
}

DEFAULT_SIDECAR: Dict = {
    "label": None,
    "rotation_offset_deg": 0.0,
    "intensity": 1.0,
    "colorspace": None,
    "has_sun": False,
    "tags": [],
}


def load_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _deep_merge(base: Dict, over: Dict) -> Dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def merge(preset: Dict, overrides: Optional[Dict]) -> Dict:
    """Preset with `overrides` merged on top (keeps the preset name)."""
    out = _deep_merge(preset, overrides or {})
    out["_name"] = preset.get("_name")
    return out


def list_presets(folder: str = PRESET_DIR) -> Dict[str, str]:
    """{name: path} of the available .json presets."""
    result = {}
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(".json") and not fn.startswith("_"):
                result[os.path.splitext(fn)[0]] = os.path.join(folder, fn)
    return result


def preset_labels(folder: str = PRESET_DIR) -> Dict[str, str]:
    """{preset name: label to show} — label from the preset's "label" key, else its name."""
    out = {}
    for name, path in list_presets(folder).items():
        data = load_json(path, {}) or {}
        out[name] = data.get("label") or name
    return out


def load_preset(name: str = "turntable_default", folder: str = PRESET_DIR) -> Dict:
    """Preset merged over the defaults (missing keys = default)."""
    path = os.path.join(folder, name + ".json")
    data = load_json(path, {}) or {}
    preset = _deep_merge(DEFAULT_PRESET, data)
    preset["_name"] = name
    return preset


# ---------------------------------------------------------------------------
# HDRI library
# ---------------------------------------------------------------------------

# Previews / blurred backgrounds / defaults of any folder other than the studio library.
# On the studio drive so they are generated once for everybody and easy to wipe.
LIBRARY_CACHE_ROOT = os.path.join(LOOKDEV_DIR, "_library_cache")
LOCAL_CACHE_ROOT = LIBRARY_CACHE_ROOT        # old name, kept for compatibility


def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


def is_default_library(root: str) -> bool:
    return _norm_path(root) == _norm_path(HDRI_DIR)


def library_cache_dir(root: str = HDRI_DIR) -> str:
    """
    Where generated files live for a library:
      default studio library -> <hdri>/_cache (shared by everybody)
      any other folder (e.g. a collection on the NAS) -> shared/lookdev/_library_cache/<id>,
      so nothing is ever written into that collection. Delete that folder any time:
      previews are regenerated by Refresh library.
    """
    if is_default_library(root):
        return os.path.join(root, CACHE_DIRNAME)
    import hashlib
    key = hashlib.md5(_norm_path(root).encode("utf-8")).hexdigest()[:8]
    slug = "".join(c if c.isalnum() else "_" for c in os.path.basename(os.path.normpath(root)))
    return os.path.join(LOCAL_CACHE_ROOT, "%s_%s" % (slug or "library", key))


def cache_dir(folder: str = HDRI_DIR) -> str:          # kept for compatibility
    return library_cache_dir(folder)


def _root_for(hdri_path: str, root: Optional[str]) -> str:
    if root:
        return root
    if _norm_path(hdri_path).startswith(_norm_path(HDRI_DIR) + os.sep):
        return HDRI_DIR
    return os.path.dirname(hdri_path)


def _cache_base(hdri_path: str, root: Optional[str] = None) -> str:
    root = _root_for(hdri_path, root)
    rel = os.path.splitext(os.path.relpath(hdri_path, root))[0]
    return os.path.join(library_cache_dir(root), rel)


def thumb_cache_path(hdri_path: str, root: Optional[str] = None) -> str:
    return _cache_base(hdri_path, root) + ".thumb.jpg"


def bg_cache_path(hdri_path: str, root: Optional[str] = None) -> str:
    return _cache_base(hdri_path, root) + ".bg.exr"


def tx_path(hdri_path: str) -> str:
    """Arnold .tx next to the source (where Arnold's auto-tx looks for it)."""
    return os.path.splitext(hdri_path)[0] + ".tx"


def sidecar_path(hdri_path: str, root: Optional[str] = None) -> str:
    """
    Where "Save as HDRI default" writes: next to the HDRI in the studio library,
    in the local cache for any other folder (the collection itself is never modified).
    """
    root = _root_for(hdri_path, root)
    if is_default_library(root):
        return os.path.splitext(hdri_path)[0] + ".json"
    return _cache_base(hdri_path, root) + ".json"


@dataclass
class HDRI:
    name: str                       # path relative to the library root, "/" separated, no ext
    path: str
    thumb: Optional[str] = None     # hand-made thumbnail next to the HDRI, else the cached one
    meta: Dict = field(default_factory=dict)
    bg: Optional[str] = None        # blurred low-res background (generated)
    root: str = HDRI_DIR            # library the HDRI belongs to
    group: str = ""                 # sub-folder inside the library ("" = top level)

    @property
    def label(self) -> str:
        return self.meta.get("label") or os.path.basename(self.name)

    @property
    def rotation_offset(self) -> float:
        return float(self.meta.get("rotation_offset_deg", 0.0))

    @property
    def intensity(self) -> float:
        return float(self.meta.get("intensity", 1.0))


def _skip_dir(d: str) -> bool:
    return d.startswith("_") or d.startswith(".")


def hdri_from_path(path: str, root: Optional[str] = None) -> HDRI:
    """HDRI record (sidecar, thumbnail, blurred bg) for one file of a library."""
    root = _root_for(path, root)
    rel = os.path.relpath(path, root)
    base = os.path.splitext(path)[0]
    name = os.path.splitext(rel)[0].replace(os.sep, "/")
    group = os.path.dirname(rel).replace(os.sep, "/")
    meta = dict(DEFAULT_SIDECAR)
    for side in (base + ".json", sidecar_path(path, root)):      # next to it, then cache
        if os.path.isfile(side):
            meta = _deep_merge(meta, load_json(side, {}) or {})
    thumb = None
    for te in THUMB_EXTS:
        if os.path.isfile(base + te):
            thumb = base + te
            break
    if thumb is None and os.path.isfile(thumb_cache_path(path, root)):
        thumb = thumb_cache_path(path, root)
    bg = bg_cache_path(path, root)
    return HDRI(name=name, path=path, thumb=thumb, meta=meta,
                bg=bg if os.path.isfile(bg) else None, root=root, group=group)


def list_hdris(folder: str = HDRI_DIR, recursive: bool = True) -> List[HDRI]:
    """
    HDRIs of a library (the studio one, or any folder such as a collection on the NAS).
    Sub-folders become groups; folders starting with "_" or "." are skipped.
    """
    items: List[HDRI] = []
    if not os.path.isdir(folder):
        return items
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted((d for d in dirnames if not _skip_dir(d)), key=str.lower)
        for fn in sorted(filenames, key=str.lower):
            base, ext = os.path.splitext(fn)
            if ext.lower() in HDRI_EXTS and not base.startswith("_"):
                items.append(hdri_from_path(os.path.join(dirpath, fn), folder))
        if not recursive:
            break
    return items


def save_sidecar(hdri_path: str, updates: Dict, root: Optional[str] = None) -> Dict:
    """Merges `updates` into the HDRI sidecar, keeping other fields (see sidecar_path)."""
    path = sidecar_path(hdri_path, root)
    data = load_json(path, {}) or {}
    if not os.path.isfile(path):          # start from a hand-made sidecar next to the HDRI
        data = load_json(os.path.splitext(hdri_path)[0] + ".json", {}) or {}
    data.update(updates)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    return data


def find_hdri(name: str, folder: str = HDRI_DIR) -> Optional[HDRI]:
    """By library name ("sub/folder/file" or just "file")."""
    items = list_hdris(folder)
    for h in items:
        if h.name == name:
            return h
    for h in items:
        if os.path.basename(h.name) == name:
            return h
    return None

# ---------------------------------------------------------------------------
# Project HDRI folder (the shot's own HDRIs)
# ---------------------------------------------------------------------------

# Folder next to the scene file (e.g. <shot>/3D/8_lighting/1_projects/hdri) that the Studio
# Toolbar creates with the task folders. When it exists, every toolbar offers it as a library
# and uses it by default, so the HDRIs a shot uses live with the shot.
PROJECT_HDRI_DIRNAME = "hdri"


def project_hdri_dir(scene_dir: Optional[str], create: bool = False) -> Optional[str]:
    """
    The project's hdri folder inside `scene_dir` (the folder of the open scene), matched
    without case; None when the scene is not saved or the folder is missing (unless
    `create`, which makes it).
    """
    if not scene_dir:
        return None
    scene_dir = os.path.normpath(str(scene_dir))
    if not os.path.isdir(scene_dir):
        return None
    try:
        for d in os.listdir(scene_dir):
            p = os.path.join(scene_dir, d)
            if d.lower() == PROJECT_HDRI_DIRNAME and os.path.isdir(p):
                return p
    except OSError:
        return None
    if create:
        p = os.path.join(scene_dir, PROJECT_HDRI_DIRNAME)
        os.makedirs(p, exist_ok=True)
        return p
    return None


def _same_file(a: str, b: str) -> bool:
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return False
    return sa.st_size == sb.st_size and abs(sa.st_mtime - sb.st_mtime) < 2.0


def copy_hdri(hdri: HDRI, dest_dir: str, overwrite: bool = False) -> Tuple[HDRI, bool]:
    """
    Copies an HDRI into another library (e.g. the shot's project hdri folder) with what goes
    with it, so it shows there at once with the same preview and defaults:
      next to the file: the hand-made thumbnail, the .json sidecar, the Arnold .tx
      generated:        the preview, the blurred background and saved defaults (cache)
    The file goes to the top of `dest_dir` (sub-folders of the source library are dropped).
    Returns (the HDRI in `dest_dir`, copied). copied is False when it was already there
    (same file, or the HDRI already belongs to `dest_dir`).
    Raises FileExistsError when a different file of that name is there and not `overwrite`.
    """
    import shutil
    dest_dir = os.path.normpath(dest_dir)
    src = os.path.normpath(hdri.path)
    if (_norm_path(src) + os.sep).startswith(_norm_path(dest_dir) + os.sep):
        return hdri_from_path(src, dest_dir), False
    dst = os.path.join(dest_dir, os.path.basename(src))
    if os.path.isfile(dst):
        if _same_file(src, dst):
            return hdri_from_path(dst, dest_dir), False
        if not overwrite:
            raise FileExistsError(dst)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src, dst)

    def put(a: str, b: str) -> None:
        if os.path.isfile(a) and _norm_path(a) != _norm_path(b):
            os.makedirs(os.path.dirname(b), exist_ok=True)
            shutil.copy2(a, b)

    sbase, dbase = os.path.splitext(src)[0], os.path.splitext(dst)[0]
    for ext in THUMB_EXTS + (".json", ".tx"):
        put(sbase + ext, dbase + ext)
    put(thumb_cache_path(src, hdri.root), thumb_cache_path(dst, dest_dir))
    put(bg_cache_path(src, hdri.root), bg_cache_path(dst, dest_dir))
    if not is_default_library(hdri.root):        # defaults saved for a non-studio folder
        put(sidecar_path(src, hdri.root), sidecar_path(dst, dest_dir))
    return hdri_from_path(dst, dest_dir), True


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

# Linear Rec.709 (D65) -> ACEScg / AP1 (D60), Bradford adaptation.
_REC709_TO_AP1 = (
    (0.6130974024, 0.3395231462, 0.0473794514),
    (0.0701937225, 0.9163538791, 0.0134523985),
    (0.0206155929, 0.1095697729, 0.8698146342),
)


def srgb_to_linear(v: float) -> float:
    """sRGB 0-1 value -> linear (piecewise EOTF)."""
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _mat_mul(m, v):
    return tuple(m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] for r in range(3))


def to_working_space(lin_rec709: Sequence[float], working_space: str) -> Vec3:
    ws = (working_space or "").lower().replace("-", "").replace("_", "")
    if ws in ("acescg", "ap1"):
        return _mat_mul(_REC709_TO_AP1, lin_rec709)
    return tuple(lin_rec709)  # lin_srgb / lin_rec709 / scene-linear Rec.709


def grey(value: float, working_space: str) -> Vec3:
    # A neutral grey is the same in Rec.709 and AP1 (same white after Bradford).
    return to_working_space((value, value, value), working_space)


def grey18(working_space: str) -> Vec3:
    return grey(0.18, working_space)


# CIELAB (D50, as X-Rite publishes chart references) -> linear Rec.709 / sRGB (D65),
# Bradford-adapted matrix (Lindbloom).
_XYZD50_TO_REC709 = (
    (3.1338561, -1.6168667, -0.4906146),
    (-0.9787684, 1.9161415, 0.0334540),
    (0.0719453, -0.2289914, 1.4052427),
)
_D50 = (0.96422, 1.0, 0.82521)


def lab_d50_to_linear_rec709(lab: Sequence[float]) -> Vec3:
    L, a, b = lab
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0

    def finv(t):
        t3 = t ** 3
        if t3 > eps:
            return t3
        return (116.0 * t - 16.0) / kappa

    xyz = (finv(fx) * _D50[0], (fy ** 3 if L > kappa * eps else L / kappa) * _D50[1],
           finv(fz) * _D50[2])
    return tuple(max(0.0, v) for v in _mat_mul(_XYZD50_TO_REC709, xyz))


def _patch_linear(p: Dict, source_space: str) -> Vec3:
    if "lab" in p or source_space == "lab_d50":
        return lab_d50_to_linear_rec709(p["lab"])
    if "linear" in p:
        return tuple(float(v) for v in p["linear"])
    return tuple(srgb_to_linear(c / 255.0) for c in p["srgb"])



def list_charts(folder: str = REFERENCE_DIR) -> Dict[str, str]:
    """{chart id: label} for every chart definition in reference/ (files with "patches")."""
    out = {}
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(".json") and not fn.startswith("_"):
                data = load_json(os.path.join(folder, fn), {}) or {}
                if data.get("patches"):
                    cid = os.path.splitext(fn)[0]
                    out[cid] = data.get("label") or cid
    return out


def load_chart(chart_id: str = DEFAULT_CHART, working_space: str = "ACEScg",
               folder: str = REFERENCE_DIR) -> Dict:
    """
    Chart definition with every patch converted to linear rgb in the working space.
    Patches can sit anywhere on the rows x cols grid ("row"/"col"; default = reading order)
    and can be given as 8-bit sRGB ("srgb"), CIELAB D50 ("lab") or linear Rec.709 ("linear").
    Returns {id, label, rows, cols, patches:[{id, name, row, col, rgb}]}.
    """
    path = os.path.join(folder, chart_id + ".json")
    data = load_json(path)
    if not data or not data.get("patches"):
        raise IOError("Chart definition not found or empty: %s" % path)
    rows, cols = int(data.get("rows", 4)), int(data.get("cols", 6))
    space = data.get("source_space", "srgb_8bit")
    patches = []
    for i, p in enumerate(data["patches"]):
        patches.append({
            "id": p.get("id", i + 1),
            "name": p.get("name", "patch_%02d" % (i + 1)),
            "row": int(p.get("row", i // cols)),
            "col": int(p.get("col", i % cols)),
            "rgb": to_working_space(_patch_linear(p, space), working_space),
        })
    return {"id": chart_id, "label": data.get("label") or chart_id,
            "rows": rows, "cols": cols, "patches": patches}


def colorchecker_patches(working_space: str = "ACEScg",
                         path: Optional[str] = None) -> List[Dict]:
    """Classic 24-patch chart (kept for compatibility; use load_chart())."""
    if path:
        folder, fn = os.path.split(path)
        return load_chart(os.path.splitext(fn)[0], working_space, folder)["patches"]
    return load_chart(DEFAULT_CHART, working_space)["patches"]


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

def _add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    l = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / l, a[1] / l, a[2] / l)


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

def bounding_sphere(bmin: Vec3, bmax: Vec3) -> Tuple[Vec3, float]:
    center = _mul(_add(bmin, bmax), 0.5)
    radius = math.sqrt(_dot(_sub(bmax, bmin), _sub(bmax, bmin))) * 0.5
    return center, max(radius, 1e-4)


def fovs(focal_mm: float, film_width_mm: float, aspect: float) -> Tuple[float, float]:
    """(horizontal FOV, vertical FOV) in radians. aspect = render width / height."""
    hfov = 2.0 * math.atan(film_width_mm * 0.5 / focal_mm)
    return hfov, vfov_from_hfov(hfov, aspect)


def vfov_from_hfov(hfov: float, aspect: float) -> float:
    return 2.0 * math.atan(math.tan(hfov * 0.5) / max(aspect, 1e-6))


def camera_distance(radius: float, hfov: float, vfov: float, margin: float) -> float:
    """Distance at which the bounding sphere fits entirely inside the narrower FOV."""
    half = min(hfov, vfov) * 0.5
    return radius * margin / math.sin(half)


@dataclass
class CameraFrame:
    position: Vec3
    x_axis: Vec3      # right
    y_axis: Vec3      # up
    z_axis: Vec3      # backwards (the camera looks down -Z)
    target: Vec3
    distance: float   # distance from the lens to the asset centre, along the view axis
    hfov: float
    vfov: float
    radius: float

    def to_world(self, cam_space: Vec3) -> Vec3:
        """Camera-space point -> world."""
        p = self.position
        p = _add(p, _mul(self.x_axis, cam_space[0]))
        p = _add(p, _mul(self.y_axis, cam_space[1]))
        p = _add(p, _mul(self.z_axis, cam_space[2]))
        return p


def camera_frame(bmin: Vec3, bmax: Vec3, preset: Dict, aspect: float,
                 up_axis: str = "Z") -> CameraFrame:
    """
    Front camera looking at the asset centre, at the exact distance to frame it.
    up_axis: "Z" (Max, Blender) -> camera looks from -Y.
             "Y" (Maya, Houdini) -> camera looks from +Z.
    """
    cam = preset["camera"]
    center, radius = bounding_sphere(bmin, bmax)
    hfov, vfov = fovs(cam["focal_mm"], cam["film_width_mm"], aspect)
    dist = camera_distance(radius, hfov, vfov, cam["framing_margin"])

    e = math.radians(cam.get("elevation_deg", 0.0))
    if up_axis.upper() == "Z":
        world_up = (0.0, 0.0, 1.0)
        offset = (0.0, -dist * math.cos(e), dist * math.sin(e))
    else:
        world_up = (0.0, 1.0, 0.0)
        offset = (0.0, dist * math.sin(e), dist * math.cos(e))

    pos = _add(center, offset)
    z_axis = _norm(_sub(pos, center))            # points backwards
    x_axis = _norm(_cross(world_up, z_axis))
    y_axis = _cross(z_axis, x_axis)
    return CameraFrame(pos, x_axis, y_axis, z_axis, center, dist, hfov, vfov, radius)


def camera_frame_from_existing(position: Vec3, x_axis: Vec3, y_axis: Vec3, z_axis: Vec3,
                               hfov: float, aspect: float,
                               bmin: Vec3, bmax: Vec3) -> CameraFrame:
    """
    Frame for a camera the artist already placed (approved framing).
    Axes come straight from the camera transform (scale is removed); hfov in radians.
    The reference depth follows the distance to the asset, so balls and chart keep the
    same on-screen size as with an automatic camera.
    """
    center, radius = bounding_sphere(bmin, bmax)
    x, y, z = _norm(x_axis), _norm(y_axis), _norm(z_axis)
    dist = _dot(_sub(center, position), _mul(z, -1.0))
    if dist < radius * 0.25:                     # asset behind / around the lens
        dist = max(radius * 2.0, 1e-3)
    return CameraFrame(tuple(position), x, y, z, center, dist,
                       hfov, vfov_from_hfov(hfov, aspect), radius)


# ---------------------------------------------------------------------------
# References (balls + chart) in camera space
# ---------------------------------------------------------------------------

BALL_KINDS = ("chrome", "grey18")


REF_ITEMS = ("chrome", "grey18", "colorchecker")
CORNERS = ("bottom_right", "bottom_left", "top_right", "top_left")


def reference_layout(frame: CameraFrame, preset: Dict, chart_aspect: float = 1.5) -> List[Dict]:
    """
    Places the references in a corner of the frame, in camera space.
    Sizes are relative to the frame height: they don't depend on the asset size.

    Returns [{kind, pos:(x,y,z), size:(w,h), radius?}], ordered from the corner inwards.
    """
    r = preset["refs"]
    depth = frame.distance * float(r["depth_factor"])
    half_h = depth * math.tan(frame.vfov * 0.5)
    half_w = depth * math.tan(frame.hfov * 0.5)
    frame_h = 2.0 * half_h

    diam = frame_h * float(r["ball_size_screen"])
    gap = diam * float(r["gap"])
    pad = frame_h * float(r["padding_screen"])
    chart_h = diam * float(r["chart_height_factor"])
    chart_w = chart_h * chart_aspect               # cols / rows of the chart

    corner = r.get("corner", "bottom_right").lower()
    hside = 1.0 if "right" in corner else -1.0
    vside = -1.0 if "bottom" in corner else 1.0

    items = []
    cursor = half_w - pad                          # distance from centre to the free edge
    for kind in r["items"]:
        if kind in BALL_KINDS:
            w = h = diam
        elif kind == "colorchecker":
            w, h = chart_w, chart_h
        else:
            continue
        cx = hside * (cursor - w * 0.5)
        cy = vside * (half_h - pad - h * 0.5)
        item = {"kind": kind, "pos": (cx, cy, -depth), "size": (w, h)}
        if kind in BALL_KINDS:
            item["radius"] = diam * 0.5
        items.append(item)
        cursor -= w + gap
    return items


def colorchecker_layout(chart_w: float, chart_h: float, rows: int = 4, cols: int = 6,
                        patch_fill: float = 0.82, board_margin: float = 0.06) -> Dict:
    """
    Chart inner layout, relative to its centre (XY plane, normal +Z).
    Returns {board:(w,h), patch:(s), cells:{(row,col):(x,y)}}.
    """
    inner_w = chart_w * (1.0 - board_margin)
    inner_h = chart_h * (1.0 - board_margin)
    pitch = min(inner_w / cols, inner_h / rows)
    size = pitch * patch_fill
    cells = {}
    for row in range(rows):
        for col in range(cols):
            x = (col - (cols - 1) * 0.5) * pitch
            y = ((rows - 1) * 0.5 - row) * pitch
            cells[(row, col)] = (x, y)
    return {"board": (chart_w, chart_h), "patch": size, "cells": cells}


# ---------------------------------------------------------------------------
# Ground (shadow catcher)
# ---------------------------------------------------------------------------

def ground_plane(bmin: Vec3, bmax: Vec3, preset: Dict, up_axis: str = "Z") -> Dict:
    """
    Square ground plane under the asset, centred on it.
    Returns {pos:(x,y,z), size:float}; the plane normal is the world up axis.
    """
    g = preset["ground"]
    center, radius = bounding_sphere(bmin, bmax)
    drop = float(g.get("offset", 0.0)) * radius
    if up_axis.upper() == "Z":
        pos = (center[0], center[1], bmin[2] - drop)
    else:
        pos = (center[0], bmin[1] - drop, center[2])
    return {"pos": pos, "size": radius * float(g.get("size_factor", 12.0))}


# ---------------------------------------------------------------------------
# Turntable
# ---------------------------------------------------------------------------

@dataclass
class TurntableTiming:
    start: int
    object_end: int       # frame where the object turn ends (= HDRI turn starts)
    hdri_end: int         # frame where the HDRI turn ends
    object_sign: float    # rotation sign of the rig (the rig orbits opposite to the object)

    @property
    def render_range(self) -> Tuple[int, int]:
        return self.start, self.hdri_end - 1


def turntable_timing(preset: Dict) -> TurntableTiming:
    """
    The asset is never touched: the rig (camera + refs + dome + ground) orbits around it.
    Orbiting camera and dome together = the object turning under fixed light.
    Then the rig holds still and only the dome spins.
    """
    t = preset["turntable"]
    s = int(t["start_frame"])
    oe = s + int(t["object_frames"])
    he = oe + int(t["hdri_frames"])
    sign = -1.0 if t.get("object_spin", "ccw").lower() == "ccw" else 1.0
    return TurntableTiming(s, oe, he, sign)
