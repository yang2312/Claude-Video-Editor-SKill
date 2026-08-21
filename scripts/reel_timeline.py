#!/usr/bin/env python3
"""
Framing and the timeline — where the picture comes from, and when.

Two ideas live here.

**Framing** decides which rectangle of the source ends up on screen. Cutting
16:9 footage to 9:16 throws away two thirds of the width, and a window that
moves across that slack is what lets one locked-off drone shot read as two
different angles. It is also the only camera move a still photograph has.

**Timeline** places finished shots on one clock and answers `frame(t)`.

The timeline is written by hand rather than assembled from moviepy's
CompositeVideoClip, and that is the point of the module. A composite treats
every moment as a stack of layers to be alpha-blended, so a reel where 95% of
frames have exactly one visible shot still pays for masks, RGBA conversion and
`alpha_composite` on all of them — measured at 27 ms a frame. Owning the
placement means the common case is `return shot.frame(t)` and costs nothing,
and the two rare cases pay only when they happen:

    crossfade  two shots overlap, so blend them — about 5% of frames
    flash      a white bloom over a junction — about 3% of frames

The old code wrapped the entire timeline in another clip to add the flash,
which made every frame in the reel pay a full float32 conversion for an
effect that touches thirteen of them.

Owning placement also fixes a quieter bug: moviepy's `subclipped` shares one
ffmpeg reader with its parent, so during a crossfade two shots reading from
different points of the same file made the decoder seek back and forth on
every frame. Here each shot opens its own reader, on first use, and closes it
the moment the timeline moves past — so at most two are ever open.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from reel_grade import Grade, Look

MOTIONS = ("none", "zoom-in", "zoom-out", "pan-left", "pan-right")

# How a clip joins the one before it.
#   crossfade — dissolve; reads as "time passing"
#   cut       — hard join; the only way to make a section feel fast
#   flash     — hard join under a white bloom; reads as "somewhere else now"
TRANSITIONS = ("crossfade", "cut", "flash")

# Rotated so adjacent shots never repeat a move — the thing that makes a
# Ken Burns reel read as mechanical rather than filmed.
_AUTO_CYCLE = ("zoom-in", "pan-right", "zoom-out", "pan-left")


def auto_motion(index: int) -> str:
    return _AUTO_CYCLE[index % len(_AUTO_CYCLE)]


def ease(p: float) -> float:
    """Smoothstep. Camera moves that start and stop abruptly read as cheap."""
    p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
    return p * p * (3.0 - 2.0 * p)


def zoom_range(zoom, motion: str) -> tuple:
    """
    Resolve a shot's zoom into (start, end) scale factors.

    `zoom` is either a number — the far end of a move that starts at 1.0 — or
    an explicit [start, end] pair. The pair matters when a shot needs to begin
    already tight: a wide establishing frame that has dead space at the edges
    should not spend its first second showing that dead space.
    """
    if isinstance(zoom, (list, tuple)):
        if len(zoom) != 2:
            raise ValueError(f"zoom pair must have exactly 2 values, got {list(zoom)}")
        a, b = (float(v) for v in zoom)
        if a < 1.0 or b < 1.0:
            raise ValueError(f"zoom values must be >= 1.0, got {list(zoom)}")
        return a, b

    z = float(zoom)
    if z < 1.0:
        raise ValueError(f"zoom must be >= 1.0, got {z}")
    if motion == "zoom-in":
        return 1.0, z
    if motion == "zoom-out":
        return z, 1.0
    return 1.0, 1.0


# How much sharpness a draft is allowed to trade for speed.
#
#   "quality" — LANCZOS everywhere. The default, and what a delivery should
#               use: it is the only setting that leaves the picture alone.
#   "fast"    — BICUBIC when enlarging, LANCZOS when shrinking. Enlarging has
#               no detail to alias, so the extra taps buy a different kind of
#               edge rather than more of it; measured 25 ms against 34 ms on a
#               9:16 crop out of 1080p, which is about 6% off a whole render.
#               Worth it for a draft you will look at once, not for a master.
#
# Shrinking stays on LANCZOS in both modes. There the wide support is not a
# preference — it is what stops a downscale from aliasing.
RESAMPLE_MODES = ("quality", "fast")


def _resampler(scale: float, mode: str = "quality"):
    """Pick a resampling filter for a given output/input scale."""
    if mode == "fast" and scale >= 1.0:
        return Image.BICUBIC
    return Image.LANCZOS


def _soften(img: Image.Image, amount: float) -> Image.Image:
    """
    Blend toward a blurred copy — the look of a lens that is not clinical.

    The blur runs at half resolution and is scaled back up. Gaussian blur
    costs time in proportion to pixel count, so a quarter of the pixels is a
    quarter of the work, and blurring something that is about to be blurred
    hides the interpolation completely.
    """
    if amount <= 0:
        return img
    w, h = img.size
    radius = max(1.0, w / 420.0)
    small = img.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(radius / 2.0))
    return Image.blend(img, small.resize((w, h), Image.BILINEAR), amount)


# ------------------------------------------------------------------ sources

class StillSource:
    """A photograph standing in for footage — an archival opener, a title plate."""

    def __init__(self, path: str):
        self._path = path
        self._image = Image.open(path).convert("RGB")
        self.size = self._image.size

    def frame(self, t: float) -> Image.Image:
        if self._image is None:
            self._image = Image.open(self._path).convert("RGB")
        return self._image

    def release(self) -> None:
        # Reopenable, like ClipSource. A release is a memory decision, not a
        # promise that the shot is finished — rendering a range twice, or
        # seeking backwards, has to keep working.
        self._image = None


class ClipSource:
    """
    One time range of a video file, with a reader of its own.

    The reader is opened on the first frame asked for, not at construction, so
    a nine-shot reel never has more than the two ffmpeg processes a crossfade
    actually needs.
    """

    def __init__(self, path: str, start: float, end: float, speed: float,
                 size: tuple, source_fps: float):
        self._path = path
        self._start = start
        self._end = end
        self._speed = speed
        self.size = size
        self._fps = source_fps
        self._clip = None

    @property
    def duration(self) -> float:
        return (self._end - self._start) / self._speed

    def frame(self, t: float) -> Image.Image:
        if self._clip is None:
            from moviepy import VideoFileClip
            self._clip = VideoFileClip(self._path)

        # Speed is applied by reading a different point rather than by
        # resampling: a 0.85x clip is the same frames, further apart.
        source_t = self._start + t * self._speed
        # Half a frame back from the end. Asking for the exact final
        # timestamp lands past the last decoded frame on some files.
        limit = self._end - 0.5 / max(self._fps, 1.0)
        return Image.fromarray(self._clip.get_frame(min(source_t, limit)))

    def release(self) -> None:
        if self._clip is not None:
            try:
                self._clip.close()
            except Exception:
                pass
            self._clip = None


# ------------------------------------------------------------------ framing

class CropFraming:
    """
    A window of the output's aspect ratio, moving across the source.

    The window is what carries the camera move. `zoom` scales it, `pan` slides
    it, `anchor` says where it sits when it is not sliding, and everything is
    eased so the move accelerates and settles rather than starting at full
    speed.
    """

    def __init__(self, source_size: tuple, out_w: int, out_h: int,
                 motion: str, zoom, pan: float, anchor: float, softness: float,
                 resample: str = "quality"):
        W, H = source_size
        aspect = out_w / out_h
        if W / H >= aspect:
            self._base = (H * aspect, float(H))
        else:
            self._base = (float(W), W / aspect)

        self._source_size = (W, H)
        self._out = (out_w, out_h)
        self._motion = motion
        self._anchor = anchor
        self._pan = pan
        self._softness = softness
        self._resample = resample
        self._zoom = zoom_range(zoom, motion)

        # A shot with no move has one window for its whole length, so resolve
        # it now and skip the arithmetic on every frame.
        self._fixed = self._window(0.0) if motion == "none" else None

    @property
    def static(self) -> bool:
        """True when every frame of this shot frames the source identically."""
        return self._fixed is not None

    def _window(self, progress: float) -> tuple:
        e = ease(progress)
        z_start, z_end = self._zoom
        z = z_start + (z_end - z_start) * e
        base_w, base_h = self._base
        W, H = self._source_size
        w, h = base_w / z, base_h / z

        slack = W - w
        if self._motion in ("pan-left", "pan-right") and slack > 1:
            span = slack * self._pan
            left = self._anchor * slack - span / 2.0
            if self._motion == "pan-left":
                left, span = left + span, -span
            left += span * e
        else:
            left = self._anchor * slack

        left = max(0.0, min(slack, left))
        top = max(0.0, min(H - h, (H - h) / 2.0))
        return left, top, w, h

    def apply(self, img: Image.Image, progress: float) -> np.ndarray:
        left, top, w, h = self._window(progress) if self._fixed is None else self._fixed
        out_w, out_h = self._out
        # PIL fuses crop and resize when given a box, so the pixels outside
        # the window are never touched.
        framed = img.resize(self._out,
                            resample=_resampler(out_w / w, self._resample),
                            box=(left, top, left + w, top + h))
        return np.asarray(_soften(framed, self._softness))


class FitFraming:
    """
    The whole source frame, letterboxed on a blurred bed of itself.

    For shots whose meaning spans the full width — a sign, a banner, a wide
    photograph — where a 9:16 crop would cut the words in half. The bed is
    built once from the middle of the shot, so it does not shimmer.

    Fit honours `motion` the same way cropping does. It did not always: an
    earlier version ignored zoom and pan here entirely, so a spec asking for a
    slow push on a letterboxed archival photo rendered a completely static
    shot and said nothing about it. A move that a spec asks for and does not
    get is worse than one that is refused, because nothing in the output tells
    you it is missing.

    The difference from cropping is only what the move is allowed to cost.
    Zooming past 1.0 grows the picture wider than the frame, so the sides go
    outside — which is the trade the shot is already making in reverse, and it
    stays centred on whatever `anchor` names.
    """

    def __init__(self, source_size: tuple, out_w: int, out_h: int,
                 softness: float, bed_frame: Image.Image,
                 resample: str = "quality", motion: str = "none",
                 zoom=1.0, pan: float = 0.30, anchor: float = 0.5):
        W, H = source_size
        self._out = (out_w, out_h)
        self._softness = softness
        self._resample = resample
        self._motion = motion
        self._anchor = anchor
        self._pan = pan
        self._zoom = zoom_range(zoom, motion)

        cover = max(out_w / W, out_h / H)
        bw, bh = max(1, int(W * cover)), max(1, int(H * cover))
        bed = bed_frame.resize((bw, bh), resample=_resampler(cover, resample)).crop((
            (bw - out_w) // 2, (bh - out_h) // 2,
            (bw - out_w) // 2 + out_w, (bh - out_h) // 2 + out_h,
        ))
        bed = bed.filter(ImageFilter.GaussianBlur(radius=max(8, out_w // 28)))
        self._bed = np.asarray(ImageEnhance.Brightness(bed).enhance(0.42))

        base_h = max(2, int(round(out_w * H / W)))
        self._base_size = (out_w, base_h - base_h % 2)
        self._source_aspect = W / H

    @property
    def static(self) -> bool:
        """True when every frame of this shot frames the source identically."""
        moving_pan = self._motion in ("pan-left", "pan-right") and self._pan != 0
        return self._zoom[0] == self._zoom[1] and not moving_pan

    def apply(self, img: Image.Image, progress: float) -> np.ndarray:
        out_w, out_h = self._out
        e = ease(progress)
        z_start, z_end = self._zoom
        z = z_start + (z_end - z_start) * e

        base_w, base_h = self._base_size
        fg_w = max(2, int(round(base_w * z)))
        fg_h = max(2, int(round(base_h * z)))

        fg = _soften(img.resize((fg_w, fg_h),
                                resample=_resampler(fg_w / img.size[0], self._resample)),
                     self._softness)

        # Whatever grew past the frame goes outside it, anchored where asked.
        slack_x = max(0, fg_w - out_w)
        if self._motion in ("pan-left", "pan-right") and slack_x > 1:
            span = slack_x * self._pan
            left = self._anchor * slack_x - span / 2.0
            if self._motion == "pan-left":
                left, span = left + span, -span
            left += span * e
        else:
            left = self._anchor * slack_x
        left = int(round(max(0, min(slack_x, left))))
        top = max(0, fg_h - out_h) // 2

        visible_w = min(fg_w, out_w)
        visible_h = min(fg_h, out_h)
        if (left, top, visible_w, visible_h) != (0, 0, fg_w, fg_h):
            fg = fg.crop((left, top, left + visible_w, top + visible_h))

        canvas = self._bed.copy()
        x = (out_w - visible_w) // 2
        y = (out_h - visible_h) // 2
        canvas[y:y + visible_h, x:x + visible_w] = np.asarray(fg)
        return canvas


# --------------------------------------------------------------------- shot

class Shot:
    """
    One finished piece of picture: source, framing and grade, resolved.

    `frame(t, index)` returns a uint8 (H, W, 3) array at the output geometry.
    `index` selects the grain phase; it is the frame's own number, so grain
    moves with the picture rather than crawling at a different rate.
    """

    __slots__ = ("source", "framing", "grade", "duration", "label",
                 "grade_name", "_freezable", "_frozen")

    def __init__(self, source, framing, grade: Optional[Grade],
                 duration: float, label: str = "", grade_name: str = "none"):
        self.source = source
        self.framing = framing
        self.grade = grade
        self.duration = duration
        self.label = label
        self.grade_name = grade_name

        # A still photograph held without a camera move produces the same
        # pixels for its whole length. Resizing and blurring it eighty-four
        # times to get eighty-four identical arrays is the most avoidable work
        # in the renderer — so frame it once. Grain still moves, because that
        # happens after this point.
        self._freezable = (not isinstance(source, ClipSource)
                           and getattr(framing, "static", False))
        self._frozen = None

    def frame(self, t: float, index: int) -> np.ndarray:
        arr = self._frozen
        if arr is None:
            progress = 0.0 if self.duration <= 0 else t / self.duration
            arr = self.framing.apply(self.source.frame(t), progress)
            if self._freezable:
                self._frozen = arr
        if self.grade is None:
            return arr
        return self.grade.apply(arr, index)

    def release(self) -> None:
        self._frozen = None
        self.source.release()


def build_framing(source, spec: dict, out_w: int, out_h: int, look: Look,
                  resample: str = "quality"):
    """Choose and configure the framing one clip asks for."""
    if spec["fit"]:
        # The bed comes from the middle of the shot rather than the first
        # frame — an opening frame is often a fade-in, and a bed built from
        # black stays black for the whole shot.
        return FitFraming(source.size, out_w, out_h, look.softness,
                          source.frame(getattr(source, "duration", 0.0) / 2.0),
                          resample, spec["motion"], spec["zoom"], spec["pan"],
                          spec["anchor"])
    return CropFraming(source.size, out_w, out_h, spec["motion"], spec["zoom"],
                       spec["pan"], spec["anchor"], look.softness, resample)


# ----------------------------------------------------------------- timeline

class Timeline:
    """
    Every shot on one clock, with the transitions between them.

    Placement follows one rule: a crossfading shot starts `crossfade` seconds
    before the previous one ends, and everything else starts exactly when the
    previous one ends. That single rule is what makes the back half of a reel
    feel faster than the front — hard cuts remove the dissolve's built-in
    pause, independently of how long the clips are.
    """

    def __init__(self, shots: list, transitions: list, crossfade: float,
                 fps: int, flash_strength: float = 0.93):
        self.fps = fps
        self._strength = flash_strength
        # Wide enough to bloom and decay rather than strobe, which is both
        # uglier and harder to watch; narrow enough to stay a punctuation mark.
        self._half = max(0.10, min(0.22, (crossfade or 0.3) * 0.6))

        self.placements = []
        self.flashes = []
        cursor = 0.0
        for i, shot in enumerate(shots):
            kind = "cut" if i == 0 else transitions[i]
            if kind == "crossfade" and crossfade > 0:
                start = max(0.0, cursor - crossfade)
            else:
                start = cursor
                if kind == "flash":
                    self.flashes.append(start)
            self.placements.append((start, start + shot.duration, shot, start))
            cursor = start + shot.duration

        self.duration = cursor
        self._crossfade = crossfade
        self._cursor = 0

    # -- playback --------------------------------------------------------

    def frame(self, t: float) -> np.ndarray:
        index = int(round(t * self.fps))
        active = self._active(t)

        if not active:
            # Only reachable at the very last timestamp, where floating point
            # can land a hair past the final shot's end.
            active = [self.placements[-1]]

        start, _, shot, origin = active[-1]
        out = shot.frame(min(t - origin, shot.duration), index)

        if len(active) > 1:
            under = active[-2]
            weight = 1.0 if self._crossfade <= 0 else min(
                1.0, max(0.0, (t - start) / self._crossfade))
            out = _blend(under[2].frame(min(t - under[3], under[2].duration), index),
                         out, weight)

        amount = self._flash_amount(t)
        if amount > 0:
            out = _bloom(out, amount)
        return out

    def _active(self, t: float) -> list:
        """
        The shots visible at t, earliest first.

        Frames arrive in order during a render, so the scan starts where the
        last one finished and shots are released as the clock passes them.
        Seeking backwards is still correct — it just rewinds the cursor.
        """
        if t < self.placements[self._cursor][0]:
            self._cursor = 0

        active = []
        i = self._cursor
        while i < len(self.placements):
            start, end, shot, _ = self.placements[i]
            if start > t:
                break
            if t < end:
                if not active:
                    self._cursor = i
                active.append(self.placements[i])
            elif i == self._cursor:
                # Fully behind us and nothing overlaps it any more.
                shot.release()
                self._cursor = i + 1
            i += 1
        return active

    def _flash_amount(self, t: float) -> float:
        best = 0.0
        for moment in self.flashes:
            distance = abs(t - moment)
            if distance < self._half:
                best = max(best, (1.0 - distance / self._half) ** 1.6 * self._strength)
        return best

    def release(self) -> None:
        for _, _, shot, _ in self.placements:
            shot.release()


def _blend(under: np.ndarray, over: np.ndarray, weight: float) -> np.ndarray:
    """Dissolve `over` onto `under`. Only runs while two shots overlap."""
    if weight >= 1.0:
        return over
    if weight <= 0.0:
        return under
    out = under.astype(np.float32)
    out *= (1.0 - weight)
    out += over.astype(np.float32) * weight
    return out.astype(np.uint8)


def _bloom(frame: np.ndarray, amount: float) -> np.ndarray:
    """
    Push a frame toward white.

    out = f + (255 - f) * a, rearranged so it is one scalar multiply and one
    scalar add rather than a subtract, a multiply and an add over a 25 MB
    buffer.
    """
    out = frame.astype(np.float32)
    out *= (1.0 - amount)
    out += 255.0 * amount
    return out.astype(np.uint8)
