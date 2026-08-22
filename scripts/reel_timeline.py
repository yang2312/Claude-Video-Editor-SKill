#!/usr/bin/env python3
"""
The camera move, the framing, and the timeline.

Three ideas live here.

**CameraMove** is where a shot's motion is decided, once. It exists because
its absence caused four separate bugs: `motion`, `zoom`, `pan` and `anchor`
used to travel as four loose numbers that two framing classes each interpreted
for themselves, in different parameter orders, disagreeing about what counts
as movement. A pan in fit mode rendered a frozen shot. The same spec animated
under one framing and froze under the other. Nothing reported any of it.

So the eased-zoom and eased-pan arithmetic is written once, here, and both
framings call it. They differ only in where their slack comes from.

**Framing** decides which rectangle of the source ends up on screen. Cutting
16:9 footage to 9:16 throws away two thirds of the width, and a window that
moves across that slack is what lets one locked-off drone shot read as two
different angles. It is also the only camera move a photograph has.

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

Owning placement means owning its limits too. `plan()` is the single
implementation of the placement rule *and* of what the rule cannot express:
at most two shots may overlap, because `frame()` blends two, and no shot may
be swallowed whole by its neighbour's dissolve. Callers validate by planning,
so the constraint can never drift from the code that depends on it.

Each shot opens its own ffmpeg reader, on first use, and closes it when the
clock moves past — moviepy's `subclipped` shares one reader with its parent,
which made two shots crossfading from different points of the same file seek
back and forth on every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from reel_grade import Grade

MOTIONS = ("none", "zoom-in", "zoom-out", "pan-left", "pan-right")

# How a clip joins the one before it.
#   crossfade — dissolve; reads as "time passing"
#   cut       — hard join; the only way to make a section feel fast
#   flash     — hard join under a white bloom; reads as "somewhere else now"
TRANSITIONS = ("crossfade", "cut", "flash")

# Rotated so adjacent shots never repeat a move — the thing that makes a
# Ken Burns reel read as mechanical rather than filmed.
_AUTO_CYCLE = ("zoom-in", "pan-right", "zoom-out", "pan-left")

# What a zoom-in or zoom-out travels when the spec does not say.
DEFAULT_ZOOM = 1.10

# How much sharpness a draft is allowed to trade for speed.
#
#   "quality" — LANCZOS everywhere. The default, and what a delivery should
#               use: it is the only setting that leaves the picture alone.
#   "fast"    — BICUBIC when enlarging, LANCZOS when shrinking. Enlarging has
#               no detail to alias, so the extra taps buy a different kind of
#               edge rather than more of it; measured 25 ms against 34 ms on a
#               9:16 crop out of 1080p, about 6% off a whole render. Worth it
#               for a draft you will look at once, not for a master.
#
# Shrinking stays on LANCZOS in both modes. There the wide support is not a
# preference — it is what stops a downscale from aliasing.
RESAMPLE_MODES = ("quality", "fast")


def auto_motion(index: int) -> str:
    """
    The move for clip `index` when the spec says "auto".

    Rotating on the clip's own index only guarantees no repeat when *every*
    clip is auto; mixing auto with explicit moves can still put two of a kind
    together. The spec layer warns about that rather than this function
    guessing what the neighbours were.
    """
    return _AUTO_CYCLE[index % len(_AUTO_CYCLE)]


def ease(p: float) -> float:
    """Smoothstep. Camera moves that start and stop abruptly read as cheap."""
    p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
    return p * p * (3.0 - 2.0 * p)


# -------------------------------------------------------------- camera move

@dataclass(frozen=True)
class CameraMove:
    """
    One shot's motion, fully resolved.

    `zoom` is always a (start, end) pair by the time it lives here — the
    scalar form in a spec is shorthand, and resolving it at the edge is what
    stops the rest of the code from having to know which motions a scalar
    applies to. It applies to all of them:

        zoom-in     1.10  ->  (1.00, 1.10)   travel out to in
        zoom-out    1.10  ->  (1.10, 1.00)   travel in to out
        pan-*       1.10  ->  (1.10, 1.10)   held tight while sliding
        none        1.10  ->  (1.10, 1.10)   held tight, no move

    Those last two rows are the fix for a silent drop: a scalar zoom under a
    pan used to resolve to (1.0, 1.0) and vanish. Omitting `zoom` entirely
    still means "no zoom" for pans, so specs written before this keep their
    framing exactly.
    """

    motion: str = "none"
    zoom: tuple = (1.0, 1.0)
    pan: float = 0.30
    anchor: tuple = (0.5, 0.5)

    # -- construction ----------------------------------------------------

    @classmethod
    def resolve(cls, motion="none", zoom=None, pan=0.30, anchor=0.5,
                where: str = "clip") -> "CameraMove":
        """
        Build a move from spec values, rejecting anything unrenderable.

        `where` names the offending clip in every error, because a spec with
        nine clips and one bad number is otherwise a guessing game.
        """
        if motion not in MOTIONS:
            raise ValueError(
                f"{where} has unknown motion {motion!r}; expected one of "
                f"{', '.join(MOTIONS)} or 'auto'")

        pan = float(pan)
        if not 0.0 <= pan <= 1.0:
            raise ValueError(f"{where} pan must be between 0 and 1, got {pan}")

        return cls(motion=motion, zoom=cls._resolve_zoom(zoom, motion, where),
                   pan=pan, anchor=cls._resolve_anchor(anchor, where))

    @staticmethod
    def _resolve_anchor(anchor, where: str) -> tuple:
        """A number places the window horizontally; a pair places both axes."""
        if isinstance(anchor, (list, tuple)):
            if len(anchor) != 2:
                raise ValueError(
                    f"{where} anchor pair must have exactly 2 values, "
                    f"got {list(anchor)}")
            pair = tuple(float(v) for v in anchor)
        else:
            pair = (float(anchor), 0.5)

        for value, axis in zip(pair, ("horizontal", "vertical")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{where} {axis} anchor must be between 0 and 1, got {value}")
        return pair

    @staticmethod
    def _resolve_zoom(zoom, motion: str, where: str) -> tuple:
        if zoom is None:
            # No zoom asked for. A zoom move still needs somewhere to travel.
            if motion == "zoom-in":
                return 1.0, DEFAULT_ZOOM
            if motion == "zoom-out":
                return DEFAULT_ZOOM, 1.0
            return 1.0, 1.0

        if isinstance(zoom, (list, tuple)):
            if len(zoom) != 2:
                raise ValueError(
                    f"{where} zoom pair must have exactly 2 values, got {list(zoom)}")
            start, end = (float(v) for v in zoom)
        else:
            z = float(zoom)
            if motion == "zoom-in":
                start, end = 1.0, z
            elif motion == "zoom-out":
                start, end = z, 1.0
            else:
                start = end = z

        for value in (start, end):
            if value < 1.0:
                raise ValueError(f"{where} zoom values must be >= 1.0, got {value}")
        return start, end

    # -- what it does ----------------------------------------------------

    @property
    def zooms(self) -> bool:
        return self.zoom[0] != self.zoom[1]

    @property
    def pans(self) -> bool:
        return self.motion in ("pan-left", "pan-right") and self.pan != 0.0

    @property
    def widest(self) -> float:
        """The largest scale the move reaches — where a pan's slack comes from."""
        return max(self.zoom)

    def is_static(self, has_slack: bool) -> bool:
        """
        True when every frame frames the source identically.

        A framing supplies `has_slack` because slack is the one thing the two
        framings genuinely disagree about: cropping gets it free from the
        aspect difference, fit only gets it by zooming past the frame edge.
        Everything else about "does this move" is decided here, so the two can
        no longer answer it differently for the same spec.
        """
        return not self.zooms and not (self.pans and has_slack)

    # -- the arithmetic, written once ------------------------------------

    def scale_at(self, progress: float) -> float:
        """The eased zoom at this point through the shot."""
        start, end = self.zoom
        return start + (end - start) * ease(progress)

    def offset_at(self, progress: float, slack: float) -> float:
        """
        Where the window sits along `slack`, eased.

        A pan spends `self.pan` of the available slack, centred on the anchor,
        so moving the anchor moves the whole travel rather than clipping it.
        """
        if slack <= 1.0:
            return max(0.0, self.anchor[0] * slack)

        if self.pans:
            span = slack * self.pan
            left = self.anchor[0] * slack - span / 2.0
            if self.motion == "pan-left":
                left, span = left + span, -span
            left += span * ease(progress)
        else:
            left = self.anchor[0] * slack

        return max(0.0, min(slack, left))


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

    Applied during framing, which is *before* the grade — so grain lands on
    top of the softness rather than under it, which is the order film works
    in. `Look.softness` is the parameter; this is where it is spent.
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

    Slack is free here: a 16:9 source cut to 9:16 leaves two thirds of the
    width unused, so a pan has somewhere to go even at zoom 1.0.
    """

    def __init__(self, source_size: tuple, out_w: int, out_h: int,
                 move: CameraMove, softness: float, resample: str = "quality"):
        W, H = source_size
        aspect = out_w / out_h
        if W / H >= aspect:
            self._base = (H * aspect, float(H))
        else:
            self._base = (float(W), W / aspect)

        self._source_size = (W, H)
        self._out = (out_w, out_h)
        self._move = move
        self._softness = softness
        self._resample = resample

        # A shot that never moves has one window for its whole length, so
        # resolve it now and skip the arithmetic on every frame.
        self._fixed = self._window(0.0) if self.static else None

    @property
    def slack(self) -> float:
        """Horizontal room at the widest point of the move."""
        base_w, _ = self._base
        return self._source_size[0] - base_w / self._move.widest

    @property
    def static(self) -> bool:
        return self._move.is_static(self.slack > 1.0)

    def _window(self, progress: float) -> tuple:
        z = self._move.scale_at(progress)
        base_w, base_h = self._base
        W, H = self._source_size
        w, h = base_w / z, base_h / z
        left = self._move.offset_at(progress, W - w)
        top = max(0.0, min(H - h, self._move.anchor[1] * (H - h)))
        return left, top, w, h

    def apply(self, img: Image.Image, progress: float) -> np.ndarray:
        window = self._fixed if self._fixed is not None else self._window(progress)
        left, top, w, h = window
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

    Unlike cropping, fit has no free slack: the picture is sized to the output
    width, so at zoom 1.0 there is nothing for a pan to travel across. A pan
    here needs a zoom above 1.0 to make room, and the spec layer refuses the
    combination rather than rendering a frozen shot and calling it a success.
    """

    def __init__(self, source_size: tuple, out_w: int, out_h: int,
                 softness: float, bed_frame: Image.Image,
                 move: CameraMove, resample: str = "quality"):
        W, H = source_size
        self._out = (out_w, out_h)
        self._softness = softness
        self._resample = resample
        self._move = move

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

    @property
    def slack(self) -> float:
        """Horizontal room, which only a zoom past 1.0 can create."""
        return max(0.0, self._base_size[0] * self._move.widest - self._out[0])

    @property
    def static(self) -> bool:
        return self._move.is_static(self.slack > 1.0)

    def apply(self, img: Image.Image, progress: float) -> np.ndarray:
        out_w, out_h = self._out
        z = self._move.scale_at(progress)

        base_w, base_h = self._base_size
        fg_w = max(2, int(round(base_w * z)))
        fg_h = max(2, int(round(base_h * z)))

        fg = _soften(img.resize((fg_w, fg_h),
                                resample=_resampler(fg_w / img.size[0], self._resample)),
                     self._softness)

        # Whatever grew past the frame goes outside it, anchored where asked.
        left = int(round(self._move.offset_at(progress, max(0, fg_w - out_w))))
        top = int(round(self._move.anchor[1] * max(0, fg_h - out_h)))

        visible_w = min(fg_w, out_w)
        visible_h = min(fg_h, out_h)
        if (left, top, visible_w, visible_h) != (0, 0, fg_w, fg_h):
            fg = fg.crop((left, top, left + visible_w, top + visible_h))

        canvas = self._bed.copy()
        x = (out_w - visible_w) // 2
        y = (out_h - visible_h) // 2
        canvas[y:y + visible_h, x:x + visible_w] = np.asarray(fg)
        return canvas


def build_framing(source, fit: bool, move: CameraMove, out_w: int, out_h: int,
                  softness: float, resample: str = "quality"):
    """
    Choose and configure the framing one clip asks for.

    The one decision this owns is the bed source: it comes from the middle of
    the shot rather than the first frame, because an opening frame is often a
    fade-in and a bed built from black stays black for the whole shot.
    """
    if fit:
        middle = source.frame(getattr(source, "duration", 0.0) / 2.0)
        return FitFraming(source.size, out_w, out_h, softness, middle, move, resample)
    return CropFraming(source.size, out_w, out_h, move, softness, resample)


# --------------------------------------------------------------------- shot

class Shot:
    """
    One finished piece of picture: source, framing and grade, resolved.

    `frame(t, index)` returns a uint8 (H, W, 3) array at the output geometry.
    `index` is the frame's own number on the timeline; it selects the grain
    phase, so grain moves with the picture rather than crawling at its own
    rate.

    `frames_built` counts how many times the framing stage actually ran, so
    the freeze optimisation below can be observed through the interface rather
    than by reaching into a private attribute for it.
    """

    __slots__ = ("source", "framing", "grade", "duration", "label",
                 "grade_name", "freezable", "frames_built", "_frozen")

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
        self.freezable = (not isinstance(source, ClipSource)
                          and getattr(framing, "static", False))
        self.frames_built = 0
        self._frozen = None

    def frame(self, t: float, index: int) -> np.ndarray:
        arr = self._frozen
        if arr is None:
            progress = 0.0 if self.duration <= 0 else t / self.duration
            arr = self.framing.apply(self.source.frame(t), progress)
            self.frames_built += 1
            if self.freezable:
                self._frozen = arr
        if self.grade is None:
            return arr
        return self.grade.apply(arr, index)

    def release(self) -> None:
        self._frozen = None
        self.source.release()


# ----------------------------------------------------------------- timeline

@dataclass(frozen=True)
class Placement:
    """Where one shot sits on the reel's clock."""

    start: float
    end: float
    index: int

    @property
    def length(self) -> float:
        return self.end - self.start


def plan(durations: list, transitions: list, crossfade: float) -> list:
    """
    Lay durations out on one clock, or say why they cannot be.

    One rule: a crossfading shot starts `crossfade` seconds before the
    previous one ends; everything else starts exactly when the previous one
    ends. That rule is what makes the back half of a reel feel faster than the
    front — hard cuts remove the dissolve's built-in pause, independently of
    how long the clips are.

    This function is also the only statement of what the rule cannot express,
    and it lives here rather than in the validator because `Timeline.frame` is
    where the limits come from:

      * a dissolve eats into *both* neighbours, so a clip shorter than the
        overlap is swallowed whole — and the previous validator measured only
        the incoming side, which let a 0.2 s clip disappear completely while
        the render reported success;

      * `frame()` blends exactly two layers, so three shots overlapping is not
        a tighter edit, it is a frame the renderer cannot draw. The previous
        validator permitted it and the oldest shot was dropped in silence.

    Callers validate a reel by planning it. A constraint stated anywhere else
    drifts away from the code that depends on it.
    """
    if len(transitions) != len(durations):
        raise ValueError(
            f"{len(durations)} clips but {len(transitions)} transitions")

    placements, cursor = [], 0.0
    for i, duration in enumerate(durations):
        if duration <= 0:
            raise ValueError(f"clips[{i}] has no length")

        # The first shot has nothing to dissolve from, so it always cuts in.
        kind = "cut" if i == 0 else transitions[i]
        if kind == "crossfade" and crossfade > 0:
            start = cursor - crossfade
            if start <= placements[-1].start:
                raise ValueError(
                    f"clips[{i - 1}] is {placements[-1].length:.2f}s but the "
                    f"crossfade into clips[{i}] is {crossfade}s — it would be "
                    f"swallowed whole and never appear. Shorten the crossfade, "
                    f"lengthen clips[{i - 1}], or make clips[{i}] a cut.")
            start = max(0.0, start)
        else:
            start = cursor

        placements.append(Placement(start, start + duration, i))
        cursor = start + duration

    for a, b, c in zip(placements, placements[1:], placements[2:]):
        if c.start < a.end:
            raise ValueError(
                f"clips[{a.index}], clips[{b.index}] and clips[{c.index}] would all "
                f"be on screen at {c.start:.2f}s. A dissolve blends two shots, not "
                f"three — the crossfade ({crossfade}s) must be shorter than half of "
                f"clips[{b.index}] ({b.length:.2f}s).")

    return placements


class Timeline:
    """
    Every shot on one clock, with the transitions between them.

    Placement and its limits come from `plan()`; this class adds playback.
    Constructing a Timeline therefore validates it — an unrenderable reel
    raises here rather than producing a wrong one.
    """

    def __init__(self, shots: list, transitions: list, crossfade: float,
                 fps: int, flash_strength: float = 0.93):
        # The frame rate is here for exactly one reason: `frame(t)` has to
        # name the frame's own number, which is what the grain phase advances
        # on. All the placement arithmetic below is in seconds.
        self.fps = fps
        self._strength = flash_strength
        # Wide enough to bloom and decay rather than strobe, which is both
        # uglier and harder to watch; narrow enough to stay a punctuation mark.
        self._half = max(0.10, min(0.22, (crossfade if crossfade > 0 else 0.3) * 0.6))

        self.shots = shots
        self.placements = plan([s.duration for s in shots], transitions, crossfade)
        # Clip 0 always cuts in, so a flash asked for there has nothing to
        # flash between. `plan` drops it; this drops it from the bloom list
        # for the same reason, and the spec layer is what warns about it.
        self.flashes = [p.start for p, kind in zip(self.placements, transitions)
                        if p.index > 0 and kind == "flash"]

        self.duration = self.placements[-1].end if self.placements else 0.0
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

        over = active[-1]
        out = self._draw(over, t, index)

        if len(active) > 1:
            under = active[-2]
            weight = 1.0 if self._crossfade <= 0 else min(
                1.0, max(0.0, (t - over.start) / self._crossfade))
            out = _blend(self._draw(under, t, index), out, weight)

        amount = self._flash_amount(t)
        if amount > 0:
            out = _bloom(out, amount)
        return out

    def _draw(self, placement: Placement, t: float, index: int) -> np.ndarray:
        shot = self.shots[placement.index]
        return shot.frame(min(t - placement.start, shot.duration), index)

    def _active(self, t: float) -> list:
        """
        The shots visible at t, earliest first.

        Frames arrive in order during a render, so the scan starts where the
        last one finished and shots are released as the clock passes them.
        Seeking backwards is still correct — it just rewinds the cursor.
        """
        if t < self.placements[self._cursor].start:
            self._cursor = 0

        active = []
        i = self._cursor
        while i < len(self.placements):
            placement = self.placements[i]
            if placement.start > t:
                break
            if t < placement.end:
                if not active:
                    self._cursor = i
                active.append(placement)
            elif i == self._cursor:
                # Fully behind us and nothing overlaps it any more.
                self.shots[placement.index].release()
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
        for shot in self.shots:
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
