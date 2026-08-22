#!/usr/bin/env python3
"""
The look engine — colour, texture, and the maths that makes them cheap.

A grade is eight knobs. Applied one at a time, as written, they cost eight
full float32 passes over every frame; on a 1080x1920 reel that measured
294 ms per frame, six times the cost of encoding it.

The saving is not a trick. Four of the eight knobs are *linear* maps:

    saturation   a channel mix     out = s*c + (1-s)*(c . w)
    temperature  a channel scale   out = k * c
    contrast     an affine map     out = C*c + (0.5 - 0.5C)
    lift         an affine map     out = (1-L)*c + L

Composing four linear maps gives one linear map, so all four collapse into a
single 3x3 matrix and a single offset — one pass instead of four, computed
once per look instead of once per frame. Gamma is pointwise and monotone, so
it collapses into a 256-entry lookup table. Only vignette and grain are
genuinely per-pixel, and their masks are built once and reused.

What survives per frame:

    one matrix multiply, one table lookup, one multiply, one add.

Everything else in this module exists to make sure that stays true: the
intermediate buffers are pooled and reused, looks are interned so five clips
sharing a preset share one compiled object and one vignette mask, and neutral
parameters drop out of the chain entirely rather than multiplying by one.

The pooling covers intermediates, not the result. A table lookup cannot write
into a pre-allocated buffer, and the caller may hold a frame across a
crossfade long after the next one is computed, so the frame handed back is
always freshly allocated.

`softness` lives here as a parameter but is applied during framing, where the
image is still a PIL image and a blur is cheap. It is part of the look, so it
is described here; see reel_timeline.Framing for where it lands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

# Rec. 709 luminance weights — the same ones broadcast uses, so desaturating
# leaves skin and foliage where the eye expects them.
LUM_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# How many distinct grain fields the cycle walks through. Real grain moves;
# it does not need to be unique for every frame of a fifteen-second reel, and
# eight is past the point where a viewer can see the loop.
GRAIN_PHASES = 8


# --------------------------------------------------------------- parameters

@dataclass(frozen=True)
class Look:
    """
    One grade, as numbers.

    saturation   1.0 untouched, 0.0 greyscale, >1 richer
    temperature  -1..1, negative cools toward blue, positive warms toward amber
    contrast     1.0 untouched
    lift         raises the black point — the milky shadows of aged film
    gamma        <1 brightens midtones, >1 deepens them
    black/white  input levels. Everything below `black` is crushed to zero and
                 everything above `white` is blown out. Pulling them together
                 is the "extract" look: high contrast, detail only where the
                 light was. Free — it folds into the same table as gamma.
    glow         0..1 bloom spilling out of the highlights
    glow_threshold  where the bloom starts, 0..1 of full brightness
    rgb_split    chromatic aberration, in pixels of red/blue separation
    vignette     0..1 strength of the corner falloff
    grain        0..1 amount of luminance noise
    softness     0..1 blend toward a blurred copy; old lenses were not sharp
    """

    saturation: float = 1.0
    temperature: float = 0.0
    contrast: float = 1.0
    lift: float = 0.0
    gamma: float = 1.0
    black: float = 0.0
    white: float = 1.0
    glow: float = 0.0
    glow_threshold: float = 0.62
    rgb_split: float = 0.0
    vignette: float = 0.0
    grain: float = 0.0
    softness: float = 0.0

    @property
    def touches_pixels(self) -> bool:
        """True when anything other than softness needs doing per frame."""
        return replace(self, softness=0.0) != Look()


# Each preset is a full parameter set, not a diff, so reading one tells you
# exactly what it does without cross-referencing a base.
PRESETS: dict[str, Look] = {
    "none": Look(),
    "vintage": Look(saturation=0.32, temperature=0.34, contrast=0.88,
                    lift=0.10, gamma=0.94, vignette=0.55, grain=0.055,
                    softness=0.28),
    "faded": Look(saturation=0.55, temperature=0.18, contrast=0.92,
                  lift=0.07, vignette=0.35, grain=0.03, softness=0.12),
    "warm": Look(saturation=1.08, temperature=0.12, contrast=1.06,
                 vignette=0.18),
    "vivid": Look(saturation=1.24, temperature=0.04, contrast=1.14,
                  gamma=1.03, vignette=0.22),

    # The two below come from the short-form edit vocabulary rather than from
    # film stock. They are heavy on purpose: both are built to sit under
    # three-frame cuts where a subtle grade would not register at all.

    # Crushed to near-monochrome with the highlights blooming — the look that
    # comes from stacking an extract, heavy noise and a glow.
    "opium": Look(saturation=0.06, contrast=1.12, black=0.30, white=0.74,
                  glow=0.55, glow_threshold=0.52, grain=0.10, vignette=0.42),

    # Noisy, split and blown — the companion to a stuttered frame rate.
    "dirty": Look(saturation=0.92, contrast=1.18, black=0.06, white=0.94,
                  glow=0.28, rgb_split=2.5, grain=0.14, vignette=0.30),
}


def resolve_look(value) -> tuple[Look, str]:
    """
    Turn a preset name, an override object, or nothing into a Look.

    Returns the Look and a short name for the render report, so callers never
    have to reconstruct "which preset was that" from the numbers.
    """
    if not value:
        return Look(), "none"

    if isinstance(value, str):
        if value not in PRESETS:
            raise ValueError(
                f"unknown grade preset {value!r}; expected one of "
                + ", ".join(sorted(PRESETS))
            )
        return PRESETS[value], value

    if isinstance(value, dict):
        fields = Look.__dataclass_fields__
        preset = value.get("preset")
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(
                    f"unknown grade preset {preset!r}; expected one of "
                    + ", ".join(sorted(PRESETS))
                )
            base = PRESETS[preset]
        else:
            base = Look()

        overrides = {}
        for k, v in value.items():
            if k == "preset":
                continue
            if k not in fields:
                raise ValueError(
                    f"unknown grade key {k!r}; expected one of "
                    + ", ".join(sorted(fields))
                )
            overrides[k] = float(v)

        name = preset if preset and not overrides else (
            f"{preset}+custom" if preset else "custom")
        return replace(base, **overrides), name

    raise ValueError(
        f"grade must be a preset name or an object, got {type(value).__name__}")


# ------------------------------------------------------------------ scratch

# One render walks one frame at a time, so every Grade can share the same
# working memory. Without this, nine shots at 1080x1920 would each hold 50 MB
# of buffers and every frame would hand the allocator 150 MB to churn.
_SCRATCH: dict[str, np.ndarray] = {}


def _scratch(key: str, shape: tuple, dtype) -> np.ndarray:
    buf = _SCRATCH.get(key)
    if buf is None or buf.shape != shape or buf.dtype != dtype:
        buf = np.empty(shape, dtype=dtype)
        _SCRATCH[key] = buf
    return buf


def release_scratch() -> None:
    """Drop the shared working memory. Prefer release_all()."""
    _SCRATCH.clear()


def cache_stats() -> dict:
    """
    What this module is currently holding, in objects and megabytes.

    An observable, so a test can assert that looks are interned and that
    release_all() works without reaching for module-private dictionaries —
    whose names and shapes are internal performance decisions.
    """
    mask_bytes = sum(m.nbytes for m in _MASKS.values())
    scratch_bytes = sum(b.nbytes for b in _SCRATCH.values())
    grain_bytes = sum(g._grain.nbytes for g in Grade._cache.values()
                      if g._grain is not None)
    return {
        "grades": len(Grade._cache),
        "masks": len(_MASKS),
        "buffers": len(_SCRATCH),
        "megabytes": round((mask_bytes + scratch_bytes + grain_bytes) / 1e6, 1),
    }


def release_all() -> None:
    """
    Free everything this module holds: scratch, interned grades, and masks.

    There used to be two cleanup calls and three caches, and the one they both
    missed was the largest: a vignette mask is 25 MB at 1080x1920, cached by
    (strength, size), and nothing cleared it. Clearing the interned grades did
    not help either, because the mask cache still held the references.

    One call, so there is nothing left to forget.
    """
    _SCRATCH.clear()
    _MASKS.clear()
    Grade.forget_all()


# -------------------------------------------------------------------- grade

class Grade:
    """
    A compiled look, ready to apply to frames of one fixed size.

    Build it once per (look, size) pair — `Grade.compile` interns them, so
    five clips sharing a preset share one object and one vignette mask.

    The public surface is one call:

        graded = grade.apply(frame_uint8, frame_index)

    `frame_index` only drives which phase of the grain cycle lands; passing a
    constant gives static grain, which reads as a dirty lens rather than film.
    """

    __slots__ = ("look", "size", "_matrix_t", "_offset", "_channel_lut",
                 "_tone_lut", "_vignette", "_grain", "_grain_stride",
                 "_pixels", "_needs_clip", "_glow", "_split")

    _cache: dict[tuple, "Grade"] = {}

    def __init__(self, look: Look, width: int, height: int):
        self.look = look
        self.size = (width, height)
        self._pixels = width * height

        matrix, offset = _linear_map(look)
        tone = _tone_curve(look.gamma, look.black, look.white)

        self._matrix_t = None
        self._offset = None
        self._channel_lut = None
        self._tone_lut = None

        if not _is_identity(matrix, offset, tone):
            self._channel_lut = _channel_table(matrix, offset, tone)
            if self._channel_lut is None:
                self._matrix_t = np.ascontiguousarray(matrix.T, dtype=np.float32)
                # A scalar add runs six times faster than adding a length-3
                # vector across the last axis of a (N, 3) buffer — numpy has to
                # set up a broadcast for the vector and cannot vectorise it the
                # same way. The offset is uniform across channels for every
                # look built from contrast and lift, so this is the normal
                # case, not a special one.
                self._offset = (float(offset[0]) if np.allclose(offset, offset[0])
                                else offset.astype(np.float32))
                # Emitting float32 straight out of the lookup saves a
                # conversion before the vignette multiply.
                self._tone_lut = None if tone is None else tone.astype(np.float32)

        self._vignette = _vignette_mask(look.vignette, width, height)
        self._grain, self._grain_stride = _grain_field(look.grain, width, height)
        self._glow = (look.glow, look.glow_threshold) if look.glow > 0 else None
        self._split = int(round(look.rgb_split)) if look.rgb_split >= 0.5 else 0

        # A final clip costs two passes over a 25 MB buffer, so only pay for
        # it when something can actually leave 0..255. Three ways to be safe:
        # a lookup table's entries are already bytes; an identity linear stage
        # passes the uint8 input straight through; and a vignette only scales
        # down. Only grain adds, and only an unclipped linear stage can
        # overshoot on contrast.
        #
        # That third case used to be missing, so a vignette-only look paid a
        # clip that provably could not do anything, on every frame.
        bounded = (self._tone_lut is not None
                   or self._channel_lut is not None
                   or self._matrix_t is None)
        self._needs_clip = (self._grain is not None or self._glow is not None
                            or not bounded)

    # -- construction ----------------------------------------------------

    @classmethod
    def compile(cls, look: Look, width: int, height: int) -> Optional["Grade"]:
        """
        Return a Grade for this look, or None when there is nothing to do.

        A neutral look returns None rather than an identity object, so an
        ungraded clip pays nothing at all — not even a function call per frame.
        """
        if not look.touches_pixels:
            return None
        key = (look, width, height)
        grade = cls._cache.get(key)
        if grade is None:
            grade = cls(look, width, height)
            cls._cache[key] = grade
        return grade

    @classmethod
    def forget_all(cls) -> None:
        """Drop interned grades. Called by release_all(); prefer that."""
        cls._cache.clear()

    @property
    def vignette(self):
        """
        The falloff mask as a single channel, or None.

        Exposed because the equivalence test needs the exact mask to compare
        against, and reaching for the private attribute would pin the fact
        that it is stored expanded to three channels — which is a speed
        decision that should stay free to change.
        """
        if self._vignette is None:
            return None
        return self._vignette[:, :, :1]

    @property
    def path(self) -> str:
        """
        Which of the three per-frame paths this look compiled to.

        "lut"      one table lookup per channel — the cheapest
        "matrix"   a 3x3 multiply, for anything that mixes channels
        "none"     no tonal stage at all; only vignette and grain
        """
        if self._channel_lut is not None:
            return "lut"
        return "matrix" if self._matrix_t is not None else "none"

    # -- application -----------------------------------------------------

    def apply(self, frame: np.ndarray, index: int = 0) -> np.ndarray:
        """
        Grade one uint8 (H, W, 3) frame and return a new uint8 frame.

        The input is never modified; the caller may be holding it for a
        crossfade.
        """
        if self._channel_lut is not None:
            out = self._channel_lut[frame, _CHANNELS]
        elif self._matrix_t is not None:
            out = self._linear(frame)
        else:
            out = frame

        if (self._vignette is not None or self._grain is not None
                or self._glow is not None or self._split):
            return self._texture(out, index)

        # No spatial stage, so nothing has copied the result out of the
        # shared scratch buffer yet. Do it here — the caller may hold this
        # frame across a crossfade, long after the next frame has overwritten
        # the buffer underneath it.
        if out.dtype != np.uint8:
            np.clip(out, 0.0, 255.0, out=out)
            return out.astype(np.uint8)
        return out if out is not frame else frame.copy()

    # -- stages ----------------------------------------------------------

    def _linear(self, frame: np.ndarray):
        """Saturation, temperature, contrast, lift and gamma in one pass each."""
        flat = frame.reshape(-1, 3)
        src = _scratch("lin_in", flat.shape, np.float32)
        dst = _scratch("lin_out", flat.shape, np.float32)

        np.copyto(src, flat)                      # uint8 -> float32
        np.dot(src, self._matrix_t, out=dst)      # the four collapsed knobs
        dst += self._offset

        if self._tone_lut is None:
            # Deliberately unclipped. A contrast above 1 pushes highlights
            # past white, and a vignette multiplying them back down is what
            # rolls them off; clipping here would flatten the corners of every
            # bright frame instead.
            return dst.reshape(frame.shape)

        # The curve is a table, so the value has to be quantised to index it —
        # which costs the clip that a power of a negative number would need
        # anyway.
        #
        # The half is what makes that quantisation round to nearest rather
        # than floor, and it matters more than it looks. Levels stretch the
        # table across a narrower input band, so they multiply whatever error
        # arrives at it: at `black` 0.30 and `white` 0.74 the band is 0.44
        # wide and a floor's half-level error comes out the other side as
        # more than two. One scalar add, and the error halves instead.
        np.clip(dst, 0.0, 255.0, out=dst)
        dst += 0.5
        idx = _scratch("tone_idx", flat.shape, np.uint8)
        np.copyto(idx, dst, casting="unsafe")
        return self._tone_lut[idx].reshape(frame.shape)

    def _texture(self, arr: np.ndarray, index: int):
        """
        The stages that are genuinely per-pixel, in the order light meets them.

        Glow spills out of the highlights before the lens darkens the corners,
        so a vignette dims the bloom the way it dims everything else. The
        split is a lens artefact and lands after both. Grain is the film, so
        it goes on last.
        """
        if arr.dtype == np.float32:
            work = arr
        else:
            work = _scratch("tex", arr.shape, np.float32)
            np.copyto(work, arr)

        if self._glow is not None:
            work += _bloom_field(work, *self._glow)
        if self._vignette is not None:
            work *= self._vignette
        if self._split:
            _split_channels(work, self._split)
        if self._grain is not None:
            work += self._grain_phase(index)

        if self._needs_clip:
            np.clip(work, 0.0, 255.0, out=work)
        return work.astype(np.uint8)

    def _grain_phase(self, index: int) -> np.ndarray:
        """
        One phase of the grain cycle.

        The phases are overlapping windows onto a single noise buffer rather
        than eight separate fields: same moving-grain look, a quarter of the
        memory, and no copy to take a window.
        """
        w, h = self.size
        offset = (index % GRAIN_PHASES) * self._grain_stride
        return self._grain[offset:offset + self._pixels * 3].reshape(h, w, 3)


_CHANNELS = np.arange(3)


# ------------------------------------------------------------- compilation

def _is_identity(matrix: np.ndarray, offset: np.ndarray, tone) -> bool:
    """True when the tonal stage would leave every pixel exactly as it found it."""
    return (tone is None
            and np.allclose(matrix, np.eye(3), atol=1e-6)
            and np.allclose(offset, 0.0, atol=1e-6))


def _channel_table(matrix: np.ndarray, offset: np.ndarray, tone):
    """
    Collapse the tonal stage into three 256-entry byte tables, or decline.

    When nothing mixes channels the whole linear stage and gamma become a
    single lookup, which is the cheapest per-frame path there is. Two things
    disqualify it:

    Off-diagonal terms. Saturation is a channel mix, so any look that touches
    saturation has to go through the matrix.

    Overshoot. A table is bytes, so building one clips anything the linear
    stage pushed past white — and a contrast above 1 does that deliberately,
    leaving headroom for the vignette to roll off. Clipping it here would
    flatten the corners of every bright frame. Gamma is the exception: the
    chain this replaces clipped before the power anyway, so there is nothing
    left to lose by then.
    """
    if not np.allclose(matrix - np.diag(np.diag(matrix)), 0.0, atol=1e-6):
        return None

    ramp = np.arange(256, dtype=np.float32)
    raw = ramp[:, None] * np.diag(matrix)[None, :] + offset[None, :]
    if tone is None and (raw.min() < -0.5 or raw.max() > 255.5):
        return None

    table = np.clip(raw, 0.0, 255.0)
    if tone is not None:
        table = tone[np.rint(table).astype(np.uint8)]
    return table.astype(np.uint8)


def _linear_map(look: Look) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse saturation, temperature, contrast and lift into one affine map.

    Working in 0..255 rather than 0..1 saves a divide and a multiply per
    frame; the only difference is that the offset is scaled by 255.
    """
    s = look.saturation
    saturation = s * np.eye(3, dtype=np.float64) + (1.0 - s) * np.tile(
        LUM_WEIGHTS.astype(np.float64), (3, 1))

    t = look.temperature
    temperature = np.diag([1.0 + 0.16 * t, 1.0 + 0.02 * t, 1.0 - 0.18 * t])

    c, lift = look.contrast, look.lift
    scale = (1.0 - lift) * c
    shift = (1.0 - lift) * (0.5 - 0.5 * c) + lift

    matrix = scale * (temperature @ saturation)
    offset = np.full(3, shift * 255.0, dtype=np.float64)
    return matrix, offset


def _tone_curve(gamma: float, black: float = 0.0,
                white: float = 1.0) -> Optional[np.ndarray]:
    """
    A 256-entry curve carrying input levels and gamma, or None for a no-op.

    Levels ride along with gamma for free: both are pointwise and monotone, so
    crushing the blacks costs nothing that gamma was not already costing. That
    is why the "extract" look — greyscale plus hard levels — needs no new
    per-frame stage at all; the greyscale half is a channel mix and folds into
    the matrix instead.
    """
    if white <= black:
        raise ValueError(
            f"grade white ({white}) must be above black ({black})")
    neutral = (abs(gamma - 1.0) < 1e-6 and abs(black) < 1e-6
               and abs(white - 1.0) < 1e-6)
    if neutral:
        return None

    ramp = np.arange(256, dtype=np.float32) / 255.0
    ramp = np.clip((ramp - black) / (white - black), 0.0, 1.0)
    return (np.power(ramp, gamma) * 255.0).astype(np.float32)


def _bloom_field(work: np.ndarray, amount: float, threshold: float) -> np.ndarray:
    """
    Light spilling out of the highlights.

    Built at a quarter of the resolution and scaled back up. A bloom is a
    low-frequency thing by definition — there is no detail in it to lose — so
    blurring a sixteenth of the pixels gives the same picture for a
    sixteenth of the work.

    Returns what to add, rather than adding in place, so the caller keeps
    control of the order the per-pixel stages run in.
    """
    from PIL import Image, ImageFilter

    h, w = work.shape[:2]
    cut = threshold * 255.0
    small = work[::4, ::4]
    over = np.clip((small - cut) / max(255.0 - cut, 1.0), 0.0, 1.0)
    lit = np.clip(small * over, 0.0, 255.0).astype(np.uint8)

    blurred = Image.fromarray(lit).filter(
        ImageFilter.GaussianBlur(max(2.0, w / 220.0)))
    spill = np.asarray(blurred.resize((w, h), Image.BILINEAR), dtype=np.float32)
    spill *= amount
    return spill


def _split_channels(work: np.ndarray, pixels: int) -> None:
    """
    Chromatic aberration: red one way, blue the other, in place.

    Two rolls of a single channel each. The green channel stays put, which is
    what keeps the picture from looking merely blurred — the eye reads the
    fringing as an optical fault rather than as softness.
    """
    work[:, :, 0] = np.roll(work[:, :, 0], -pixels, axis=1)
    work[:, :, 2] = np.roll(work[:, :, 2], pixels, axis=1)


_MASKS: dict[tuple, np.ndarray] = {}


def _vignette_mask(strength: float, width: int, height: int) -> Optional[np.ndarray]:
    """
    Radial corner falloff, flat across the middle third.

    The `- 0.32` keeps the centre untouched so the vignette reads as a lens
    rather than a spotlight, and the 1.6 exponent puts the falloff in the
    corners instead of spreading it up the edges.

    Stored expanded to three channels rather than one. A (H, W, 1) mask is a
    third of the memory but multiplying by it costs 20 ms a frame instead of
    3 — numpy cannot vectorise a broadcast across the innermost axis, and it
    has to do that on every one of two million pixels. Masks are cached by
    (strength, size) so looks sharing a vignette share the array.
    """
    if strength <= 0:
        return None

    key = (round(strength, 6), width, height)
    mask = _MASKS.get(key)
    if mask is not None:
        return mask

    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt(xs * xs + ys * ys) / np.sqrt(2.0)
    flat = np.clip(1.0 - strength * np.clip(radius - 0.32, 0.0, None) ** 1.6,
                   0.0, 1.0).astype(np.float32)
    mask = np.ascontiguousarray(np.broadcast_to(flat[:, :, None], (height, width, 3)))
    _MASKS[key] = mask
    return mask


def _grain_field(sigma: float, width: int, height: int):
    """
    One noise buffer wide enough to slice GRAIN_PHASES overlapping frames out.

    Three decisions worth knowing about:

    Sliding windows, not separate fields. White noise has no autocorrelation
    at any non-zero lag, so a shifted window is statistically as independent
    as a fresh draw — for a quarter of the memory of eight full frames.

    Three channels, not one. Adding a (H, W, 1) field to a (H, W, 3) frame
    means broadcasting across the innermost axis, which numpy cannot
    vectorise: 22 ms a frame against 3 ms for a matching-shape add. The
    buffer costs 14 MB more and saves 20 ms on every frame that has grain.

    The same value in all three channels. Independent per-channel noise would
    be chroma grain — coloured speckle — where this is luminance grain, the
    monochrome kind that reads as film stock. Repeating each sample three
    times keeps that, and striding by a multiple of three keeps the channels
    aligned as the window slides.

    A fixed seed keeps a re-render byte-identical, which is what makes it
    possible to change one clip and compare the result against the last cut.
    """
    if sigma <= 0:
        return None, 0

    pixels = width * height
    # Far enough that a phase change reads as new grain rather than as the
    # same grain sliding, cheap enough that the slack is a fifth of a frame.
    stride_px = max(1, pixels // 32)
    rng = np.random.default_rng(20260822)
    mono = rng.normal(0.0, sigma * 255.0,
                      size=pixels + stride_px * (GRAIN_PHASES - 1))
    return np.repeat(mono.astype(np.float32), 3), stride_px * 3
