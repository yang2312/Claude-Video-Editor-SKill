#!/usr/bin/env python3
"""
Tests for the cinematic-reel renderer.

    python tests/test_reel.py
    python tests/test_reel.py grade      # only tests whose name contains "grade"

No pytest, no network, no fixture files in the repo. A synthetic source video
is generated on first run and cached — it deliberately contains a dip to
black and a hard cut, so the scanner has something real to find.

What is worth testing here is narrow and specific:

  * the grade produces the same picture as the eight-step chain it replaces,
    because that collapse is the whole performance argument;
  * placement matches what the transition says, and refuses what it cannot
    draw, because an off-by-a-crossfade is invisible in a still;
  * a move the spec asks for actually happens, or is refused — a move that
    silently does nothing is the bug this suite exists over, and it has now
    happened twice;
  * the spec rejects what it cannot render, naming the clip.

Every test named `regression_*` corresponds to a defect that shipped.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import traceback
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import build_reel as br  # noqa: E402
import reel_grade as rg  # noqa: E402
import reel_timeline as rt  # noqa: E402
import scan_source as ss  # noqa: E402

WORK = os.path.join(tempfile.gettempdir(), "cinematic-reel-tests")
SOURCE = os.path.join(WORK, "source.mp4")
STILL = os.path.join(WORK, "still.png")

# Where the fixture's boundaries are, so tests can assert the scanner finds
# them rather than asserting a count that means nothing.
DIP_AT = 1.4          # centre of a ramp to black and back
CUT_AT = 2.6          # a hard scene change


# ------------------------------------------------------------------ harness

_tests: list = []


def test(fn):
    _tests.append(fn)
    return fn


def eq(actual, expected, what: str):
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected!r}, got {actual!r}")


def close(actual, expected, tol, what: str):
    if abs(actual - expected) > tol:
        raise AssertionError(f"{what}: expected {expected} +/- {tol}, got {actual}")


def raises(fn, fragment: str, what: str):
    try:
        fn()
    except Exception as error:  # noqa: BLE001 — that is the point
        if fragment.lower() not in str(error).lower():
            raise AssertionError(
                f"{what}: expected a message containing {fragment!r}, "
                f"got {str(error)!r}") from None
        return str(error)
    raise AssertionError(f"{what}: expected an error, none was raised")


def moves(framing, source, a=0.0, b=1.0) -> bool:
    """Does this framing actually produce different pixels across the shot?"""
    return not np.array_equal(framing.apply(source.frame(0), a),
                              framing.apply(source.frame(0), b))


# ----------------------------------------------------------------- fixtures

def build_fixtures() -> None:
    """A short source with a dip, a hard cut, moving picture and a tone."""
    os.makedirs(WORK, exist_ok=True)
    from PIL import Image

    if not os.path.exists(STILL):
        rng = np.random.default_rng(11)
        art = (np.linspace(20, 235, 640)[None, :, None]
               + np.linspace(0, 40, 420)[:, None, None]
               + rng.normal(0, 8, (420, 640, 3)))
        Image.fromarray(np.clip(art, 0, 255).astype(np.uint8)).save(STILL)

    if os.path.exists(SOURCE):
        return

    from moviepy import AudioArrayClip, VideoClip

    def frame(t):
        # A hard-edged bar sliding across a gradient: motion a resampler
        # cannot fake and a crop window will visibly track.
        base = np.linspace(0, 255, 640, dtype=np.float32)[None, :, None]
        img = np.repeat(np.repeat(base, 360, axis=0), 3, axis=2)
        x = int(40 + t * 120) % 560
        img[:, x:x + 60] = 255.0

        if t < DIP_AT - 0.2:                       # scene A
            img[:, :, 2] *= 0.55
        elif t < CUT_AT:                           # scene B, after the dip
            img[:, :, 0] *= 0.55
        else:                                      # scene C, hard cut
            img = 255.0 - img
            img[:, :, 1] *= 0.4

        # A ramp to black and back, 0.4s wide — the boundary a spike detector
        # cannot see, because it never spikes.
        fade = abs(t - DIP_AT) / 0.2
        if fade < 1.0:
            img *= fade ** 2
        return img.astype(np.uint8)

    rate = 22050
    samples = np.linspace(0, 4.0, int(rate * 4.0), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 440 * samples)
    audio = AudioArrayClip(np.stack([tone, tone], axis=1), fps=rate)

    clip = VideoClip(frame, duration=4.0).with_fps(24)
    clip.audio = audio
    clip.write_videofile(SOURCE, fps=24, codec="libx264", audio_codec="aac",
                         preset="ultrafast", logger=None)
    clip.close()
    audio.close()


def decode_audio(path: str, rate: int = 8000) -> np.ndarray:
    """
    Mono float samples, straight out of ffmpeg.

    moviepy's own AudioFileClip.to_soundarray() returns near-silence for files
    ffmpeg decodes at full level, so anything checking whether audio survived
    has to go around it.
    """
    import imageio_ffmpeg

    raw = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path, "-vn",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(rate), "-ac", "1", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def make_args(**overrides):
    """A parsed-CLI stand-in. Everything None, as argparse now leaves it."""
    args = types.SimpleNamespace(
        input=None, output=None, hold=None, size=None, aspect=None, fps=None,
        crossfade=None, crf=None, preset=None, threads=None, codec=None,
        resample=None, audio=None, guard=None)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def spec_for(clips, args=None, **top):
    raw = {"source": SOURCE, "output": os.path.join(WORK, "out.mp4"),
           "size": "180x320", "fps": 24, "clips": clips}
    raw.update(top)
    return br.ReelSpec.load(raw, args or make_args())


# --------------------------------------------------------------- timestamps

@test
def timestamps_parse_every_accepted_form():
    eq(br.parse_timestamp(12), 12.0, "bare int")
    eq(br.parse_timestamp("12"), 12.0, "SS")
    eq(br.parse_timestamp("1:30"), 90.0, "MM:SS")
    eq(br.parse_timestamp("1:00:05"), 3605.0, "HH:MM:SS")
    close(br.parse_timestamp("0:02.5"), 2.5, 1e-9, "fractional seconds")


@test
def timestamps_reject_nonsense_by_name():
    raises(lambda: br.parse_timestamp("1:2:3:4"), "too many", "four groups")
    raises(lambda: br.parse_timestamp("banana"), "non-numeric", "words")
    raises(lambda: br.parse_timestamp(-4), "negative", "negative number")
    raises(lambda: br.parse_timestamp("  "), "empty", "blank")


@test
def regression_duration_accepts_the_same_time_formats_as_start():
    """`duration` used to go through bare float(), losing the clip index."""
    clip = br.ClipSpec.parse({"start": "0:01", "duration": "0:02"}, 3, 3.0, 100.0)
    close(clip.length, 2.0, 1e-9, "duration given as MM:SS")
    message = raises(lambda: br.ClipSpec.parse({"start": 0, "duration": "banana"},
                                               3, 3.0, 100.0),
                     "non-numeric", "unparseable duration")
    if "clips[3]" not in message:
        raise AssertionError(f"error did not name the clip: {message!r}")


@test
def geometry_parses_and_rounds_to_even():
    eq(br.parse_size("1081x1921"), (1080, 1920), "odd sizes rounded down")
    close(br.parse_aspect("9:16"), 0.5625, 1e-9, "aspect ratio")
    raises(lambda: br.parse_aspect("9"), "9:16", "aspect without a colon")
    raises(lambda: br.parse_size("wide"), "1080x1920", "unparseable size")


# ------------------------------------------------------------- camera move

@test
def a_scalar_zoom_resolves_for_every_motion():
    eq(rt.CameraMove.resolve(motion="zoom-in", zoom=1.2).zoom, (1.0, 1.2), "zoom-in")
    eq(rt.CameraMove.resolve(motion="zoom-out", zoom=1.2).zoom, (1.2, 1.0), "zoom-out")
    eq(rt.CameraMove.resolve(motion="pan-right", zoom=1.2).zoom, (1.2, 1.2),
       "a pan holds the zoom it was given")
    eq(rt.CameraMove.resolve(motion="none", zoom=1.2).zoom, (1.2, 1.2),
       "a held shot sits at the zoom it was given")


@test
def regression_a_scalar_zoom_is_no_longer_dropped_on_a_pan():
    """`zoom` under pan-* used to resolve to (1.0, 1.0) and vanish."""
    move = rt.CameraMove.resolve(motion="pan-right", zoom=1.3)
    if move.widest == 1.0:
        raise AssertionError("the zoom was discarded again")
    eq(move.widest, 1.3, "widest scale the move reaches")


@test
def omitting_zoom_still_means_no_zoom_on_a_pan():
    """Specs written before the fix must keep their framing exactly."""
    eq(rt.CameraMove.resolve(motion="pan-right").zoom, (1.0, 1.0), "pan, no zoom")
    eq(rt.CameraMove.resolve(motion="none").zoom, (1.0, 1.0), "held, no zoom")
    eq(rt.CameraMove.resolve(motion="zoom-in").zoom, (1.0, rt.DEFAULT_ZOOM),
       "a zoom move still needs somewhere to travel")


@test
def camera_moves_reject_values_they_cannot_render():
    raises(lambda: rt.CameraMove.resolve(motion="dolly"), "zoom-in", "unknown motion")
    raises(lambda: rt.CameraMove.resolve(zoom=0.5), ">= 1.0", "zoom below 1")
    raises(lambda: rt.CameraMove.resolve(zoom=[1.0, 1.1, 1.2]), "exactly 2",
           "three-element zoom")
    raises(lambda: rt.CameraMove.resolve(pan=5.0), "between 0 and 1", "pan above 1")
    raises(lambda: rt.CameraMove.resolve(pan=-0.5), "between 0 and 1", "negative pan")
    raises(lambda: rt.CameraMove.resolve(anchor=1.5), "between 0 and 1", "anchor above 1")
    raises(lambda: rt.CameraMove.resolve(anchor=[0.5]), "exactly 2", "one-element anchor")


@test
def regression_pan_out_of_range_is_refused_not_clamped():
    """pan was the one numeric field with no bounds; -0.5 reversed direction."""
    raises(lambda: br.ClipSpec.parse({"start": 0, "end": 1, "motion": "pan-right",
                                      "pan": -0.5}, 0, 3.0, 100.0),
           "pan must be between 0 and 1", "negative pan through a spec")


@test
def an_anchor_can_place_the_window_in_both_axes():
    eq(rt.CameraMove.resolve(anchor=0.2).anchor, (0.2, 0.5), "a number is horizontal")
    eq(rt.CameraMove.resolve(anchor=[0.2, 0.9]).anchor, (0.2, 0.9), "a pair is both")


# -------------------------------------------------------------------- grade

def reference_grade(look: rg.Look, frame: np.ndarray, mask, grain) -> np.ndarray:
    """
    The eight-step chain, written out plainly.

    This is the specification the compiled Grade has to match. It is
    deliberately the slow, obvious version — if the two ever disagree, this
    one is right.
    """
    f = frame.astype(np.float32) / 255.0
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    if look.saturation != 1.0:
        lum = (f * weights).sum(axis=2, keepdims=True)
        f = lum + (f - lum) * look.saturation
    if look.temperature != 0.0:
        t = look.temperature
        f = f * np.array([1 + 0.16 * t, 1 + 0.02 * t, 1 - 0.18 * t], np.float32)
    if look.contrast != 1.0:
        f = (f - 0.5) * look.contrast + 0.5
    if look.lift > 0:
        f = f * (1.0 - look.lift) + look.lift
    if look.gamma != 1.0:
        f = np.clip(f, 0.0, 1.0) ** look.gamma
    if mask is not None:
        f = f * mask
    if grain is not None:
        f = f + grain
    return (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)


def _check_against_reference(look: rg.Look, label: str) -> None:
    from dataclasses import replace

    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, (120, 90, 3), dtype=np.uint8)
    look = replace(look, grain=0.0)          # grain is a fresh draw either way
    grade = rg.Grade.compile(look, 90, 120)
    want = reference_grade(look, frame, grade.vignette, None).astype(np.int16)
    got = grade.apply(frame, 0).astype(np.int16)
    worst = int(np.abs(want - got).max())
    if worst > 2:
        raise AssertionError(
            f"{label}: compiled grade differs from the reference chain by "
            f"{worst} levels (2 is the quantisation budget)")


@test
def grade_matches_the_chain_it_replaces():
    for name in ("vintage", "faded", "warm", "vivid"):
        _check_against_reference(rg.PRESETS[name], name)


@test
def regression_the_lookup_path_matches_the_reference_too():
    """
    Every shipped preset takes the matrix path, so the cheaper lookup path
    used to be exercised by nothing at all.
    """
    covered = set()
    for label, look in (("gamma only", rg.Look(gamma=0.9)),
                        ("lift only", rg.Look(lift=0.12)),
                        ("contrast below 1", rg.Look(contrast=0.85)),
                        ("gamma + vignette", rg.Look(gamma=1.1, vignette=0.3))):
        grade = rg.Grade.compile(look, 90, 120)
        covered.add(grade.path)
        _check_against_reference(look, label)
    if "lut" not in covered:
        raise AssertionError(
            f"no case reached the lookup path; saw {sorted(covered)}")


@test
def each_look_compiles_to_the_cheapest_path_it_can():
    eq(rg.Grade.compile(rg.PRESETS["vivid"], 64, 64).path, "matrix",
       "saturation mixes channels, so it needs the matrix")
    eq(rg.Grade.compile(rg.Look(gamma=0.9), 64, 64).path, "lut",
       "a per-channel curve is a lookup")
    eq(rg.Grade.compile(rg.Look(vignette=0.3), 64, 64).path, "none",
       "a vignette needs no tonal stage at all")


@test
def regression_grade_leaves_highlights_for_the_vignette_to_roll_off():
    """
    Clipping before the vignette flattens the corners of a bright frame.

    A contrast above 1 pushes highlights past white on purpose; the vignette
    multiplying them back down is what rolls them off.
    """
    grade = rg.Grade.compile(rg.Look(contrast=1.3, vignette=0.6), 64, 64)
    white = np.full((64, 64, 3), 255, np.uint8)
    corner = int(grade.apply(white, 0)[2, 2, 0])
    if corner < 200:
        raise AssertionError(
            f"corner clipped to {corner}; the overshoot was thrown away "
            "before the vignette could use it")


@test
def regression_a_vignette_only_look_skips_the_pointless_clip():
    """The comment claimed this; the predicate did not do it."""
    grade = rg.Grade.compile(rg.Look(vignette=0.2), 64, 64)
    if grade._needs_clip:
        raise AssertionError(
            "a vignette-only look still pays a clip that cannot do anything")


@test
def neutral_grade_compiles_to_nothing():
    eq(rg.Grade.compile(rg.Look(), 64, 64), None, "neutral look")
    eq(rg.Grade.compile(rg.Look(softness=0.4), 64, 64), None,
       "softness alone is not a per-pixel stage")
    if rg.Grade.compile(rg.Look(vignette=0.2), 64, 64) is None:
        raise AssertionError("a vignette is a per-pixel stage and must compile")


@test
def grade_never_writes_to_the_frame_it_was_given():
    grade = rg.Grade.compile(rg.PRESETS["vivid"], 64, 64)
    frame = np.full((64, 64, 3), 120, np.uint8)
    before = frame.copy()
    grade.apply(frame, 0)
    if not np.array_equal(frame, before):
        raise AssertionError("apply() modified its input in place")


@test
def successive_results_are_independent_arrays():
    """
    The invariant a crossfade depends on.

    The timeline holds one shot's frame while asking a second shot for its
    own, and both go through the same pooled buffers. If a result were ever a
    view into that pool, the dissolve would blend a frame with itself.
    """
    for look in (rg.PRESETS["vivid"], rg.Look(gamma=0.9), rg.Look(vignette=0.3)):
        grade = rg.Grade.compile(look, 64, 64)
        first = grade.apply(np.full((64, 64, 3), 60, np.uint8), 0)
        held = first.copy()
        grade.apply(np.full((64, 64, 3), 200, np.uint8), 1)
        if not np.array_equal(first, held):
            raise AssertionError(
                f"{grade.path} path: the first result changed when the second "
                "was computed — it was a view into the shared buffer")


@test
def looks_are_interned_and_releasable():
    rg.release_all()
    a = rg.Grade.compile(rg.PRESETS["vivid"], 200, 160)
    b = rg.Grade.compile(rg.PRESETS["vivid"], 200, 160)
    if a is not b:
        raise AssertionError("two clips sharing a preset built two grades")
    held = rg.cache_stats()
    eq(held["grades"], 1, "grades interned for two identical looks")
    if held["masks"] < 1 or held["megabytes"] <= 0:
        raise AssertionError(f"nothing was cached: {held}")

    rg.release_all()
    freed = rg.cache_stats()
    eq(freed, {"grades": 0, "masks": 0, "buffers": 0, "megabytes": 0.0},
       "everything released")


@test
def grain_is_luminance_grain_of_the_right_strength():
    sigma = 0.05
    grade = rg.Grade.compile(rg.Look(grain=sigma), 200, 160)
    flat = np.full((160, 200, 3), 128, np.uint8)
    noise = grade.apply(flat, 0).astype(np.float32) - 128.0

    close(float(noise.std()), sigma * 255, 1.5, "grain sigma")
    spread = float(np.abs(noise.max(axis=2) - noise.min(axis=2)).mean())
    close(spread, 0.0, 0.6, "per-pixel spread across channels (chroma grain)")

    a = grade.apply(flat, 0).astype(np.float32).ravel()
    b = grade.apply(flat, 1).astype(np.float32).ravel()
    close(float(np.corrcoef(a, b)[0, 1]), 0.0, 0.05,
          "correlation between two grain phases")


@test
def unknown_grade_names_say_what_was_expected():
    raises(lambda: rg.resolve_look("cinematic"), "vintage", "unknown preset")
    raises(lambda: rg.resolve_look({"saturationn": 1.2}), "unknown grade key",
           "misspelled key")
    _, name = rg.resolve_look({"preset": "warm", "grain": 0.02})
    eq(name, "warm+custom", "preset with overrides is labelled as such")


# ----------------------------------------------------------------- planning

def dummy_shots(lengths):
    class Flat:
        def __init__(self, value, duration):
            self.value = value
            self.duration = duration

        def frame(self, t, index):
            return np.full((4, 4, 3), self.value, np.uint8)

        def release(self):
            pass

    return [Flat(40 + i * 60, d) for i, d in enumerate(lengths)]


@test
def crossfade_overlaps_and_a_cut_does_not():
    line = rt.Timeline(dummy_shots([2.0, 2.0, 2.0]),
                       ["cut", "crossfade", "cut"], 0.5, 30)
    eq([round(p.start, 3) for p in line.placements], [0.0, 1.5, 3.5], "shot starts")
    close(line.duration, 5.5, 1e-6, "total length")


@test
def regression_a_clip_swallowed_by_the_next_dissolve_is_refused():
    """
    The validator used to measure only the incoming clip, so a 0.2s clip
    followed by a 0.35s crossfade passed — and never appeared on screen.
    """
    raises(lambda: rt.plan([0.2, 2.0], ["cut", "crossfade"], 0.35),
           "swallowed whole", "short outgoing clip")
    # The same lengths are fine on a hard cut, which has no overlap.
    rt.plan([0.2, 2.0], ["cut", "cut"], 0.35)


@test
def regression_three_overlapping_shots_are_refused():
    """
    frame() blends two layers. The validator used to permit three and the
    oldest was dropped mid-dissolve, silently.
    """
    message = raises(lambda: rt.plan([2.0, 0.35, 2.0],
                                     ["cut", "crossfade", "crossfade"], 0.3),
                     "blends two shots, not three", "three-way overlap")
    if "clips[1]" not in message:
        raise AssertionError(f"error did not name the middle clip: {message!r}")
    # Halving the crossfade makes the same reel legal.
    rt.plan([2.0, 0.35, 2.0], ["cut", "crossfade", "crossfade"], 0.15)


@test
def planning_is_what_validation_does():
    """A spec is validated by planning it, so the two can never disagree."""
    raises(lambda: spec_for([{"start": 0.0, "end": 1.0},
                             {"start": 1.0, "end": 1.2, "transition": "crossfade"},
                             {"start": 2.0, "end": 3.0, "transition": "crossfade"}],
                            crossfade=0.6),
           "clips[", "an unrenderable reel is refused at parse time")


@test
def a_flash_is_recorded_at_the_junction_it_belongs_to():
    line = rt.Timeline(dummy_shots([1.0, 1.0]), ["cut", "flash"], 0.4, 30)
    eq([round(f, 3) for f in line.flashes], [1.0], "flash times")

    quiet = float(line.frame(0.5).mean())
    peak = float(line.frame(1.0).mean())
    if peak <= quiet + 40:
        raise AssertionError(
            f"flash did not bloom: {quiet:.0f} before, {peak:.0f} at the junction")
    close(float(line.frame(0.2).mean()), 40.0, 0.5,
          "frames away from the flash are untouched")


@test
def regression_a_flash_on_the_first_clip_is_refused():
    """It used to be swallowed: no flash, no error, no mention in the report."""
    raises(lambda: br.ClipSpec.parse({"start": 0, "end": 1, "transition": "flash"},
                                     0, 3.0, 100.0),
           "nothing before it", "flash on clip 0")


@test
def the_first_clip_never_dissolves_in_from_nothing():
    line = rt.Timeline(dummy_shots([1.0, 1.0]), ["crossfade", "crossfade"], 0.3, 30)
    close(line.placements[0].start, 0.0, 1e-9, "first shot starts at zero")


@test
def a_dissolve_blends_both_shots_at_its_midpoint():
    line = rt.Timeline(dummy_shots([1.0, 1.0]), ["cut", "crossfade"], 0.4, 30)
    close(float(line.frame(0.8).mean()), 70.0, 3.0,
          "midpoint of a 40 -> 100 dissolve")


@test
def only_the_shots_on_screen_are_asked_for_a_frame():
    calls = {"n": 0}

    class Counting:
        duration = 1.0

        def frame(self, t, index):
            calls["n"] += 1
            return np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    line = rt.Timeline([Counting() for _ in range(4)],
                       ["cut", "cut", "cut", "cut"], 0.0, 30)
    line.frame(2.5)
    eq(calls["n"], 1, "frames rendered for a moment with one visible shot")


# ------------------------------------------------------------------ framing

@test
def a_crop_window_moves_when_it_is_told_to():
    still = rt.StillSource(STILL)
    move = rt.CameraMove.resolve(motion="pan-right", pan=0.6)
    panning = rt.CropFraming(still.size, 120, 200, move, 0.0)
    if not moves(panning, still):
        raise AssertionError("pan-right produced identical first and last frames")
    eq(panning.static, False, "a pan is not static")

    held = rt.CropFraming(still.size, 120, 200, rt.CameraMove.resolve(), 0.0)
    eq(held.static, True, "no motion is static")
    if moves(held, still):
        raise AssertionError("a static framing produced two different frames")


@test
def regression_fit_framing_honours_a_zoom():
    still = rt.StillSource(STILL)
    move = rt.CameraMove.resolve(motion="zoom-in", zoom=[1.0, 1.4])
    framing = rt.FitFraming(still.size, 120, 260, 0.0, still.frame(0), move)
    eq(framing.static, False, "a zoom is not static")

    def band(progress):
        arr = framing.apply(still.frame(0), progress)
        rows = arr.mean(axis=(1, 2))
        lit = np.flatnonzero(rows > rows.max() * 0.55)
        return int(lit.max() - lit.min())

    if band(1.0) <= band(0.0):
        raise AssertionError(
            f"fit zoom did not grow the picture: {band(0.0)}px -> {band(1.0)}px")


@test
def regression_fit_framing_honours_a_pan():
    """
    The second half of the same bug. Fit ignored pan entirely, and a scalar
    zoom — the thing that gives a pan room in fit mode — was being dropped
    before it arrived.
    """
    still = rt.StillSource(STILL)
    move = rt.CameraMove.resolve(motion="pan-right", zoom=1.3, pan=0.6)
    framing = rt.FitFraming(still.size, 120, 260, 0.0, still.frame(0), move)
    if framing.slack <= 1.0:
        raise AssertionError(f"a zoom of 1.3 gave no slack ({framing.slack})")
    eq(framing.static, False, "a pan with slack is not static")
    if not moves(framing, still):
        raise AssertionError("fit pan produced identical first and last frames")


@test
def regression_fit_pan_without_room_is_refused_not_frozen():
    """
    At zoom 1.0 a fit pan has nowhere to travel. Rendering it as a still and
    reporting success is the exact failure this codebase says it cares about.
    """
    raises(lambda: br.ClipSpec.parse({"start": 0, "end": 1, "fit": True,
                                      "motion": "pan-right"}, 2, 3.0, 100.0),
           "nowhere to go", "fit pan at zoom 1.0")
    # With room, it is accepted.
    br.ClipSpec.parse({"start": 0, "end": 1, "fit": True,
                       "motion": "pan-right", "zoom": 1.2}, 2, 3.0, 100.0)


@test
def regression_both_framings_agree_on_what_static_means():
    """
    `motion: none` with an explicit zoom pair used to animate under fit and
    freeze under crop — the same spec, decided by a flag about letterboxing.
    """
    still = rt.StillSource(STILL)
    for label, zoom, expect_move in (("a held zoom pair", [1.0, 1.5], True),
                                     ("no move at all", None, False),
                                     ("an equal pair", [1.2, 1.2], False)):
        move = rt.CameraMove.resolve(motion="none", zoom=zoom)
        crop = rt.CropFraming(still.size, 120, 200, move, 0.0)
        fit = rt.FitFraming(still.size, 120, 200, 0.0, still.frame(0), move)
        eq(crop.static, fit.static, f"{label}: crop and fit agree on static")
        eq(moves(crop, still), expect_move, f"{label}: crop moves")
        eq(moves(fit, still), expect_move, f"{label}: fit moves")


@test
def an_off_centre_anchor_moves_the_window():
    still = rt.StillSource(STILL)
    left = rt.CropFraming(still.size, 120, 200,
                          rt.CameraMove.resolve(anchor=0.0), 0.0)
    right = rt.CropFraming(still.size, 120, 200,
                           rt.CameraMove.resolve(anchor=1.0), 0.0)
    if np.array_equal(left.apply(still.frame(0), 0.0),
                      right.apply(still.frame(0), 0.0)):
        raise AssertionError("anchor 0.0 and 1.0 framed the same pixels")


@test
def regression_a_vertical_anchor_actually_moves_the_window():
    """The vertical clamp was dead code; anchor only ever worked horizontally."""
    still = rt.StillSource(STILL)          # 640x420, wider than 9:16
    # A tall output crops vertically only when the window is shorter than the
    # source, so use a wide output to leave vertical slack.
    top = rt.CropFraming(still.size, 400, 120,
                         rt.CameraMove.resolve(anchor=[0.5, 0.0]), 0.0)
    bottom = rt.CropFraming(still.size, 400, 120,
                            rt.CameraMove.resolve(anchor=[0.5, 1.0]), 0.0)
    if np.array_equal(top.apply(still.frame(0), 0.0),
                      bottom.apply(still.frame(0), 0.0)):
        raise AssertionError("vertical anchor 0.0 and 1.0 framed the same pixels")


@test
def a_held_still_is_framed_once_and_reused():
    still = rt.StillSource(STILL)
    framing = rt.FitFraming(still.size, 120, 260, 0.0, still.frame(0),
                            rt.CameraMove.resolve())
    shot = rt.Shot(still, framing, None, 2.0)
    eq(shot.freezable, True, "a static still is freezable")
    for i in range(5):
        shot.frame(i * 0.4, i)
    eq(shot.frames_built, 1, "framings built for five frames of a held still")


@test
def footage_is_never_frozen():
    source = rt.ClipSource(SOURCE, 0.0, 1.0, 1.0, (640, 360), 24.0)
    framing = rt.CropFraming(source.size, 120, 200, rt.CameraMove.resolve(), 0.0)
    shot = rt.Shot(source, framing, None, 1.0)
    eq(shot.freezable, False, "moving footage must not be frozen")
    shot.release()


@test
def the_fast_resampler_renders_and_differs_from_quality():
    still = rt.StillSource(STILL)
    move = rt.CameraMove.resolve(motion="zoom-in", zoom=1.3)
    a = rt.CropFraming(still.size, 240, 400, move, 0.0, "quality")
    b = rt.CropFraming(still.size, 240, 400, move, 0.0, "fast")
    left, right = a.apply(still.frame(0), 0.5), b.apply(still.frame(0), 0.5)
    eq(left.shape, right.shape, "both resamplers produce the same geometry")
    if np.array_equal(left, right):
        raise AssertionError("'fast' produced a byte-identical picture; "
                             "either it is not wired up or the source is flat")


@test
def softness_blurs_without_changing_geometry():
    still = rt.StillSource(STILL)
    move = rt.CameraMove.resolve()
    sharp = rt.CropFraming(still.size, 240, 400, move, 0.0).apply(still.frame(0), 0)
    soft = rt.CropFraming(still.size, 240, 400, move, 0.6).apply(still.frame(0), 0)
    eq(sharp.shape, soft.shape, "softness leaves geometry alone")
    edge = lambda a: float(np.abs(np.diff(a.astype(np.float32), axis=1)).mean())
    if edge(soft) >= edge(sharp):
        raise AssertionError(
            f"softness did not reduce edge energy ({edge(sharp):.2f} -> {edge(soft):.2f})")


# --------------------------------------------------------------------- spec

@test
def a_clip_past_the_end_of_the_source_is_named():
    raises(lambda: spec_for([{"start": 3.0, "end": 9.0}]),
           "past the end", "range beyond the source")
    raises(lambda: spec_for([{"start": 2.0, "end": 1.0}]),
           "before it starts", "backwards range")
    raises(lambda: spec_for([{"end": 1.0}]),
           "clips[0]", "clip with no start names its index")


@test
def regression_unknown_keys_are_refused_at_every_level():
    """A misspelled key used to be absorbed in silence."""
    raises(lambda: spec_for([{"start": 0, "end": 1, "durationn": 9}]),
           "unknown key", "misspelled clip key")
    raises(lambda: spec_for([{"start": 0, "end": 1}], crossfeed=0.9),
           "unknown key", "misspelled top-level key")
    # Underscore keys are comments and stay allowed.
    spec_for([{"start": 0, "end": 1, "_note": "keep"}], _comment="fine")


@test
def regression_aspect_and_size_together_are_refused():
    """`size` used to win in silence, so editing `aspect` did nothing."""
    raw = {"source": SOURCE, "output": os.path.join(WORK, "out.mp4"),
           "aspect": "1:1", "size": "180x320", "clips": [{"start": 0, "end": 1}]}
    raises(lambda: br.ReelSpec.load(raw, make_args()), "pick one",
           "aspect and size together")


@test
def regression_speed_on_a_still_is_refused():
    """It used to render one length, report another, and validate a third."""
    raises(lambda: br.ClipSpec.parse({"image": STILL, "duration": 2.8, "speed": 2.0},
                                     0, 3.0, 100.0),
           "no time axis", "speed on an image")


@test
def an_image_clip_cannot_also_name_a_source_range():
    raises(lambda: br.ClipSpec.parse({"image": STILL, "duration": 1.0, "start": 2.0},
                                     0, 3.0, 100.0),
           "timed by `duration` alone", "image with a range")


@test
def encoder_settings_are_validated_before_anything_is_decoded():
    raises(lambda: spec_for([{"start": 0, "end": 1}], preset="turbo"),
           "unknown preset", "bad x264 preset")
    raises(lambda: spec_for([{"start": 0, "end": 1}], resample="blurry"),
           "unknown resample", "bad resample mode")
    raises(lambda: spec_for([{"start": 0, "end": 1}], crossfade=-1),
           "cannot be negative", "negative crossfade")


@test
def a_missing_source_is_refused_by_name():
    raw = {"source": os.path.join(WORK, "nope.mp4"), "output": "x.mp4",
           "clips": [{"start": 0, "end": 1}]}
    raises(lambda: br.ReelSpec.load(raw, make_args()), "not found", "missing source")
    raises(lambda: br.ReelSpec.load({"source": SOURCE, "clips": [{"start": 0, "end": 1}]},
                                    make_args()), "no output path", "missing output")
    raises(lambda: br.ReelSpec.load({"source": SOURCE, "output": "x.mp4", "clips": []},
                                    make_args()), "no 'clips'", "empty clips")


@test
def the_hold_default_fills_in_a_missing_end():
    spec = spec_for([{"start": 0.5}], hold=1.25)
    close(spec.clips[0].length, 1.25, 1e-9, "clip length from hold")


@test
def regression_a_typed_cli_flag_beats_the_spec_file():
    """`raw.get(key, args.fps)` could not tell a typed flag from a default."""
    spec = spec_for([{"start": 0, "end": 1}], args=make_args(fps=60), fps=24)
    eq(spec.fps, 60, "explicitly typed --fps")

    spec = spec_for([{"start": 0, "end": 1}], fps=24)
    eq(spec.fps, 24, "spec value when no flag was typed")

    # spec_for pins fps, so the untouched case needs a bare spec.
    bare = br.ReelSpec.load({"source": SOURCE, "output": os.path.join(WORK, "o.mp4"),
                             "size": "180x320", "clips": [{"start": 0, "end": 1}]},
                            make_args())
    eq(bare.fps, 30, "the built-in default when neither says")


@test
def auto_motion_resolves_through_a_spec_and_never_repeats():
    spec = spec_for([{"start": 0, "end": 0.5, "motion": "auto"},
                     {"start": 1, "end": 1.5, "motion": "auto", "transition": "cut"},
                     {"start": 2, "end": 2.5, "motion": "auto", "transition": "cut"}])
    got = [c.move.motion for c in spec.clips]
    eq(got, ["zoom-in", "pan-right", "zoom-out"], "auto rotation through a spec")
    for a, b in zip(got, got[1:]):
        if a == b:
            raise AssertionError(f"auto motion repeated {a!r} back to back")


@test
def two_of_the_same_move_in_a_row_is_warned_about():
    spec = spec_for([{"start": 0, "end": 0.5, "motion": "zoom-in"},
                     {"start": 1, "end": 1.5, "motion": "zoom-in", "transition": "cut"}])
    if not any("mechanical" in w for w in spec.warnings):
        raise AssertionError(f"no warning about repeated moves: {spec.warnings}")


@test
def a_grade_override_object_survives_the_spec():
    spec = spec_for([{"start": 0, "end": 1,
                      "grade": {"preset": "warm", "vignette": 0.5}}])
    eq(spec.clips[0].grade_name, "warm+custom", "grade name")
    close(spec.clips[0].look.vignette, 0.5, 1e-9, "overridden vignette")
    close(spec.clips[0].look.saturation, rg.PRESETS["warm"].saturation, 1e-9,
          "preset value kept where not overridden")


# -------------------------------------------------------------------- guard

@test
def the_scanner_finds_both_a_dip_and_a_hard_cut():
    result = ss.scan(SOURCE, rate=12.0, min_length=0.3)
    eq(result["status"], "success", "scan status")

    if not any(abs(t - DIP_AT) < 0.25 for t in result["dips"]):
        raise AssertionError(
            f"the dip at {DIP_AT}s was missed; found {result['dips']}")
    if not any(abs(t - CUT_AT) < 0.25 for t in result["hard_cuts"]):
        raise AssertionError(
            f"the cut at {CUT_AT}s was missed; found {result['hard_cuts']}")
    if not result["clean"]:
        raise AssertionError("no clean stretch reported in a 4s source")


@test
def the_scanner_reports_failure_the_same_way_the_renderer_does():
    raises(lambda: ss.scan(os.path.join(WORK, "nope.mp4")), "not found",
           "missing source")

    argv = sys.argv
    sys.argv = ["scan_source.py", "-i", os.path.join(WORK, "nope.mp4")]
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            code = ss.main()
    finally:
        sys.argv = argv
    eq(code, 1, "exit code")
    payload = json.loads(buffer.getvalue())
    eq(payload["status"], "error", "status field matches build_reel's shape")
    eq(payload["error_type"], "FileNotFoundError", "error type reported")


@test
def regression_a_range_straddling_a_dip_is_refused_before_encoding():
    """
    The loop the pipeline was missing. The scan knew where the dips were and
    the renderer never saw them, so a bad range was found only afterwards —
    by a human, reading brightness numbers off a finished file.
    """
    guard = os.path.join(WORK, "scan.json")
    with open(guard, "w", encoding="utf-8") as handle:
        json.dump(ss.scan(SOURCE, rate=12.0, min_length=0.3), handle)

    message = raises(
        lambda: spec_for([{"start": DIP_AT - 0.4, "end": DIP_AT + 0.4}], guard=guard),
        "straddles", "a range across the dip")
    if "clips[0]" not in message:
        raise AssertionError(f"error did not name the clip: {message!r}")

    # A range inside a clean stretch is accepted.
    spec = spec_for([{"start": 1.8, "end": 2.4}], guard=guard)
    eq(spec.guard, guard, "guard path carried onto the spec")


@test
def a_reel_with_no_guard_says_so():
    spec = spec_for([{"start": 0.2, "end": 1.0}])
    if not any("guard" in w for w in spec.warnings):
        raise AssertionError(f"no warning about the missing guard: {spec.warnings}")


@test
def a_broken_guard_file_is_refused_clearly():
    bad = os.path.join(WORK, "bad-guard.json")
    with open(bad, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    raises(lambda: spec_for([{"start": 0.2, "end": 1.0}], guard=bad),
           "not JSON", "malformed guard")
    raises(lambda: spec_for([{"start": 0.2, "end": 1.0}],
                            guard=os.path.join(WORK, "absent.json")),
           "cannot read guard", "missing guard file")


# ---------------------------------------------------------------- end to end

@test
def a_two_clip_reel_renders_at_the_size_and_length_asked_for():
    out = os.path.join(WORK, "reel.mp4")
    spec = spec_for([
        {"start": 0.2, "end": 1.1, "grade": "warm", "motion": "zoom-in"},
        {"start": 2.8, "end": 3.7, "grade": "vivid", "transition": "flash",
         "motion": "pan-left", "label": "the fast one"},
    ], output=out, crossfade=0.3)
    result = br.render(spec)

    eq(result["status"], "success", "render status")
    eq(result["output_size"], "180x320", "output size")
    eq(result["clips"][1]["label"], "the fast one", "label echoed")
    eq(result["clips"][0]["zoom"], [1.0, 1.1], "resolved zoom pair in the report")
    close(result["duration"], 1.8, 0.05, "reel length")

    from moviepy import VideoFileClip
    made = VideoFileClip(out)
    try:
        eq(list(made.size), [180, 320], "encoded size")
        close(made.duration, 1.8, 0.1, "encoded length")
        darkest = min(float(made.get_frame(t).mean())
                      for t in np.arange(0.05, made.duration - 0.05, 0.1))
        if darkest < 8:
            raise AssertionError(
                f"a frame came back nearly black (mean {darkest:.1f}) — the "
                "usual cause is a range straddling a dip in the source")
    finally:
        made.close()


@test
def audio_follows_the_same_cuts_as_the_picture():
    out = os.path.join(WORK, "reel-audio.mp4")
    spec = spec_for([
        {"start": 0.2, "end": 1.1},
        {"start": 2.8, "end": 3.7, "transition": "cut"},
    ], output=out, crossfade=0.3, audio=True)
    result = br.render(spec)
    eq(result["audio"], True, "audio reported in the result")

    from moviepy import VideoFileClip
    made = VideoFileClip(out)
    try:
        if made.audio is None:
            raise AssertionError("the encoded file has no audio track")
        close(made.audio.duration, result["duration"], 0.15,
              "audio length against picture length")
    finally:
        made.close()

    # Measured by decoding the file, not through moviepy's to_soundarray().
    # That reader returns near-silence for a track ffmpeg decodes at full
    # level, so a test built on it fails on working audio.
    level = float(np.abs(decode_audio(out)).mean())
    if level < 0.05:
        raise AssertionError(f"the audio track is silent (mean |x| {level:.4f})")

    quiet = float(np.abs(decode_audio(SOURCE)).mean())
    if level < quiet * 0.5:
        raise AssertionError(
            f"the reel is much quieter than its source ({level:.3f} against "
            f"{quiet:.3f}) — the cuts are landing on silence")


@test
def regression_the_first_clips_audio_cuts_in_with_its_picture():
    """
    "Clip 0 never dissolves in" lived only in the timeline, so every reel's
    sound used to ramp up over 0.35s under a picture already at full strength.
    """
    out = os.path.join(WORK, "reel-audio-head.mp4")
    spec = spec_for([{"start": 0.2, "end": 1.1},
                     {"start": 2.8, "end": 3.7, "transition": "cut"}],
                    output=out, crossfade=0.3, audio=True)
    br.render(spec)

    samples = decode_audio(out, rate=8000)
    head = float(np.abs(samples[:400]).mean())        # first 50 ms
    body = float(np.abs(samples[2000:4000]).mean())   # 0.25s - 0.5s
    if head < body * 0.4:
        raise AssertionError(
            f"the reel opens on a fade: first 50ms at {head:.3f} against "
            f"{body:.3f} in the body")


@test
def speed_changes_both_the_picture_and_the_sound():
    out = os.path.join(WORK, "reel-speed.mp4")
    spec = spec_for([{"start": 0.2, "end": 1.2, "speed": 2.0}],
                    output=out, crossfade=0.0, audio=True)
    result = br.render(spec)
    close(result["duration"], 0.5, 0.05, "a 1.0s range at 2x runs half as long")
    eq(result["clips"][0]["speed"], 2.0, "speed echoed in the report")

    samples = decode_audio(out)
    close(len(samples) / 8000.0, 0.5, 0.15, "audio length after the speed change")


@test
def a_still_can_open_a_reel_of_footage():
    out = os.path.join(WORK, "reel-still.mp4")
    spec = spec_for([
        {"image": STILL, "duration": 0.8, "grade": "vintage", "fit": True,
         "motion": "zoom-in", "zoom": [1.0, 1.15]},
        {"start": 1.8, "end": 2.5, "grade": "vivid", "transition": "flash"},
    ], output=out, crossfade=0.3)
    result = br.render(spec)
    eq(result["clip_count"], 2, "clip count")
    eq(result["clips"][0]["grade"], "vintage", "grade name in the report")
    close(result["duration"], 1.5, 0.05, "reel length")


@test
def fit_works_on_footage_not_just_stills():
    out = os.path.join(WORK, "reel-fit.mp4")
    spec = spec_for([{"start": 0.2, "end": 1.0, "fit": True, "grade": "faded"}],
                    output=out, crossfade=0.0)
    result = br.render(spec)
    eq(result["clips"][0]["fit"], True, "fit echoed in the report")

    from moviepy import VideoFileClip
    made = VideoFileClip(out)
    try:
        frame = made.get_frame(0.4)
        # A letterboxed 16:9 source in a 180x320 frame leaves a dim bed above
        # and below the picture.
        rows = frame.mean(axis=(1, 2))
        if rows[:20].mean() >= rows[140:180].mean():
            raise AssertionError("no darker bed above the letterboxed picture")
    finally:
        made.close()


# ---------------------------------------------------------------------- CLI

@test
def the_ranges_shorthand_builds_a_spec():
    eq(br.ranges_to_clips("0:08-0:13,1:20-1:25"),
       [{"start": "0:08", "end": "0:13"}, {"start": "1:20", "end": "1:25"}],
       "two ranges")
    raises(lambda: br.ranges_to_clips("0:08"), "needs a start and an end",
           "range with no dash")
    raises(lambda: br.ranges_to_clips(" , "), "empty", "blank ranges")


def run_cli(*argv):
    """Run main() with a fake argv and return (exit code, parsed stdout)."""
    saved = sys.argv
    sys.argv = ["build_reel.py", *argv]
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            code = br.main()
    finally:
        sys.argv = saved
    text = buffer.getvalue()
    try:
        return code, json.loads(text)
    except json.JSONDecodeError:
        return code, text


@test
def the_ranges_shorthand_renders_end_to_end():
    out = os.path.join(WORK, "reel-cli.mp4")
    code, payload = run_cli("-i", SOURCE, "-o", out, "--ranges", "0:00.2-0:01.0",
                            "--size", "120x160", "--fps", "24", "--crf", "30",
                            "--preset", "ultrafast", "--threads", "1")
    eq(code, 0, "exit code")
    eq(payload["status"], "success", "status")
    eq(payload["output_size"], "120x160", "size from the flag")
    eq(payload["crf"], 30, "crf from the flag")
    eq(payload["preset"], "ultrafast", "preset from the flag")


@test
def grades_lists_the_presets():
    code, payload = run_cli("--grades")
    eq(code, 0, "exit code")
    eq(sorted(payload), sorted(rg.PRESETS), "preset names")
    close(payload["vintage"]["saturation"], 0.32, 1e-9, "a preset value")


@test
def the_cli_reports_failure_as_json_and_a_non_zero_code():
    code, payload = run_cli("--spec", os.path.join(WORK, "does-not-exist.json"))
    eq(code, 1, "exit code on failure")
    eq(payload["status"], "error", "status field")
    if not payload.get("message"):
        raise AssertionError("an error result carried no message")
    if "REEL_TRACEBACK" not in payload.get("hint", ""):
        raise AssertionError("the error did not point at the traceback switch")

    code, payload = run_cli()
    eq(code, 1, "exit code with no arguments")
    if "--spec" not in payload["message"]:
        raise AssertionError(f"unhelpful message: {payload['message']!r}")


@test
def reel_traceback_adds_a_stack_trace():
    os.environ["REEL_TRACEBACK"] = "1"
    try:
        code, payload = run_cli("--spec", os.path.join(WORK, "does-not-exist.json"))
    finally:
        del os.environ["REEL_TRACEBACK"]
    eq(code, 1, "exit code")
    if "Traceback" not in payload.get("traceback", ""):
        raise AssertionError("REEL_TRACEBACK did not produce a stack trace")


# --------------------------------------------------------------------- main

def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else ""
    chosen = [f for f in _tests if wanted in f.__name__]
    if not chosen:
        print(f"no test matches {wanted!r}")
        return 1

    print("building fixtures...", flush=True)
    build_fixtures()

    passed = failed = 0
    for fn in chosen:
        name = fn.__name__.replace("_", " ")
        try:
            fn()
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}")
            print("        " + traceback.format_exc().strip().replace("\n", "\n        "))
        else:
            passed += 1
            print(f"  ok    {name}")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
