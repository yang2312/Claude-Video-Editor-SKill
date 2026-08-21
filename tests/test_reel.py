#!/usr/bin/env python3
"""
Tests for the cinematic-reel renderer.

    python tests/test_reel.py

No pytest, no fixtures directory, no network. A synthetic source video is
generated on first run and cached, so the suite runs anywhere the renderer
runs and never depends on someone's footage.

What is worth testing here is narrow and specific:

  * the grade produces the same picture as the eight-step chain it replaces,
    because that collapse is the whole performance argument and a silent
    drift in it would change every delivery;
  * the timeline places shots where the transition says, because an off-by-a-
    crossfade is invisible in a still and obvious in motion;
  * framing honours the move it was asked for, because a move that silently
    does nothing is the bug this suite was started over;
  * the spec rejects what it cannot render, with a message naming the clip.
"""

from __future__ import annotations

import os
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

WORK = os.path.join(tempfile.gettempdir(), "cinematic-reel-tests")
SOURCE = os.path.join(WORK, "source.mp4")
STILL = os.path.join(WORK, "still.png")


# ------------------------------------------------------------------ harness

_results: list = []


def test(fn):
    _results.append(fn)
    return fn


def eq(actual, expected, what: str):
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected!r}, got {actual!r}")


def close(actual, expected, tol, what: str):
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"{what}: expected {expected} +/- {tol}, got {actual}")


def raises(fn, fragment: str, what: str):
    try:
        fn()
    except Exception as error:  # noqa: BLE001 — that is the point
        if fragment.lower() not in str(error).lower():
            raise AssertionError(
                f"{what}: expected a message containing {fragment!r}, "
                f"got {str(error)!r}") from None
        return
    raise AssertionError(f"{what}: expected an error, none was raised")


# ------------------------------------------------------------------ fixtures

def build_fixtures() -> None:
    """A short source with moving picture and a tone, plus a still."""
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
        img[:, :, 2] *= 0.55
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
    import subprocess

    import imageio_ffmpeg

    raw = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path, "-vn",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(rate), "-ac", "1", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def make_args(**overrides):
    args = types.SimpleNamespace(
        input=None, output=None, hold=3.0, size=None, aspect=None, fps=24,
        crossfade=0.3, crf=28, preset="ultrafast", threads=1, codec="libx264",
        resample="quality")
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# -------------------------------------------------------------- timestamps

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
def geometry_parses_and_rounds_to_even():
    eq(br.parse_size("1081x1921"), (1080, 1920), "odd sizes rounded down")
    close(br.parse_aspect("9:16"), 0.5625, 1e-9, "aspect ratio")
    raises(lambda: br.parse_aspect("9"), "9:16", "aspect without a colon")


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


@test
def grade_matches_the_chain_it_replaces():
    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, (120, 90, 3), dtype=np.uint8)

    for name in ("vintage", "faded", "warm", "vivid"):
        from dataclasses import replace
        look = replace(rg.PRESETS[name], grain=0.0)   # grain is a fresh draw
        grade = rg.Grade.compile(look, 90, 120)
        mask = grade._vignette[:, :, :1] if grade._vignette is not None else None

        want = reference_grade(look, frame, mask, None).astype(np.int16)
        got = grade.apply(frame, 0).astype(np.int16)
        worst = int(np.abs(want - got).max())
        if worst > 2:
            raise AssertionError(
                f"{name}: compiled grade differs from the reference chain by "
                f"{worst} levels (2 is the quantisation budget)")


@test
def grade_leaves_highlights_for_the_vignette_to_roll_off():
    """
    Clipping before the vignette would flatten the corners of a bright frame.

    A contrast above 1 pushes highlights past white on purpose; the vignette
    multiplying them back down is what rolls them off. This was a real
    regression during the rewrite, so it gets a test of its own.
    """
    look = rg.Look(contrast=1.3, vignette=0.6)
    grade = rg.Grade.compile(look, 64, 64)
    white = np.full((64, 64, 3), 255, np.uint8)
    corner = int(grade.apply(white, 0)[2, 2, 0])
    if corner < 200:
        raise AssertionError(
            f"corner clipped to {corner}; the overshoot was thrown away "
            "before the vignette could use it")


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


# ----------------------------------------------------------------- timeline

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
    shots = dummy_shots([2.0, 2.0, 2.0])
    line = rt.Timeline(shots, ["cut", "crossfade", "cut"], 0.5, 30)
    starts = [round(p[0], 3) for p in line.placements]
    eq(starts, [0.0, 1.5, 3.5], "shot start times")
    close(line.duration, 5.5, 1e-6, "total length")


@test
def a_flash_is_recorded_at_the_junction_it_belongs_to():
    shots = dummy_shots([1.0, 1.0])
    line = rt.Timeline(shots, ["cut", "flash"], 0.4, 30)
    eq([round(f, 3) for f in line.flashes], [1.0], "flash times")

    quiet = float(line.frame(0.5).mean())
    peak = float(line.frame(1.0).mean())
    if peak <= quiet + 40:
        raise AssertionError(
            f"flash did not bloom: {quiet:.0f} before, {peak:.0f} at the junction")
    close(float(line.frame(0.2).mean()), 40.0, 0.5,
          "frames away from the flash are untouched")


@test
def the_first_clip_never_dissolves_in_from_nothing():
    shots = dummy_shots([1.0, 1.0])
    line = rt.Timeline(shots, ["crossfade", "crossfade"], 0.3, 30)
    close(line.placements[0][0], 0.0, 1e-9, "first shot starts at zero")


@test
def a_dissolve_blends_both_shots_at_its_midpoint():
    shots = dummy_shots([1.0, 1.0])       # values 40 and 100
    line = rt.Timeline(shots, ["cut", "crossfade"], 0.4, 30)
    middle = float(line.frame(0.8).mean())   # halfway through the overlap
    close(middle, 70.0, 3.0, "midpoint of a 40 -> 100 dissolve")


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
    moving = rt.CropFraming(still.size, 120, 200, "pan-right", 1.2, 0.6, 0.5, 0.0)
    first = moving.apply(still.frame(0), 0.0)
    last = moving.apply(still.frame(0), 1.0)
    if np.array_equal(first, last):
        raise AssertionError("pan-right produced identical first and last frames")
    eq(moving.static, False, "a pan is not static")

    held = rt.CropFraming(still.size, 120, 200, "none", 1.0, 0.0, 0.5, 0.0)
    eq(held.static, True, "motion 'none' is static")


@test
def fit_framing_honours_a_zoom():
    """
    The regression this suite exists for.

    Fit mode used to ignore motion, so a spec asking for a push on a
    letterboxed photo rendered a still frame and reported success.
    """
    still = rt.StillSource(STILL)
    framing = rt.FitFraming(still.size, 120, 260, 0.0, still.frame(0),
                            "quality", "zoom-in", [1.0, 1.4], 0.3, 0.5)
    eq(framing.static, False, "a zoom is not static")

    def band_height(progress):
        arr = rt.FitFraming.apply(framing, still.frame(0), progress)
        rows = arr.mean(axis=(1, 2))
        lit = np.flatnonzero(rows > rows.max() * 0.55)
        return int(lit.max() - lit.min())

    start, end = band_height(0.0), band_height(1.0)
    if end <= start:
        raise AssertionError(
            f"fit zoom did not grow the picture: {start}px -> {end}px")


@test
def fit_framing_without_a_move_is_static():
    still = rt.StillSource(STILL)
    framing = rt.FitFraming(still.size, 120, 260, 0.0, still.frame(0))
    eq(framing.static, True, "no motion means static")
    if not np.array_equal(framing.apply(still.frame(0), 0.0),
                          framing.apply(still.frame(0), 1.0)):
        raise AssertionError("a static fit framing produced two different frames")


@test
def a_held_still_is_framed_once_and_reused():
    still = rt.StillSource(STILL)
    framing = rt.FitFraming(still.size, 120, 260, 0.0, still.frame(0))
    shot = rt.Shot(still, framing, None, 2.0)
    shot.frame(0.0, 0)
    first = shot._frozen
    if first is None:
        raise AssertionError("a static still was not frozen after its first frame")
    shot.frame(1.0, 1)
    if shot._frozen is not first:
        raise AssertionError("the frozen frame was rebuilt instead of reused")


@test
def footage_is_never_frozen():
    source = rt.ClipSource(SOURCE, 0.0, 1.0, 1.0, (640, 360), 24.0)
    framing = rt.CropFraming(source.size, 120, 200, "none", 1.0, 0.0, 0.5, 0.0)
    shot = rt.Shot(source, framing, None, 1.0)
    eq(shot._freezable, False, "moving footage must not be frozen")
    shot.release()


# --------------------------------------------------------------------- spec

def spec_for(clips, **top):
    raw = {"source": SOURCE, "output": os.path.join(WORK, "out.mp4"),
           "size": "180x320", "fps": 24, "clips": clips}
    raw.update(top)
    return br.ReelSpec.load(raw, make_args())


@test
def a_clip_past_the_end_of_the_source_is_named():
    raises(lambda: spec_for([{"start": 3.0, "end": 9.0}]),
           "past the end", "range beyond the source")
    raises(lambda: spec_for([{"start": 2.0, "end": 1.0}]),
           "before it starts", "backwards range")
    raises(lambda: spec_for([{"end": 1.0}]),
           "clips[0]", "clip with no start names its index")


@test
def a_crossfade_longer_than_a_clip_is_refused():
    raises(lambda: spec_for([{"start": 0.0, "end": 1.0},
                             {"start": 1.0, "end": 1.4,
                              "transition": "crossfade"}], crossfade=0.6),
           "never fully appears", "crossfade wider than the clip it dissolves")

    # The same clip length is fine on a hard cut, which has no overlap.
    spec_for([{"start": 0.0, "end": 1.0},
              {"start": 1.0, "end": 1.4, "transition": "cut"}], crossfade=0.6)


@test
def unknown_motions_and_transitions_list_the_valid_ones():
    raises(lambda: spec_for([{"start": 0.0, "end": 1.0, "motion": "dolly"}]),
           "zoom-in", "unknown motion")
    raises(lambda: spec_for([{"start": 0.0, "end": 1.0, "transition": "wipe"}]),
           "crossfade", "unknown transition")
    raises(lambda: spec_for([{"start": 0.0, "end": 1.0, "zoom": 0.5}]),
           ">= 1.0", "zoom below 1")


@test
def auto_motion_never_repeats_a_move_back_to_back():
    moves = [rt.auto_motion(i) for i in range(8)]
    for a, b in zip(moves, moves[1:]):
        if a == b:
            raise AssertionError(f"auto motion repeated {a!r} back to back")


# ---------------------------------------------------------------- end to end

@test
def a_two_clip_reel_renders_at_the_size_and_length_asked_for():
    out = os.path.join(WORK, "reel.mp4")
    spec = spec_for([
        {"start": 0.2, "end": 1.2, "grade": "warm", "motion": "zoom-in"},
        {"start": 2.0, "end": 3.0, "grade": "vivid", "transition": "flash",
         "motion": "pan-left"},
    ], output=out, crossfade=0.3)
    result = br.render(spec)

    eq(result["status"], "success", "render status")
    eq(result["output_size"], "180x320", "output size")
    close(result["duration"], 2.0, 0.05, "reel length")

    from moviepy import VideoFileClip
    made = VideoFileClip(out)
    try:
        eq(list(made.size), [180, 320], "encoded size")
        close(made.duration, 2.0, 0.1, "encoded length")
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
        {"start": 0.2, "end": 1.2},
        {"start": 2.0, "end": 3.0, "transition": "cut"},
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
def a_still_can_open_a_reel_of_footage():
    out = os.path.join(WORK, "reel-still.mp4")
    spec = spec_for([
        {"image": STILL, "duration": 0.8, "grade": "vintage", "fit": True,
         "motion": "zoom-in", "zoom": [1.0, 1.15]},
        {"start": 1.0, "end": 2.0, "grade": "vivid", "transition": "flash"},
    ], output=out, crossfade=0.3)
    result = br.render(spec)
    eq(result["clip_count"], 2, "clip count")
    eq(result["clips"][0]["grade"], "vintage", "grade name in the report")
    close(result["duration"], 1.8, 0.05, "reel length")


@test
def the_cli_reports_failure_as_json_and_a_non_zero_code():
    import io
    import contextlib
    import json

    argv = sys.argv
    sys.argv = ["build_reel.py", "--spec", os.path.join(WORK, "does-not-exist.json")]
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            code = br.main()
    finally:
        sys.argv = argv

    eq(code, 1, "exit code on failure")
    payload = json.loads(buffer.getvalue())
    eq(payload["status"], "error", "status field")
    if not payload.get("message"):
        raise AssertionError("an error result carried no message")


# --------------------------------------------------------------------- main

def main() -> int:
    print("building fixtures...", flush=True)
    build_fixtures()

    passed = failed = 0
    for fn in _results:
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
