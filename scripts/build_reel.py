#!/usr/bin/env python3
"""
Cinematic Reel — cut highlights out of a video and join them into one reel.

Reads a JSON spec, cuts the ranges it names out of the source, reframes each
to the delivery aspect with a camera move of its own, grades it, and joins
them with the transition each junction asks for.

    python build_reel.py --spec highlights.json

    python build_reel.py -i source.mp4 -o reel.mp4 \
        --ranges 0:08-0:13,1:20-1:25 --aspect 9:16

Spec — every key, with its default:

    source      (required)   the video to cut from
    output      (required)   where to write the reel
    aspect      "9:16"       delivery aspect; height comes from the source
    size        —            exact output, e.g. "1080x1920". Give one of
                             aspect or size, not both — they disagree about
                             resolution and silently picking one is how a reel
                             comes out at a third of its intended size.
    fps         30
    crossfade   0.35         dissolve length, in seconds
    hold        3.0          clip length when neither end nor duration is given
    audio       false        lay the same ranges onto an audio track
    crf         23           x264 quality: 18 near-lossless, 23, 28 small
    preset      "medium"     x264 effort: ultrafast .. veryslow
    threads     0            encoder threads; 0 uses every core
    codec       "libx264"
    audio_codec "aac"
    resample    "quality"    or "fast": BICUBIC when enlarging, ~6% quicker
    guard       —            path to scan_source --json output; every clip
                             range is checked against the dips and cuts it
                             found, so a bad range is refused before encoding
                             rather than found afterwards
    clips       (required)

Per clip:

    start/end       a range in the source; `duration` may replace `end`
    image           a still standing in for footage; needs its own `duration`
    anchor          0=left .. 1=right (default 0.5), or [x, y] for both axes
    motion          none | zoom-in | zoom-out | pan-left | pan-right | auto
    zoom            a number, or [start, end] to begin already tight.
                    On a zoom-in/zoom-out a number is where the move travels
                    to; on a pan or a held shot it is where the framing sits.
    pan             0..1, how much of the available slack a pan spends
    speed           <1 slows, >1 speeds up (default 1.0). Footage only — a
                    photograph has no time axis for it to act on.
    shutter         motion blur, in degrees of shutter angle. 0 off, 180 the
                    film convention, 360 fully open. Blurs this renderer's
                    own move, which is the part of the picture that has none.
    shutter_samples how many sub-frames the exposure is built from (default 3)
    stutter         hold the shot at this many frames a second — 8 or 12 gives
                    the choppy, posterised look. 0 leaves it smooth.
    spill           under an out-of-bounds transition, which block of picture
                    breaks the border: [top, bottom, left, right] as
                    fractions. Put it over something real.
    freeze          hold the last frame for this many extra seconds. A
                    closing card needs about three seconds to be read, and
                    the shot under it is rarely that long.
    fit             letterbox the whole frame on a blurred bed instead of
                    cropping — for shots whose meaning spans the full width
    grade           a preset name, or an object of overrides
    ease            how a move spends its time: smooth (default) or impact,
                    which puts almost all the travel in the first fifth
    transition      how it joins the clip before it:
                      crossfade  dissolve; reads as time passing
                      cut        hard join; the only way to feel fast
                      flash      hard join under a white bloom
                      invert     two frames of inverted colour
                      invert-r   the same on one channel, so the fault has
                      invert-g   a colour; alternate them across a burst
                      invert-b
                      shake      decaying jitter with a brightness pop
                      shutter-shake  the same with the shutter open, so each
                                 jolt smears instead of stepping
                      film-roll  the strip yanked through the gate; belongs
                                 between sections, not between shots
                      out-of-bounds  the picture drops into a bordered frame
                                 and one block breaks out of it
    label           free text, echoed in the report

Keys starting with `_` are ignored, so a spec can carry comments. Anything
else unrecognised is an error: a spec is easier to debug when a typo is
refused than when it is absorbed.

Where settings can come from more than one place, an explicitly typed CLI
flag wins, then the spec file, then the default.

Timestamps accept SS, MM:SS, HH:MM:SS, or a plain number of seconds.

Pick ranges with scan_source.py first, and pass its output as `guard`.
Eyeballing two frames will not tell you that a range straddles a dip to
black, and one that does plays as a glitch.

Requires moviepy, which bundles its own ffmpeg, so this does not need ffmpeg
on PATH. Set REEL_TRACEBACK=1 to get a stack trace in the error JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reel_grade import (Grade, Look, PRESETS, release_all,  # noqa: E402
                        resolve_look)
from reel_timeline import (DEFAULT_SPILL, MARK_SECONDS, MOTIONS,  # noqa: E402
                           RESAMPLE_MODES, STRADDLE_SECONDS, TRANSITIONS,
                           CameraMove, ClipSource, Shot, StillSource, Timeline,
                           auto_motion, build_framing, plan)

# x264 effort levels, cheapest first. Named here so a typo is caught at parse
# time rather than surfacing as an opaque ffmpeg failure ten minutes in.
X264_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium", "slow", "slower", "veryslow", "placebo")


# ------------------------------------------------------------------ parsing

def parse_timestamp(value, where: str = "") -> float:
    """Parse SS, MM:SS, HH:MM:SS or a raw number into seconds."""
    prefix = f"{where} " if where else ""
    if isinstance(value, bool):
        raise ValueError(f"{prefix}expected a time, got {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"{prefix}negative timestamp: {value}")
        return seconds

    text = str(value).strip()
    if not text:
        raise ValueError(f"{prefix}empty timestamp")

    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"{prefix}too many ':' groups in timestamp: {text!r}")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"{prefix}non-numeric timestamp: {text!r}") from None
    if any(n < 0 for n in nums):
        raise ValueError(f"{prefix}negative component in timestamp: {text!r}")

    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


def format_timestamp(seconds: float) -> str:
    """Render seconds back as MM:SS.mmm or HH:MM:SS.mmm."""
    total = int(seconds)
    ms = int(round((seconds - total) * 1000))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}" if h else f"{m:02d}:{s:02d}.{ms:03d}"


def parse_aspect(text: str) -> float:
    parts = str(text).replace(":", "/").split("/")
    if len(parts) != 2:
        raise ValueError(f"aspect must look like 9:16, got {text!r}")
    try:
        w, h = (float(p) for p in parts)
    except ValueError:
        raise ValueError(f"aspect must look like 9:16, got {text!r}") from None
    if w <= 0 or h <= 0:
        raise ValueError(f"aspect components must be positive, got {text!r}")
    return w / h


def parse_size(text: str) -> tuple:
    parts = str(text).lower().replace("x", " ").split()
    if len(parts) != 2:
        raise ValueError(f"size must look like 1080x1920, got {text!r}")
    try:
        w, h = (int(float(p)) for p in parts)
    except ValueError:
        raise ValueError(f"size must look like 1080x1920, got {text!r}") from None
    if w <= 0 or h <= 0:
        raise ValueError(f"size components must be positive, got {text!r}")
    return w - w % 2, h - h % 2


def reject_unknown(entry: dict, allowed, where: str) -> None:
    """
    Refuse keys nobody reads.

    A misspelled key used to be absorbed in silence — `"durationn": 2` quietly
    became a clip at the default hold. Keys starting with `_` are left alone
    so a spec can carry comments.
    """
    unknown = [k for k in entry if not k.startswith("_") and k not in allowed]
    if unknown:
        raise ValueError(
            f"{where} has unknown key{'s' if len(unknown) > 1 else ''} "
            f"{', '.join(repr(k) for k in sorted(unknown))}; expected one of "
            + ", ".join(sorted(allowed)))


# -------------------------------------------------------------------- specs

CLIP_KEYS = frozenset({
    "start", "end", "duration", "image", "anchor", "speed", "motion", "zoom",
    "pan", "ease", "fit", "grade", "transition", "label", "shutter",
    "shutter_samples", "stutter", "spill", "freeze",
})

REEL_KEYS = frozenset({
    "source", "output", "aspect", "size", "fps", "crossfade", "hold", "audio",
    "crf", "preset", "threads", "codec", "audio_codec", "resample", "guard",
    "clips",
})


@dataclass
class ClipSpec:
    """One entry of the spec, parsed and checked."""

    start: float = 0.0
    end: float = 0.0
    image: Optional[str] = None
    speed: float = 1.0
    move: CameraMove = field(default_factory=CameraMove)
    shutter: float = 0.0
    shutter_samples: int = 3
    stutter: float = 0.0
    spill: tuple = DEFAULT_SPILL
    freeze: float = 0.0
    fit: bool = False
    look: Look = field(default_factory=Look)
    grade_name: str = "none"
    transition: str = "crossfade"
    label: str = ""

    @property
    def length(self) -> float:
        """How long this clip runs on the timeline, after speed and freeze."""
        if self.image:
            return self.end + self.freeze   # a still has no time axis to speed
        return (self.end - self.start) / self.speed + self.freeze

    @classmethod
    def parse(cls, entry: dict, index: int, default_hold: float,
              source_duration: float) -> "ClipSpec":
        if not isinstance(entry, dict):
            raise ValueError(f"clips[{index}] must be an object")

        where = f"clips[{index}]"
        reject_unknown(entry, CLIP_KEYS, where)

        image = entry.get("image")
        speed = float(entry.get("speed", 1.0))
        if speed <= 0:
            raise ValueError(f"{where} speed must be positive, got {speed}")

        if image:
            if not os.path.exists(image):
                raise ValueError(f"{where} image not found: {image}")
            if "speed" in entry:
                raise ValueError(
                    f"{where} sets speed on an image. A photograph has no time "
                    f"axis — set its `duration` instead.")
            if "start" in entry or "end" in entry:
                raise ValueError(
                    f"{where} sets both `image` and a source range; an image "
                    f"clip is timed by `duration` alone.")
            start = 0.0
            end = float(entry.get("duration", default_hold))
            if end <= 0:
                raise ValueError(f"{where} duration must be positive, got {end}")
        else:
            if "start" not in entry:
                raise ValueError(f"{where} has neither 'start' nor 'image'")
            start = parse_timestamp(entry["start"], f"{where} start")
            if "end" in entry:
                end = parse_timestamp(entry["end"], f"{where} end")
            elif "duration" in entry:
                end = start + parse_timestamp(entry["duration"], f"{where} duration")
            else:
                end = start + default_hold
            if end <= start:
                raise ValueError(
                    f"{where} ends at or before it starts "
                    f"({format_timestamp(start)} -> {format_timestamp(end)})")
            if end > source_duration:
                raise ValueError(
                    f"{where} runs to {format_timestamp(end)}, past the end of "
                    f"the source ({source_duration:.2f}s)")

        motion = entry.get("motion", "none")
        if motion == "auto":
            motion = auto_motion(index)
        move = CameraMove.resolve(motion=motion, zoom=entry.get("zoom"),
                                  pan=float(entry.get("pan", 0.30)),
                                  anchor=entry.get("anchor", 0.5),
                                  curve=entry.get("ease", "smooth"), where=where)

        fit = bool(entry.get("fit", False))

        transition = entry.get("transition", "crossfade")
        if transition not in TRANSITIONS:
            raise ValueError(
                f"{where} has unknown transition {transition!r}; expected one "
                "of " + ", ".join(TRANSITIONS))
        if index == 0 and (transition == "flash" or transition in MARK_SECONDS
                           or transition in STRADDLE_SECONDS):
            raise ValueError(
                f"clips[0] asks for {transition!r}, which marks a junction — "
                f"and the first clip has nothing before it.")

        shutter = float(entry.get("shutter", 0.0))
        if not 0.0 <= shutter <= 360.0:
            raise ValueError(
                f"{where} shutter is an angle in degrees, 0 to 360; got {shutter}")
        shutter_samples = int(entry.get("shutter_samples", 3))
        if shutter_samples < 2:
            raise ValueError(
                f"{where} shutter_samples must be at least 2, got {shutter_samples}")

        stutter = float(entry.get("stutter", 0.0))
        if stutter < 0:
            raise ValueError(f"{where} stutter cannot be negative, got {stutter}")

        spill = cls._resolve_spill(entry.get("spill"), transition, where)

        freeze = float(entry.get("freeze", 0.0))
        if freeze < 0:
            raise ValueError(f"{where} freeze cannot be negative, got {freeze}")

        look, grade_name = resolve_look(entry.get("grade"))

        return cls(start=start, end=end, image=image, speed=speed, move=move,
                   shutter=shutter, shutter_samples=shutter_samples,
                   stutter=stutter, spill=spill, freeze=freeze, fit=fit,
                   look=look, grade_name=grade_name, transition=transition,
                   label=entry.get("label", ""))

    @staticmethod
    def _resolve_spill(value, transition: str, where: str) -> tuple:
        """
        Which block of picture breaks out of an out-of-bounds frame.

        Four fractions, top bottom left right. It is the one number in the
        effect that decides whether it reads: a block with a roofline or a
        tree crossing it reads as depth, and the same block over empty ground
        reads as a rectangle. Refusing it on any other transition keeps a
        spec from claiming an effect it did not ask for.
        """
        if value is None:
            return DEFAULT_SPILL
        if transition != "out-of-bounds":
            raise ValueError(
                f"{where} sets `spill`, which only means anything under an "
                f"out-of-bounds transition; this clip joins with "
                f"{transition!r}.")
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(
                f"{where} spill takes four fractions — top, bottom, left, "
                f"right — got {value!r}")

        top, bottom, left, right = (float(v) for v in value)
        for name, v in zip(("top", "bottom", "left", "right"),
                           (top, bottom, left, right)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(
                    f"{where} spill {name} must be between 0 and 1, got {v}")
        if top >= bottom or left >= right:
            raise ValueError(
                f"{where} spill has no area: top {top} to bottom {bottom}, "
                f"left {left} to right {right}")
        return top, bottom, left, right


@dataclass
class ReelSpec:
    """The whole job: what to cut, how to frame it, how to encode it."""

    source: str
    output: str
    clips: list
    size: tuple
    fps: int = 30
    crossfade: float = 0.35
    audio: bool = False
    crf: int = 23
    preset: str = "medium"
    threads: int = 0
    codec: str = "libx264"
    audio_codec: str = "aac"
    resample: str = "quality"
    guard: Optional[str] = None
    warnings: list = field(default_factory=list)
    source_size: tuple = (0, 0)
    source_fps: float = 30.0
    source_duration: float = 0.0

    @classmethod
    def load(cls, raw: dict, args) -> "ReelSpec":
        reject_unknown(raw, REEL_KEYS, "spec")

        def pick(key, fallback):
            """CLI when typed, then the spec file, then the default."""
            typed = getattr(args, key, None)
            if typed is not None:
                return typed
            return raw.get(key, fallback)

        source = raw.get("source") or getattr(args, "input", None)
        output = raw.get("output") or getattr(args, "output", None)
        if not source:
            raise ValueError("no source video — pass -i or set 'source' in the spec")
        if not os.path.exists(source):
            raise ValueError(f"source video not found: {source}")
        if not output:
            raise ValueError("no output path — pass -o or set 'output' in the spec")
        if not raw.get("clips"):
            raise ValueError("spec has no 'clips'")

        if raw.get("size") and raw.get("aspect"):
            raise ValueError(
                "spec sets both 'aspect' and 'size'. They resolve to different "
                "resolutions — from a 1080p source, aspect '9:16' gives 608x1080 "
                "while size gives whatever it says — so pick one.")

        from moviepy import VideoFileClip
        probe = VideoFileClip(source)
        try:
            src_size = (probe.size[0], probe.size[1])
            src_fps = float(probe.fps or 30.0)
            src_duration = float(probe.duration)
        finally:
            probe.close()

        crossfade = float(pick("crossfade", 0.35))
        if crossfade < 0:
            raise ValueError(f"crossfade cannot be negative, got {crossfade}")

        preset = str(pick("preset", "medium"))
        if preset not in X264_PRESETS:
            raise ValueError(
                f"unknown preset {preset!r}; expected one of "
                + ", ".join(X264_PRESETS))

        resample = str(pick("resample", "quality"))
        if resample not in RESAMPLE_MODES:
            raise ValueError(
                f"unknown resample mode {resample!r}; expected one of "
                + ", ".join(RESAMPLE_MODES))

        hold = float(pick("hold", 3.0))
        clips = [ClipSpec.parse(e, i, hold, src_duration)
                 for i, e in enumerate(raw["clips"])]

        size_text, aspect_text = pick("size", None), pick("aspect", None)
        if size_text:
            size = parse_size(size_text)
        elif aspect_text:
            h = src_size[1] - src_size[1] % 2
            w = int(round(h * parse_aspect(aspect_text)))
            size = (w - w % 2, h)
        else:
            size = (src_size[0] - src_size[0] % 2, src_size[1] - src_size[1] % 2)

        spec = cls(
            source=source, output=output, clips=clips, size=size,
            fps=int(pick("fps", 30)), crossfade=crossfade,
            audio=bool(pick("audio", False)), crf=int(pick("crf", 23)),
            preset=preset, threads=int(pick("threads", 0)),
            codec=str(pick("codec", "libx264")),
            audio_codec=str(raw.get("audio_codec", "aac")),
            resample=resample, guard=raw.get("guard"),
            source_size=src_size, source_fps=src_fps, source_duration=src_duration,
        )

        # Planning is validating: it is the one implementation of the
        # placement rule and of what that rule cannot express.
        plan([c.length for c in spec.clips],
             [c.transition for c in spec.clips], spec.crossfade)

        spec.warnings = spec._collect_warnings(raw)
        return spec

    # -- advisory --------------------------------------------------------

    def _collect_warnings(self, raw: dict) -> list:
        """Things worth saying that are not worth refusing over."""
        notes = []

        for i, clip in enumerate(self.clips):
            if clip.shutter > 0 and not (clip.move.zooms or clip.move.pans):
                notes.append(
                    f"clips[{i}] sets a shutter but has no move for it to blur "
                    f"— the footage's own motion is already exposed")

        moves = [c.move.motion for c in self.clips]
        for i, (a, b) in enumerate(zip(moves, moves[1:]), start=1):
            if a == b and a != "none":
                notes.append(
                    f"clips[{i - 1}] and clips[{i}] both use {a!r} — two of the "
                    f"same move in a row reads as mechanical")

        if self.guard:
            notes.extend(self._guard_notes())
        elif not all(c.image for c in self.clips):
            notes.append(
                "no 'guard' set — clip ranges were not checked against the "
                "source's dips and cuts. Run scan_source.py --json and point "
                "'guard' at the result.")
        return notes

    def _guard_notes(self) -> list:
        """
        Check every clip range against a saved scan.

        This is the step that used to be a human retyping numbers into a
        `label` field. A range straddling a dip to black plays as a glitch,
        and until now nothing could catch it before the encode finished.
        """
        try:
            with open(self.guard, encoding="utf-8") as handle:
                report = json.load(handle)
        except OSError as error:
            raise ValueError(f"cannot read guard file {self.guard!r}: {error}") from None
        except json.JSONDecodeError as error:
            raise ValueError(f"guard file {self.guard!r} is not JSON: {error}") from None

        boundaries = [(t, kind)
                      for kind, key in (("dip", "dips"), ("cut", "hard_cuts"))
                      for t in report.get(key, [])]
        if not boundaries:
            return []

        notes = []
        for i, clip in enumerate(self.clips):
            if clip.image:
                continue
            hits = sorted(t for t, _ in boundaries if clip.start <= t <= clip.end)
            if hits:
                kinds = {kind for t, kind in boundaries if t in hits}
                raise ValueError(
                    f"clips[{i}] ({format_timestamp(clip.start)} -> "
                    f"{format_timestamp(clip.end)}) straddles "
                    f"{len(hits)} {'/'.join(sorted(kinds))} boundar"
                    f"{'ies' if len(hits) > 1 else 'y'} in the source, at "
                    f"{', '.join(f'{t:.2f}s' for t in hits[:5])}"
                    f"{' ...' if len(hits) > 5 else ''}. It will play as a "
                    f"glitch — pick a range inside one of the clean stretches "
                    f"scan_source.py reported.")
        return notes


# ------------------------------------------------------------------- render

def build_shots(spec: ReelSpec) -> list:
    """Turn every parsed clip into a Shot at the output geometry."""
    out_w, out_h = spec.size
    shots = []
    for clip in spec.clips:
        if clip.image:
            source = StillSource(clip.image)
            duration = clip.end
        else:
            source = ClipSource(spec.source, clip.start, clip.end, clip.speed,
                                spec.source_size, spec.source_fps)
            duration = source.duration

        framing = build_framing(source, clip.fit, clip.move, out_w, out_h,
                                clip.look.softness, spec.resample)

        # A pan with no room to travel renders a frozen shot and reports
        # success. This used to be checked for fit only, which missed the
        # case that bites hardest: delivering 16:9 from a 16:9 source, where
        # cropping has no free slack either and every pan in the spec is
        # silently dead.
        if clip.move.pans and framing.static:
            source.release()
            raise ValueError(
                f"clips[{len(shots)}] asks for {clip.move.motion}, but at "
                f"{out_w}x{out_h} this shot's crop already fills the frame and "
                f"the pan has nowhere to go. Add a zoom above 1.0 to make "
                f"room, or use a delivery aspect narrower than the source.")
        shots.append(Shot(source, framing, Grade.compile(clip.look, out_w, out_h),
                          duration, clip.label, clip.grade_name, fps=spec.fps,
                          shutter=clip.shutter,
                          shutter_samples=clip.shutter_samples,
                          stutter=clip.stutter, spill=clip.spill,
                          freeze=clip.freeze))
    return shots


def build_audio(spec: ReelSpec, timeline: Timeline):
    """
    Lay the same ranges out on an audio track, at the same times.

    Every clip gets a few milliseconds of fade at each end. Cutting audio at
    an arbitrary sample leaves a step in the waveform, and a step is a click.

    A clip that dissolves gets a fade as long as the dissolve, so the two
    tracks cross at the rate the pictures do. Clip 0 is the exception in both
    media: it cuts in, so its audio does too. That rule used to live only in
    the timeline, and every reel's sound ramped up under a picture that was
    already at full strength.
    """
    from moviepy import AudioFileClip, CompositeAudioClip, afx

    probe = AudioFileClip(spec.source)
    if probe.duration is None:
        probe.close()
        return None, None

    edge = min(0.02, spec.crossfade / 2.0 if spec.crossfade > 0 else 0.02)
    tracks = []
    for clip, placement in zip(spec.clips, timeline.placements):
        if clip.image:
            continue
        piece = probe.subclipped(clip.start, clip.end)
        if clip.speed != 1.0:
            # `with_speed_scaled`, not an effect from afx — moviepy has no
            # audio speed effect, and asking for one raised AttributeError
            # every time a spec combined `speed` with `audio`. Applied before
            # the fades so their lengths refer to the scaled clip.
            piece = piece.with_speed_scaled(clip.speed)
        dissolves = (placement.index > 0 and clip.transition == "crossfade"
                     and spec.crossfade > 0)
        piece = piece.with_effects([
            afx.AudioFadeIn(spec.crossfade if dissolves else edge),
            afx.AudioFadeOut(edge),
        ])
        tracks.append(piece.with_start(placement.start))

    if not tracks:
        probe.close()
        return None, None
    return CompositeAudioClip(tracks), probe


def render(spec: ReelSpec, progress: bool = False) -> dict:
    """Cut every range out of the source and write the reel."""
    from moviepy import VideoClip

    directory = os.path.dirname(spec.output)
    if directory:
        os.makedirs(directory, exist_ok=True)

    started = time.perf_counter()
    shots = build_shots(spec)
    timeline = Timeline(shots, [c.transition for c in spec.clips],
                        spec.crossfade, spec.fps)

    audio_track = audio_probe = None
    clip = VideoClip(timeline.frame, duration=timeline.duration)
    try:
        if spec.audio:
            audio_track, audio_probe = build_audio(spec, timeline)
            if audio_track is not None:
                clip.audio = audio_track

        # moviepy's default bitrate is wildly generous — a 15s vertical reel
        # lands near 40 MB. CRF lets x264 pick the rate from the picture
        # instead. Grain is expensive to encode, so a heavily graded reel
        # needs a higher CRF than clean footage for the same size.
        clip.write_videofile(
            spec.output, fps=spec.fps, codec=spec.codec,
            audio=audio_track is not None,
            # Named rather than inferred: left to itself moviepy picks mp3 for
            # an .mp4, which plays but is not what anything expects to find in
            # that container.
            audio_codec=spec.audio_codec,
            preset=spec.preset,
            threads=spec.threads or None,
            ffmpeg_params=["-crf", str(spec.crf), "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart"],
            logger="bar" if progress else None,
        )
    finally:
        timeline.release()
        for closeable in (audio_track, audio_probe, clip):
            if closeable is None:
                continue
            try:
                closeable.close()
            except Exception:
                pass
        # Frees the pooled scratch, the interned grades, and the vignette
        # masks — which are the largest allocation in the renderer and used to
        # survive every documented cleanup call.
        release_all()

    elapsed = time.perf_counter() - started
    frames = int(round(timeline.duration * spec.fps))
    result = {
        "status": "success",
        "output_path": spec.output,
        "duration": round(timeline.duration, 3),
        "clip_count": len(spec.clips),
        "fps": spec.fps,
        "output_size": f"{spec.size[0]}x{spec.size[1]}",
        "source_resolution": f"{spec.source_size[0]}x{spec.source_size[1]}",
        "crossfade": spec.crossfade,
        "audio": audio_track is not None,
        "crf": spec.crf,
        "preset": spec.preset,
        "resample": spec.resample,
        "guard": spec.guard,
        "size_mb": round(os.path.getsize(spec.output) / 1e6, 2),
        "render_seconds": round(elapsed, 1),
        "ms_per_frame": round(elapsed / max(frames, 1) * 1000, 1),
        "frames_built": sum(s.frames_built for s in shots),
        "clips": [
            {
                "source": (os.path.basename(c.image) if c.image else
                           f"{format_timestamp(c.start)} -> {format_timestamp(c.end)}"),
                "length": round(c.length, 2),
                "grade": c.grade_name,
                "motion": c.move.motion,
                # The resolved pair, not the shorthand — a scalar zoom means
                # different things under different motions, and the report is
                # where that should stop being a guess.
                "zoom": [round(z, 3) for z in c.move.zoom],
                "pan": c.move.pan,
                "anchor": list(c.move.anchor),
                "transition": c.transition,
                "speed": c.speed,
                "shutter": c.shutter,
                "stutter": c.stutter,
                "freeze": c.freeze,
                "fit": c.fit,
                "label": c.label,
            }
            for c in spec.clips
        ],
    }
    if spec.warnings:
        result["warnings"] = spec.warnings
    return result


# ---------------------------------------------------------------------- CLI

def ranges_to_clips(text: str) -> list:
    """Turn `0:08-0:13,1:20-1:25` into spec entries."""
    entries = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(
                f"range {chunk!r} needs a start and an end, e.g. 0:08-0:13")
        head, tail = chunk.rsplit("-", 1)
        entries.append({"start": head.strip(), "end": tail.strip()})
    if not entries:
        raise ValueError("--ranges was empty")
    return entries



# The one-line summary of every effect, keyed by the exact string that turns
# it on. It lives next to the code so `--effects` can never describe a knob
# that no longer exists; references/EFFECTS.md is the long form of the same
# list. A key here with no counterpart in the code is a bug, and the test
# suite checks for exactly that.
EFFECT_NOTES = {
    "zoom-in": "pushes in; the default forward move",
    "zoom-out": "pulls back; reveals context, belongs on a last clip",
    "pan-left": "slides left across the frame",
    "pan-right": "slides right; free slack at 9:16, needs zoom if aspects match",
    "none": "holds the frame still",

    "crossfade": "dissolve; reads as time passing, and costs a beat",
    "cut": "hard join; the only way to make a section feel fast",
    "flash": "white bloom over the join; 'somewhere else now'",
    "invert": "two frames of inverted colour; reads as a fault, not as light",
    "invert-r": "the same on red only, so the fault has a colour",
    "invert-g": "the same on green only",
    "invert-b": "the same on blue only",
    "shake": "decaying jitter with a brightness pop, about a third of a second",
    "shutter-shake": "the same with the shutter open, so each jolt smears",
    "film-roll": "the strip yanked through the gate, frame-lines and all",
    "out-of-bounds": "a bordered frame, with one block of picture breaking out",

    "speed": "below 1 slows, above 1 speeds up. Footage only",
    "shutter": "motion blur, 0-360 degrees of shutter angle. 180 is film",
    "shutter_samples": "sub-frames the exposure is built from (default 3)",
    "stutter": "holds the shot at 8 or 12 frames a second",
    "zoom": "how far a move travels: 1.10, or [1.0, 1.3] for both ends",
    "pan": "fraction of the available slack a pan crosses, 0-1",
    "anchor": "where the crop sits when nothing is moving",
    "ease": "how a move spends its time: smooth, or impact for a snap",
    "spill": "which block breaks an out-of-bounds border: [t, b, l, r]",
    "freeze": "hold the last frame this many extra seconds",
    "fit": "letterbox instead of cropping",

    "saturation": "1.0 untouched, 0.0 greyscale, above 1 richer",
    "temperature": "-1 cools toward blue, +1 warms toward amber",
    "contrast": "separation between dark and light, around 1.0",
    "lift": "raises the black point - the milky shadows of aged film",
    "gamma": "below 1 brightens midtones, above 1 deepens them",
    "black": "input level: everything below this is crushed to zero",
    "white": "input level: everything above this is blown out",
    "glow": "light spilling out of the highlights. Spatial, so it costs time",
    "glow_threshold": "where that spill starts, as a fraction of brightness",
    "rgb_split": "chromatic aberration, in pixels of red/blue separation",
    "vignette": "corner falloff, which pulls the eye to the middle",
    "grain": "luminance noise. 0.03 is film, 0.10-0.15 is short-form",
    "softness": "blend toward a blurred copy; old lenses were not sharp",
}


def describe_effects() -> str:
    """
    The whole vocabulary, grouped the way an editor reaches for it.

    Printed rather than returned as JSON because this exists to be read by
    someone who cannot remember what the skill can do.
    """
    from reel_grade import Look

    def rows(title, names, note=""):
        width = max(len(n) for n in names)
        out = [title, "-" * len(title)]
        if note:
            out.append(note)
        out += [f"  {n:<{width}}  {EFFECT_NOTES[n]}" for n in names]
        return "\n".join(out)

    knobs = [f.name for f in dataclasses.fields(Look)]
    blocks = [
        rows("Camera moves  (clip key: motion)", list(MOTIONS)),
        rows("Transitions   (clip key: transition)", list(TRANSITIONS),
             "  set on the incoming clip; clip 0 can only cut"),
        rows("Time          (clip keys)",
             ["speed", "shutter", "shutter_samples", "stutter", "freeze"]),
        rows("Framing       (clip keys)",
             ["zoom", "pan", "anchor", "ease", "fit", "spill"]),
        rows("Grade knobs   (clip key: grade)", knobs,
             "  free except glow and rgb_split, which are spatial"),
        "Grade presets\n-------------\n  " + ", ".join(sorted(PRESETS))
        + "\n  --grades prints them with their numbers",
        "Long form: references/EFFECTS.md",
    ]
    return "\n\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cut highlights out of a video and join them into one reel")
    parser.add_argument("--spec", help="JSON spec file (the full-control path)")
    parser.add_argument("-i", "--input", help="Source video")
    parser.add_argument("-o", "--output", help="Output reel")
    parser.add_argument("--ranges", help="Shorthand: 0:08-0:13,1:20-1:25")
    parser.add_argument("--guard", help="scan_source --json output to check ranges against")

    # Everything below defaults to None so that an explicitly typed flag can
    # be told apart from an untouched one. Without that, `--fps 60` lost to a
    # spec saying 30, silently.
    parser.add_argument("--aspect", help="Delivery aspect, e.g. 9:16")
    parser.add_argument("--size", help="Exact output size, e.g. 1080x1920")
    parser.add_argument("--fps", type=int, default=None, help="Output frame rate (default 30)")
    parser.add_argument("--hold", type=float, default=None,
                        help="Clip length when neither end nor duration is given (default 3.0)")
    parser.add_argument("--crossfade", type=float, default=None, help="Dissolve length (default 0.35)")
    parser.add_argument("--audio", action="store_true", default=None,
                        help="Carry the source audio through the same cuts")
    parser.add_argument("--crf", type=int, default=None,
                        help="x264 quality: 18 near-lossless, 23 default, 28 small")
    parser.add_argument("--preset", default=None, choices=X264_PRESETS,
                        help="x264 effort (default medium)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Encoder threads; 0 uses every core")
    parser.add_argument("--codec", default=None, help="Video codec (default libx264)")
    parser.add_argument("--resample", default=None, choices=RESAMPLE_MODES,
                        help="quality keeps LANCZOS everywhere; fast uses BICUBIC "
                             "when enlarging, about 6%% quicker")
    parser.add_argument("--progress", action="store_true",
                        help="Show an encoding progress bar on stderr")
    parser.add_argument("--grades", action="store_true",
                        help="List the grade presets and exit")
    parser.add_argument("--effects", action="store_true",
                        help="List every effect and exit")
    args = parser.parse_args()

    if args.effects:
        print(describe_effects())
        return 0

    if args.grades:
        print(json.dumps({name: vars(look) for name, look in PRESETS.items()},
                         indent=2))
        return 0

    try:
        if args.spec:
            with open(args.spec, encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError("spec file must contain a JSON object")
        elif args.ranges:
            raw = {"clips": ranges_to_clips(args.ranges)}
        else:
            raise ValueError("pass --spec, or -i/-o with --ranges")

        if args.guard:
            raw = dict(raw, guard=args.guard)

        spec = ReelSpec.load(raw, args)
        result = render(spec, progress=args.progress)
        result["message"] = (
            f"Built {result['clip_count']}-clip reel, {result['duration']}s at "
            f"{result['output_size']}, {result['size_mb']} MB in "
            f"{result['render_seconds']}s -> {spec.output}")
    except Exception as error:  # noqa: BLE001 — the CLI contract is JSON, always
        result = {"status": "error", "message": str(error),
                  "error_type": type(error).__name__}
        if os.environ.get("REEL_TRACEBACK"):
            import traceback
            result["traceback"] = traceback.format_exc()
        else:
            result["hint"] = "set REEL_TRACEBACK=1 for a stack trace"

    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
