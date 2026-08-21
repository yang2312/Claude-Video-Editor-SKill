#!/usr/bin/env python3
"""
Cinematic Reel — cut highlights out of a video and join them into one reel.

Reads a JSON spec, cuts the ranges it names out of the source, reframes each
to the delivery aspect with a camera move of its own, grades it, and joins
them with the transition each junction asks for.

    python build_reel.py --spec highlights.json

    python build_reel.py -i source.mp4 -o reel.mp4 \
        --ranges 0:08-0:13,1:20-1:25 --aspect 9:16

Spec:

    {
      "source": "source.mp4",
      "output": "reel.mp4",
      "aspect": "9:16",          // or "size": "1080x1920"
      "fps": 30,
      "crossfade": 0.35,
      "audio": false,
      "crf": 23,                 // 18 near-lossless, 23 default, 28 small
      "preset": "medium",        // x264 effort: ultrafast .. veryslow
      "threads": 0,              // 0 = every core; set lower to stay light
      "clips": [
        {"start": "0:08", "end": "0:13", "grade": "warm", "motion": "zoom-in"},
        {"image": "gate.png", "duration": 2.8, "grade": "vintage", "fit": true},
        {"start": "1:20", "duration": 3.0, "transition": "flash", "speed": 0.85}
      ]
    }

Per clip:

    start/end       a range in the source; `duration` may replace `end`
    image           a still standing in for footage; needs its own `duration`
    anchor          0=left .. 1=right, where the crop window sits (default 0.5)
    motion          none | zoom-in | zoom-out | pan-left | pan-right | auto
    zoom            a number, or [start, end] to begin already tight
    pan             0..1, how much of the available slack a pan spends
    speed           <1 slows, >1 speeds up (default 1.0)
    fit             letterbox the whole frame on a blurred bed instead of
                    cropping — for shots whose meaning spans the full width
    grade           a preset name, or an object of overrides
    transition      crossfade | cut | flash — how it joins the clip before it
    label           free text, echoed in the report

Timestamps accept SS, MM:SS, HH:MM:SS, or a plain number of seconds.

Pick ranges with scan_source.py first. Eyeballing two frames will not tell you
that a range straddles a dip to black, and one that does plays as a glitch.

Requires moviepy, which bundles its own ffmpeg, so this does not need ffmpeg
on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reel_grade import (Grade, Look, PRESETS, release_scratch,  # noqa: E402
                        resolve_look)
from reel_timeline import (MOTIONS, RESAMPLE_MODES, TRANSITIONS,  # noqa: E402
                           ClipSource, Shot, StillSource, Timeline,
                           auto_motion, build_framing, zoom_range)


# ------------------------------------------------------------------ parsing

def parse_timestamp(value) -> float:
    """Parse SS, MM:SS, HH:MM:SS or a raw number into seconds."""
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"negative timestamp: {value}")
        return seconds

    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")

    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"too many ':' groups in timestamp: {text!r}")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"non-numeric timestamp: {text!r}") from None
    if any(n < 0 for n in nums):
        raise ValueError(f"negative component in timestamp: {text!r}")

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
    w, h = (float(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError(f"aspect components must be positive, got {text!r}")
    return w / h


def parse_size(text: str) -> tuple:
    parts = str(text).lower().replace("x", " ").split()
    if len(parts) != 2:
        raise ValueError(f"size must look like 1080x1920, got {text!r}")
    w, h = (int(float(p)) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError(f"size components must be positive, got {text!r}")
    return w - w % 2, h - h % 2


# -------------------------------------------------------------------- specs

@dataclass
class ClipSpec:
    """One entry of the spec, parsed and checked."""

    start: float = 0.0
    end: float = 0.0
    image: Optional[str] = None
    anchor: float = 0.5
    speed: float = 1.0
    motion: str = "none"
    zoom: object = 1.10
    pan: float = 0.30
    fit: bool = False
    look: Look = field(default_factory=Look)
    grade_name: str = "none"
    transition: str = "crossfade"
    label: str = ""

    @property
    def length(self) -> float:
        """How long this clip runs on the timeline, after speed."""
        span = self.end if self.image else self.end - self.start
        return span / self.speed

    @classmethod
    def parse(cls, entry: dict, index: int, default_hold: float,
              source_duration: float) -> "ClipSpec":
        if not isinstance(entry, dict):
            raise ValueError(f"clips[{index}] must be an object")

        where = f"clips[{index}]"
        image = entry.get("image")
        if image:
            if not os.path.exists(image):
                raise ValueError(f"{where} image not found: {image}")
            start = 0.0
            end = float(entry.get("duration", default_hold))
            if end <= 0:
                raise ValueError(f"{where} duration must be positive, got {end}")
        else:
            if "start" not in entry:
                raise ValueError(f"{where} has neither 'start' nor 'image'")
            start = parse_timestamp(entry["start"])
            if "end" in entry:
                end = parse_timestamp(entry["end"])
            elif "duration" in entry:
                end = start + float(entry["duration"])
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

        anchor = float(entry.get("anchor", 0.5))
        if not 0.0 <= anchor <= 1.0:
            raise ValueError(f"{where} anchor must be between 0 and 1, got {anchor}")

        speed = float(entry.get("speed", 1.0))
        if speed <= 0:
            raise ValueError(f"{where} speed must be positive, got {speed}")

        motion = entry.get("motion", "none")
        if motion == "auto":
            motion = auto_motion(index)
        if motion not in MOTIONS:
            raise ValueError(
                f"{where} has unknown motion {motion!r}; expected one of "
                f"{', '.join(MOTIONS)} or 'auto'")

        transition = entry.get("transition", "crossfade")
        if transition not in TRANSITIONS:
            raise ValueError(
                f"{where} has unknown transition {transition!r}; expected one "
                "of " + ", ".join(TRANSITIONS))

        zoom = entry.get("zoom", 1.10)
        zoom_range(zoom, motion)  # validate now, not on the first frame

        look, grade_name = resolve_look(entry.get("grade"))

        return cls(start=start, end=end, image=image, anchor=anchor, speed=speed,
                   motion=motion, zoom=zoom, pan=float(entry.get("pan", 0.30)),
                   fit=bool(entry.get("fit", False)), look=look,
                   grade_name=grade_name, transition=transition,
                   label=entry.get("label", ""))


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
    source_size: tuple = (0, 0)
    source_fps: float = 30.0
    source_duration: float = 0.0

    @classmethod
    def load(cls, raw: dict, args) -> "ReelSpec":
        source = raw.get("source", args.input)
        output = raw.get("output", args.output)
        if not source or not os.path.exists(source):
            raise ValueError(f"source video not found: {source}")
        if not output:
            raise ValueError("no output path — pass -o or set 'output' in the spec")
        if not raw.get("clips"):
            raise ValueError("spec has no 'clips'")

        from moviepy import VideoFileClip
        probe = VideoFileClip(source)
        try:
            src_size = (probe.size[0], probe.size[1])
            src_fps = float(probe.fps or 30.0)
            src_duration = float(probe.duration)
        finally:
            probe.close()

        hold = float(raw.get("hold", args.hold))
        clips = [ClipSpec.parse(e, i, hold, src_duration)
                 for i, e in enumerate(raw["clips"])]

        size_text = raw.get("size", args.size)
        aspect_text = raw.get("aspect", args.aspect)
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
            fps=int(raw.get("fps", args.fps)),
            crossfade=float(raw.get("crossfade", args.crossfade)),
            audio=bool(raw.get("audio", False)),
            crf=int(raw.get("crf", args.crf)),
            preset=str(raw.get("preset", args.preset)),
            threads=int(raw.get("threads", args.threads)),
            codec=str(raw.get("codec", args.codec)),
            audio_codec=str(raw.get("audio_codec", "aac")),
            resample=str(raw.get("resample", args.resample)),
            source_size=src_size, source_fps=src_fps, source_duration=src_duration,
        )
        if spec.resample not in RESAMPLE_MODES:
            raise ValueError(
                f"unknown resample mode {spec.resample!r}; expected one of "
                + ", ".join(RESAMPLE_MODES))
        spec.check_crossfade()
        return spec

    def check_crossfade(self) -> None:
        """
        A dissolve eats into both neighbours, so a clip shorter than the
        overlap never fully appears. Only dissolving clips are at risk — a
        hard cut can be as short as you like.
        """
        if self.crossfade <= 0:
            return
        dissolving = [c.length for i, c in enumerate(self.clips)
                      if i > 0 and c.transition == "crossfade"]
        if dissolving and self.crossfade >= min(dissolving):
            raise ValueError(
                f"crossfade ({self.crossfade}s) must be shorter than the "
                f"shortest dissolving clip ({min(dissolving):.2f}s), otherwise "
                "that clip never fully appears")


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

        framing = build_framing(source, {
            "fit": clip.fit, "motion": clip.motion, "zoom": clip.zoom,
            "pan": clip.pan, "anchor": clip.anchor,
        }, out_w, out_h, clip.look, spec.resample)

        shots.append(Shot(source, framing, Grade.compile(clip.look, out_w, out_h),
                          duration, clip.label, clip.grade_name))
    return shots


def build_audio(spec: ReelSpec, timeline: Timeline):
    """
    Lay the same ranges out on an audio track, at the same times.

    Every clip gets a few milliseconds of fade at each end. Cutting audio at
    an arbitrary sample leaves a step in the waveform, and a step is a click.

    A clip that dissolves gets a fade as long as the dissolve instead, so the
    two tracks cross over each other at the same rate the pictures do.
    """
    from moviepy import AudioFileClip, CompositeAudioClip, afx

    probe = AudioFileClip(spec.source)
    if probe.duration is None:
        probe.close()
        return None, None

    edge = min(0.02, spec.crossfade / 2.0 if spec.crossfade > 0 else 0.02)
    tracks = []
    for clip, (start, _, _, _) in zip(spec.clips, timeline.placements):
        if clip.image:
            continue
        piece = probe.subclipped(clip.start, clip.end)
        effects = []
        if clip.speed != 1.0:
            effects.append(afx.MultiplySpeed(clip.speed))
        if clip.transition == "crossfade" and spec.crossfade > 0:
            effects.append(afx.AudioFadeIn(spec.crossfade))
        else:
            effects.append(afx.AudioFadeIn(edge))
        effects.append(afx.AudioFadeOut(edge))
        tracks.append(piece.with_effects(effects).with_start(start))

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
        params = ["-crf", str(spec.crf), "-pix_fmt", "yuv420p",
                  "-movflags", "+faststart"]
        clip.write_videofile(
            spec.output, fps=spec.fps, codec=spec.codec,
            audio=audio_track is not None,
            # Named rather than inferred: left to itself moviepy picks mp3 for
            # an .mp4, which plays but is not what anything expects to find
            # in that container.
            audio_codec=spec.audio_codec,
            preset=spec.preset,
            threads=spec.threads or None,
            ffmpeg_params=params,
            logger="bar" if progress else None,
        )
    finally:
        timeline.release()
        for closeable in (audio_track, audio_probe, clip):
            try:
                closeable.close()
            except Exception:
                pass
        release_scratch()

    elapsed = time.perf_counter() - started
    frames = int(round(timeline.duration * spec.fps))
    return {
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
        "size_mb": round(os.path.getsize(spec.output) / 1e6, 2),
        "render_seconds": round(elapsed, 1),
        "ms_per_frame": round(elapsed / max(frames, 1) * 1000, 1),
        "clips": [
            {
                "source": (os.path.basename(c.image) if c.image else
                           f"{format_timestamp(c.start)} -> {format_timestamp(c.end)}"),
                "length": round(c.length, 2),
                "grade": c.grade_name,
                "motion": c.motion,
                "transition": c.transition,
                "speed": c.speed,
                "fit": c.fit,
                "label": c.label,
            }
            for c in spec.clips
        ],
    }


# ---------------------------------------------------------------------- CLI

def ranges_to_clips(text: str) -> list:
    """Turn `0:08-0:13,1:20-1:25` into spec entries."""
    entries = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"range {chunk!r} needs a start and an end, e.g. 0:08-0:13")
        head, tail = chunk.rsplit("-", 1)
        entries.append({"start": head.strip(), "end": tail.strip()})
    if not entries:
        raise ValueError("--ranges was empty")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cut highlights out of a video and join them into one reel")
    parser.add_argument("--spec", help="JSON spec file (the full-control path)")
    parser.add_argument("-i", "--input", help="Source video")
    parser.add_argument("-o", "--output", help="Output reel")
    parser.add_argument("--ranges", help="Shorthand: 0:08-0:13,1:20-1:25")
    parser.add_argument("--aspect", help="Delivery aspect, e.g. 9:16")
    parser.add_argument("--size", help="Exact output size, e.g. 1080x1920")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold", type=float, default=3.0,
                        help="Default clip length when neither end nor duration is given")
    parser.add_argument("--crossfade", type=float, default=0.35)
    parser.add_argument("--crf", type=int, default=23,
                        help="x264 quality: 18 near-lossless, 23 default, 28 small")
    parser.add_argument("--preset", default="medium",
                        help="x264 effort: ultrafast .. veryslow (default medium)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Encoder threads; 0 uses every core")
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--resample", default="quality", choices=RESAMPLE_MODES,
                        help="quality keeps LANCZOS everywhere; fast uses "
                             "BICUBIC when enlarging, about 6%% quicker")
    parser.add_argument("--progress", action="store_true",
                        help="Show an encoding progress bar on stderr")
    parser.add_argument("--grades", action="store_true",
                        help="List the grade presets and exit")
    args = parser.parse_args()

    if args.grades:
        print(json.dumps({name: vars(look) for name, look in PRESETS.items()}, indent=2))
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

    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
