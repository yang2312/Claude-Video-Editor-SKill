#!/usr/bin/env python3
"""
Source Scanner — find the stretches of footage that are safe to cut from.

Why this exists
---------------
Picking a range by eyeballing two still frames is not enough. An edited source
hides two different kinds of boundary, and they need different detectors:

    hard cut        a single-frame spike in frame-to-frame difference
    dip transition  a smooth ramp down to black and back, a second or more
                    wide, that never produces a spike at all

A range that straddles either one plays as a glitch in the finished reel. You
cannot see a dip by comparing the two ends of a range either, because it is a
ramp — both ends look fine. Scanning for spikes alone misses every dip, which
is exactly the mistake this script was written to stop repeating.

Usage
-----
    python scan_source.py -i source.mp4 > scan.json
    python scan_source.py -i source.mp4 --table --min-length 2.0

Then point a reel spec at the result and every clip range is checked against
it before anything is encoded:

    {"source": "source.mp4", "guard": "scan.json", "clips": [...]}

That last step is the point. Before it existed, the scan's numbers were
retyped by hand into the spec's free-text labels and nothing could check them,
so a bad range was found only after the render finished.

What it costs, and what it assumes
----------------------------------
One decoded frame per sample, `--rate` samples a second, seeking each time.
A three-minute source at the default rate is about 1,400 decodes — expect
tens of seconds, and use `--progress` if you want to watch it.

Both thresholds are relative to the source's own median, so the detectors
adapt to how a video is lit rather than to an absolute level. The assumption
underneath that is a source of roughly consistent exposure. Deliberately dark
footage raises the dip threshold against itself and may report nothing; tune
with `--dip-ratio` if a scan looks wrong, and always sanity-check the count.

Output is JSON on stdout, always, and errors take the same shape as
build_reel.py so one pipeline never has to parse two failure formats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Every sample is decimated to one pixel in eight on both axes before being
# compared. It makes the scan roughly sixty times cheaper and costs nothing
# for either detector, both of which work on whole-frame averages.
#
# The thresholds below are tuned against this decimation. Change it and the
# median frame difference changes with it, which silently retunes SPIKE_RATIO.
DECIMATE = 8

# A sample darker than this fraction of the source's median brightness is
# inside a dip. Set well above black: the shoulders of a ramp are already
# visibly dark long before the frame reaches zero, and a range that starts on
# a shoulder still looks like a fault.
DIP_RATIO = 0.55

# A frame difference this many times the median is a hard cut. High enough
# that a fast pan or a camera flash does not trip it; the `+ 0.02` floor
# alongside it stops a static source, where the median difference is nearly
# zero, from flagging every sample.
SPIKE_RATIO = 2.6

# Rec. 709, matching the renderer, so "dark" means the same thing in both.
LUM_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def scan(path: str, rate: float = 8.0, min_length: float = 1.5,
         dip_ratio: float = DIP_RATIO, spike_ratio: float = SPIKE_RATIO,
         progress=None) -> dict:
    """
    Walk the video and classify every sample as usable or boundary.

    rate         samples per second
    min_length   shortest interval worth reporting
    dip_ratio    brightness below this fraction of the median counts as a dip
    spike_ratio  frame difference above this multiple of the median is a cut
    progress     optional callable(done, total) for a long scan
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"source video not found: {path}")

    from moviepy import VideoFileClip

    clip = VideoFileClip(path)
    try:
        duration = float(clip.duration)
        times = np.arange(0.0, duration, 1.0 / rate)

        bright, diffs, previous = [], [0.0], None
        for i, t in enumerate(times):
            small = clip.get_frame(float(t))[::DECIMATE, ::DECIMATE]
            small = small.astype(np.float32) / 255.0
            bright.append(float((small * LUM_WEIGHTS).sum(axis=2).mean()))
            if previous is not None:
                diffs.append(float(np.abs(small - previous).mean()))
            previous = small
            if progress is not None and i % 40 == 0:
                progress(i, len(times))

        bright = np.array(bright)
        diffs = np.array(diffs)

        bright_median = float(np.median(bright))
        diff_median = (float(np.median(diffs[diffs > 0]))
                       if (diffs > 0).any() else 0.0)

        is_dip = bright < bright_median * dip_ratio
        is_spike = diffs > max(diff_median * spike_ratio, diff_median + 0.02)
        bad = is_dip | is_spike

        # Widen each boundary by one sample on both sides — a dip's shoulders
        # are already visibly dark before the detector trips.
        padded = bad.copy()
        for i in np.flatnonzero(bad):
            padded[max(0, i - 1):min(len(bad), i + 2)] = True

        intervals, run_start = [], None
        for i, is_bad in enumerate(padded):
            if not is_bad and run_start is None:
                run_start = i
            elif is_bad and run_start is not None:
                intervals.append((float(times[run_start]), float(times[i - 1])))
                run_start = None
        if run_start is not None:
            # The last inspected sample, not the file's end — nothing was
            # looked at in between, so nothing can be promised about it.
            intervals.append((float(times[run_start]), float(times[-1])))

        clean = [
            {"start": round(a, 2), "end": round(b, 2), "length": round(b - a, 2)}
            for a, b in intervals if b - a >= min_length
        ]
        clean.sort(key=lambda c: c["length"], reverse=True)

        return {
            "status": "success",
            "source": path,
            "duration": round(duration, 2),
            "resolution": f"{clip.size[0]}x{clip.size[1]}",
            "sample_rate": rate,
            "min_length": min_length,
            "median_brightness": round(bright_median, 3),
            "dips": [round(float(times[i]), 2) for i in np.flatnonzero(is_dip)],
            "hard_cuts": [round(float(times[i]), 2) for i in np.flatnonzero(is_spike)],
            "clean_count": len(clean),
            "clean": clean,
        }
    finally:
        try:
            clip.close()
        except Exception:
            pass


def as_table(result: dict) -> str:
    lines = [
        f"{result['source']}  {result['duration']}s  {result['resolution']}",
        f"dip samples: {len(result['dips'])}   hard cuts: {len(result['hard_cuts'])}",
        "",
        f"clean stretches >= {result['min_length']}s, longest first:",
    ]
    for c in result["clean"]:
        lines.append(f"  {c['start']:8.2f} - {c['end']:8.2f}   ({c['length']:6.2f}s)")
    if not result["clean"]:
        lines.append("  (none — try a shorter --min-length)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find the stretches of a video that are safe to cut clips from")
    parser.add_argument("-i", "--input", required=True, help="Source video")
    parser.add_argument("--rate", type=float, default=8.0,
                        help="Samples per second (default 8)")
    parser.add_argument("--min-length", type=float, default=1.5,
                        help="Shortest interval to report (default 1.5s)")
    parser.add_argument("--dip-ratio", type=float, default=DIP_RATIO,
                        help=f"Brightness below this fraction of the median is a "
                             f"dip (default {DIP_RATIO})")
    parser.add_argument("--spike-ratio", type=float, default=SPIKE_RATIO,
                        help=f"Frame difference above this multiple of the median "
                             f"is a cut (default {SPIKE_RATIO})")
    parser.add_argument("--table", action="store_true",
                        help="Human-readable table instead of JSON")
    parser.add_argument("--progress", action="store_true",
                        help="Report sampling progress on stderr")
    args = parser.parse_args()

    def report(done, total):
        print(f"\rscanning {done}/{total} samples", end="", file=sys.stderr, flush=True)

    try:
        result = scan(args.input, args.rate, args.min_length, args.dip_ratio,
                      args.spike_ratio, report if args.progress else None)
    except Exception as error:  # noqa: BLE001 — same JSON contract as build_reel
        result = {"status": "error", "message": str(error),
                  "error_type": type(error).__name__}
        if os.environ.get("REEL_TRACEBACK"):
            import traceback
            result["traceback"] = traceback.format_exc()
        print(json.dumps(result, indent=2))
        return 1

    if args.progress:
        print("", file=sys.stderr)

    print(as_table(result) if args.table else json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
