#!/usr/bin/env python3
"""
Source Scanner — find stretches of footage safe to cut from.

Why this exists:
    Picking a range by eyeballing two still frames is not enough. An edited
    source hides two different kinds of boundary, and they need different
    detectors:

      hard cut       a single-frame spike in frame-to-frame difference
      dip transition a smooth ramp down to black and back, a second or more
                     wide, that never produces a spike at all

    A range that straddles either one plays as a glitch in the finished reel.
    Scanning for spikes alone misses every dip — which is exactly the mistake
    this script was written to stop repeating.

Usage:
    python scan_source.py -i source.mp4
    python scan_source.py -i source.mp4 --min-length 2.0 --json

Output: the clean intervals, longest first, plus the boundaries it found.
"""

import argparse
import json
import sys

import numpy as np


def scan(path: str, rate: float = 8.0, min_length: float = 1.5,
         dip_ratio: float = 0.55, spike_ratio: float = 2.6) -> dict:
    """
    Walk the video and classify each sample as usable or boundary.

    rate        samples per second
    min_length  shortest interval worth reporting
    dip_ratio   brightness below this fraction of the median counts as a dip
    spike_ratio frame difference above this multiple of the median is a cut
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        return {"status": "error", "message": "moviepy not installed. Run: pip install moviepy"}

    clip = VideoFileClip(path)
    try:
        duration = clip.duration
        ts = np.arange(0.0, duration, 1.0 / rate)

        lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        bright, diffs, prev = [], [0.0], None

        for t in ts:
            small = clip.get_frame(float(t))[::8, ::8].astype(np.float32) / 255.0
            bright.append(float((small * lum_w).sum(axis=2).mean()))
            if prev is not None:
                diffs.append(float(np.abs(small - prev).mean()))
            prev = small

        bright = np.array(bright)
        diffs = np.array(diffs)

        bright_med = float(np.median(bright))
        diff_med = float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 0.0

        is_dip = bright < bright_med * dip_ratio
        is_spike = diffs > max(diff_med * spike_ratio, diff_med + 0.02)
        bad = is_dip | is_spike

        # Widen each boundary by one sample on both sides — a dip's shoulders
        # are already visibly dark before the detector trips.
        pad = bad.copy()
        for i in np.flatnonzero(bad):
            pad[max(0, i - 1):min(len(bad), i + 2)] = True

        intervals, run_start = [], None
        for i, is_bad in enumerate(pad):
            if not is_bad and run_start is None:
                run_start = i
            elif is_bad and run_start is not None:
                intervals.append((float(ts[run_start]), float(ts[i - 1])))
                run_start = None
        if run_start is not None:
            intervals.append((float(ts[run_start]), float(duration)))

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
            "median_brightness": round(bright_med, 3),
            "dips": [round(float(ts[i]), 2) for i in np.flatnonzero(is_dip)],
            "hard_cuts": [round(float(ts[i]), 2) for i in np.flatnonzero(is_spike)],
            "clean_count": len(clean),
            "clean": clean,
        }
    finally:
        try:
            clip.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Find stretches of a video that are safe to cut clips from")
    parser.add_argument("-i", "--input", required=True, help="Source video")
    parser.add_argument("--rate", type=float, default=8.0, help="Samples per second (default 8)")
    parser.add_argument("--min-length", type=float, default=1.5,
                        help="Shortest interval to report (default 1.5s)")
    parser.add_argument("--json", action="store_true", help="Full JSON instead of a table")

    args = parser.parse_args()
    result = scan(args.input, args.rate, args.min_length)

    if result["status"] != "success":
        print(json.dumps(result, indent=2))
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
        sys.exit(0)

    print(f"{result['source']}  {result['duration']}s  {result['resolution']}")
    print(f"dip transitions: {len(result['dips'])} samples   hard cuts: {len(result['hard_cuts'])}")
    print(f"\nclean stretches >= {args.min_length}s, longest first:")
    for c in result["clean"]:
        print(f"  {c['start']:7.2f} - {c['end']:7.2f}   ({c['length']:5.2f}s)")
    sys.exit(0)


if __name__ == "__main__":
    main()
