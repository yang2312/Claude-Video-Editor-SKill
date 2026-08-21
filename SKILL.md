---
name: cinematic-reel
description: Cut highlight ranges out of a long video and join them into a short graded reel at any delivery aspect (9:16, 1:1, 16:9). Use when the user wants a promo reel, a highlights cut, a vertical edit for social, or a short film made from existing footage, with colour grading, camera moves on static shots, and transitions. Also use to find safe cut points in a source video.
---

# Cinematic Reel

Turn a long video into a short one that is worth watching.

The pipeline is three steps, and skipping the first is the most common way to
produce a broken reel:

```
scan_source.py    find stretches that are safe to cut from
   ↓
(you pick ranges by looking at frames)
   ↓
build_reel.py     cut, reframe, grade, join, encode
```

## Scan before you pick

```bash
python scripts/scan_source.py -i source.mp4 --min-length 2.0
```

An edited source hides two kinds of boundary, and they need different
detectors:

- a **hard cut** is a single-frame spike in frame-to-frame difference
- a **dip transition** is a smooth ramp down to black and back, often more
  than a second wide, that never produces a spike at all

A range straddling either one plays as a glitch. Looking at two still frames
will not tell you a dip sits between them — it is a ramp, so both ends look
fine. Take ranges only from the clean stretches the scan reports.

## Write a spec

```json
{
  "source": "source.mp4",
  "output": "reel.mp4",
  "aspect": "9:16",
  "fps": 30,
  "crossfade": 0.35,
  "crf": 25,
  "clips": [
    {"image": "old-photo.png", "duration": 2.8, "fit": true,
     "grade": "vintage", "motion": "zoom-in", "zoom": [1.0, 1.10]},
    {"start": 15.5, "end": 17.6, "grade": "faded", "motion": "zoom-out",
     "transition": "flash"},
    {"start": 152.8, "end": 154.1, "grade": "vivid", "motion": "pan-right"}
  ]
}
```

```bash
python scripts/build_reel.py --spec highlights.json
```

It prints a JSON report — length, size, render time, and every clip with the
grade and move it got. Exit code 1 with `{"status": "error"}` on failure; the
message names the offending clip by index.

For a quick cut with no grading:

```bash
python scripts/build_reel.py -i source.mp4 -o reel.mp4 \
    --ranges 0:08-0:13,1:20-1:25 --aspect 9:16
```

## The three levers that make a reel feel edited

**Grade** carries the story. Presets: `vintage`, `faded`, `warm`, `vivid`,
`none`. A grade arc — starting desaturated and warming into saturated — reads
as time passing without a single caption. Any preset can be overridden:
`{"preset": "vintage", "grain": 0.02}`. The eight knobs are `saturation`,
`temperature`, `contrast`, `lift`, `gamma`, `vignette`, `grain`, `softness`;
run `build_reel.py --grades` to print them all.

**Motion** gives one locked-off angle two different framings. Cutting 16:9 to
9:16 throws away two thirds of the width, and `zoom-in`, `zoom-out`,
`pan-left`, `pan-right` spend that slack instead of discarding it. `auto`
rotates through them so no two neighbours repeat a move. `zoom` takes a
number, or `[start, end]` when a shot needs to begin already tight.

**Transition** controls pace, independently of clip length:

- `crossfade` — dissolves; reads as time passing, and it costs a beat
- `cut` — hard join; the only way to make a section feel fast
- `flash` — hard join under a white bloom; reads as "somewhere else now",
  which is what carries a jump between two eras

A reel that dissolves throughout has one tempo. Shortening clips *and*
switching to hard cuts is what makes a back half accelerate.

## When to use `fit`

By default a clip is cropped to the delivery aspect. Use `"fit": true` for
shots whose meaning spans the full width — a sign, a banner, a wide
photograph — where a 9:16 crop would cut the words in half. The frame is
letterboxed on a blurred, darkened bed of itself.

Before choosing, do the arithmetic: a 9:16 window on a 940px-wide photo is
only 355px across, so anything wider than that gets cut. Fit costs you
screen area — the picture occupies a band and the rest is bed, which doubles
as a place to put a title.

Fit honours `motion` too, growing the picture past the frame edges.

## Encoding

`crf` sets quality, and moviepy's default bitrate is far too generous
without it: 18 is near-lossless, 23 is the default, 28 is small. Grain is
expensive to encode, so a heavily graded reel needs a higher `crf` than clean
footage to hit the same size — a 15s vertical reel with `vintage` on it lands
near 40 MB at default bitrate and 20 MB at `crf` 25.

`preset` is x264 effort, `medium` by default. `slow` buys roughly 10% off the
file for four times the encoding time. `threads` caps the encoder; 0 uses
every core.

`resample` is `quality` (LANCZOS everywhere) by default. `fast` uses BICUBIC
when enlarging for about 6% off the render — fine for a draft, not for a
master, and it does change the picture.

## Audio

Off by default. `"audio": true` lays the same ranges onto an audio track at
the same times, with a fade at every cut so the joins do not click, and a
fade as long as the dissolve on clips that dissolve.

Many downloads are video-only streams with no audio track at all — check
before promising a soundtrack.

## Checking the result

Render parameters are not evidence. Read frames out of the finished file and
look at them:

```bash
python -c "
from moviepy import VideoFileClip; import numpy as np
c = VideoFileClip('reel.mp4')
for t in np.arange(0.1, c.duration, 0.5):
    print(round(t,1), round(float(c.get_frame(t).mean())/255, 3))
c.close()"
```

A frame under about 0.05 mean brightness means a range straddled a dip that
the scan should have caught. Beyond that, actually look at frames from each
clip — a crop can cut the school name off a sign, or centre on the wrong
building, and no number will tell you.

## Things it cannot do

- **Add depth of field.** Shallow focus is decided at the lens. Faking it in
  post on wide drone footage looks wrong.
- **Change the lighting.** It can be graded, not relit.
- **Compose music.** Audio can be cut and placed, not created.
- **Remove a watermark.** A burned-in logo survives cropping and grading, and
  is most visible in `fit` shots where the full frame is on screen.

Say so rather than implying otherwise. Footage rights belong to whoever shot
it; credit and permission are the user's to arrange.

## Files

| | |
|---|---|
| `scripts/scan_source.py` | find cut-safe stretches |
| `scripts/build_reel.py` | CLI, spec parsing, orchestration, encoding |
| `scripts/reel_grade.py` | the look engine |
| `scripts/reel_timeline.py` | framing, shot placement, transitions |
| `tests/test_reel.py` | `python tests/test_reel.py` — no pytest, no network |
| `examples/` | a worked spec |

Needs `moviepy`, `numpy` and `pillow`. moviepy bundles its own ffmpeg, so
ffmpeg does **not** need to be on PATH.
