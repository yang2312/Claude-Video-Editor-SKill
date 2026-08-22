---
name: cinematic-reel
description: Cut highlight ranges out of a long video and join them into a short graded reel at any delivery aspect (9:16, 1:1, 16:9). Use when the user wants a promo reel, a highlights cut, a vertical edit for social, or a short film made from existing footage, with colour grading, camera moves on static shots, and transitions. Also use to find safe cut points in a source video.
---

# Cinematic Reel

Turn a long video into a short one that is worth watching.

```
scan_source.py    find stretches that are safe to cut from  ->  scan.json
   ↓
(you pick ranges by looking at frames)
   ↓
build_reel.py     cut, reframe, grade, join, encode
                  every range checked against scan.json first
```

```bash
python scripts/scan_source.py -i source.mp4 --min-length 2.0 > scan.json
python scripts/build_reel.py --spec highlights.json
```

Both scripts always print JSON and exit 1 on failure. Errors name the
offending clip by index. Set `REEL_TRACEBACK=1` for a stack trace.

## Scan before you pick

An edited source hides two kinds of boundary, and they need different
detectors:

- a **hard cut** is a single-frame spike in frame-to-frame difference
- a **dip transition** is a smooth ramp down to black and back, often more
  than a second wide, that never produces a spike at all

A range straddling either one plays as a glitch. You cannot see a dip by
comparing two still frames, because it is a ramp — both ends look fine.

Take ranges only from the clean stretches the scan reports, then **point the
spec's `guard` at the scan output**. Every range is checked against it before
anything is encoded, so a bad range is refused in a second rather than found
after the render. Without `guard` the reel still builds, and the report says
the ranges went unchecked.

`--table` prints a human-readable version instead of JSON. `--progress`
reports sampling on stderr; a three-minute source is about 1,400 decodes.

## Write a spec

```json
{
  "source": "source.mp4",
  "output": "reel.mp4",
  "guard": "scan.json",
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

Keys starting with `_` are ignored, so a spec can carry comments. Anything
else unrecognised is an error — a typo is refused, not absorbed.

### Reel keys

| key | default | |
|---|---|---|
| `source` | — | the video to cut from (required) |
| `output` | — | where to write the reel (required) |
| `clips` | — | required |
| `guard` | — | path to `scan_source --json` output |
| `aspect` | `"9:16"` | height comes from the source |
| `size` | — | exact output, e.g. `"1080x1920"` |
| `fps` | 30 | |
| `crossfade` | 0.35 | dissolve length in seconds |
| `hold` | 3.0 | clip length when neither `end` nor `duration` is given |
| `audio` | false | |
| `crf` | 23 | 18 near-lossless, 23 default, 28 small |
| `preset` | `"medium"` | x264 effort, `ultrafast` … `veryslow` |
| `threads` | 0 | encoder threads; 0 uses every core |
| `codec` | `"libx264"` | |
| `audio_codec` | `"aac"` | |
| `resample` | `"quality"` | or `"fast"` |

**Give `aspect` or `size`, not both** — it is an error, because they resolve
differently. From a 1080p source, `aspect: "9:16"` yields 608×1080 (the height
comes from the source); `size` yields exactly what it says.

### Clip keys

| key | default | |
|---|---|---|
| `start` / `end` | — | a range in the source; `duration` may replace `end` |
| `image` | — | a still standing in for footage; timed by `duration` |
| `motion` | `"none"` | `zoom-in`, `zoom-out`, `pan-left`, `pan-right`, `auto` |
| `zoom` | — | a number, or `[start, end]` |
| `pan` | 0.30 | 0..1, how much of the available slack a pan spends |
| `anchor` | 0.5 | 0=left … 1=right, or `[x, y]` for both axes |
| `speed` | 1.0 | <1 slows, >1 speeds up. Footage only |
| `shutter` | 0 | motion blur, in degrees of shutter angle |
| `shutter_samples` | 3 | sub-frames the exposure is built from |
| `stutter` | 0 | hold the shot at this many frames a second |
| `fit` | false | letterbox instead of cropping |
| `grade` | — | a preset name, or an object of overrides |
| `transition` | `"crossfade"` | see below |
| `label` | `""` | free text, echoed in the report |

`zoom` means different things per motion, and the report always echoes the
resolved pair so it never has to be guessed:

| | `zoom: 1.10` | omitted |
|---|---|---|
| `zoom-in` | 1.00 → 1.10 | 1.00 → 1.10 |
| `zoom-out` | 1.10 → 1.00 | 1.10 → 1.00 |
| `pan-left` / `pan-right` | held at 1.10 | held at 1.00 |
| `none` | held at 1.10 | held at 1.00 |

### CLI

`--spec` takes the file above. Everything else is also a flag:
`-i/--input`, `-o/--output`, `--ranges`, `--guard`, `--aspect`, `--size`,
`--fps`, `--hold`, `--crossfade`, `--audio`, `--crf`, `--preset`,
`--threads`, `--codec`, `--resample`, `--progress`, `--grades`.

A flag you type wins over the spec file, which wins over the default.

```bash
python scripts/build_reel.py -i source.mp4 -o reel.mp4 \
    --ranges 0:08-0:13,1:20-1:25 --aspect 9:16
```

## The three levers that make a reel feel edited

**Grade** carries the story. Presets: `vintage`, `faded`, `warm`, `vivid`,
`opium`, `dirty`, `none`. A grade arc — desaturated at the top, saturated at
the end — reads as time passing without a caption. Override any preset:
`{"preset": "vintage", "grain": 0.02}`. `build_reel.py --grades` prints every
preset with its numbers.

The knobs are `saturation`, `temperature`, `contrast`, `lift`, `gamma`,
`black`, `white`, `glow`, `glow_threshold`, `rgb_split`, `vignette`, `grain`,
`softness`.

`black` and `white` are input levels: everything below `black` is crushed to
zero, everything above `white` is blown out. Pulling them together is the
"extract" look — high contrast, detail only where the light was — and it costs
nothing, because it folds into the same table as gamma. `glow` spills light
out of the highlights; `rgb_split` separates red and blue by a few pixels for
chromatic aberration. Those two are spatial and do cost time; the rest are
free once the look is compiled.

`opium` and `dirty` are the heavy ones, built to sit under three-frame cuts
where a subtle grade would not register: `opium` is crushed to near-monochrome
with blooming highlights, `dirty` is noisy and split.

**Motion** gives one locked-off angle two framings. Cutting 16:9 to 9:16
throws away two thirds of the width, and the moves spend that slack instead of
discarding it. `auto` rotates through them; mixing `auto` with explicit moves
can still put two of a kind together, and the report warns when it does.

**Transition** controls pace, independently of clip length:

| | |
|---|---|
| `crossfade` | dissolves; reads as time passing, and costs a beat |
| `cut` | hard join; the only way to make a section feel fast |
| `flash` | hard join under a white bloom; "somewhere else now" |
| `invert` | two frames of inverted colour |
| `invert-r`, `invert-g`, `invert-b` | the same on one channel, so the fault has a colour |
| `shake` | a decaying jitter with a brightness pop, about a third of a second |

A reel that dissolves throughout has one tempo. Shortening clips *and*
switching to hard cuts is what makes a back half accelerate.

White reads as light; **inversion reads as a fault in the picture**, which is
why it lands harder between very short cuts. Alternating the channel across a
burst — `invert`, then `invert-r`, then `invert-b` — stops a run of them
turning into a strobe. Punctuation on the first clip is refused: there is
nothing before it to punctuate.

Two limits are enforced, not documented-and-hoped: a clip cannot be shorter
than the crossfade that dissolves into it (it would never appear), and no
three clips may overlap at once (a dissolve blends two). Both are refused at
parse time with the clip named.

## When to use `fit`

By default a clip is cropped to the delivery aspect. Use `"fit": true` for
shots whose meaning spans the full width — a sign, a banner, a wide photograph
— where a 9:16 crop would cut the words in half. The frame is letterboxed on a
blurred, darkened bed of itself.

Do the arithmetic before choosing. A 9:16 window on a 940×632 photo is only
355 px across (the window height is the source height, and its width follows
the delivery aspect), so anything wider than that gets cut. Fit costs screen
area — the picture occupies a band and the rest is bed, which doubles as a
place to put a title.

Fit honours `motion`, growing the picture past the frame edges. A **pan in fit
mode needs a `zoom` above 1.0** to have anywhere to travel, and is refused
without one rather than rendering a frozen shot.


## Cutting for the short-form look

The fast, high-energy edit has a small and specific vocabulary. All of it is
reachable from a spec:

**Very short clips.** Three to nine frames — 0.1 to 0.3 seconds at 30fps.
Shorter is more energy. Set `crossfade` to 0 and use `cut`, or the dissolves
will eat clips this short (and the planner will refuse the spec if they do).

**A move on every clip, and motion blur under it.** A push from `zoom: 1.35`
across four frames reads as a slideshow of sharp frames without `shutter`.
180 is the film convention; 360 is fully open and smears. It only blurs this
renderer's own move — the footage already carries whatever the camera did —
so a clip with no move gets a warning rather than wasted work.

**Punctuation between, not dissolves.** `invert` for two frames, alternating
channels across a burst. `shake` at the end of a run.

**A long clip to land on.** A burst is setup; without something longer after
it there is nothing to pay it off. Give the last clip `speed` below 1.

**Texture over the top.** `grain` between 0.10 and 0.15, `stutter` at 8 or 12
for a choppy frame rate, `glow` on the highlights.

```json
{"start": 12.0, "end": 12.15, "grade": "dirty", "motion": "zoom-in",
 "zoom": 1.35, "shutter": 180, "transition": "invert"}
```

Four things from that vocabulary are **not** reachable here, because they need
per-object segmentation and tracking rather than pixel maths: masking a person
out of the background, locking a subject in frame, a silhouette glow, and the
out-of-bounds effect where a subject breaks the edge of the frame. An editor
with a modern NLE has those; this does not, and faking them with pixel maths
looks wrong.

## Encoding

`crf` sets quality, and moviepy's default bitrate is far too generous without
it. Grain is expensive to encode, so a heavily graded reel needs a higher
`crf` than clean footage for the same size — a 15s vertical reel with
`vintage` on it lands near 40 MB at default bitrate and 20 MB at `crf` 25.

`preset` is x264 effort. `slow` buys roughly 10% off the file for four times
the encoding time. `threads` caps the encoder; 0 uses every core.

`resample: "fast"` uses BICUBIC when enlarging for about 6% off the render —
fine for a draft, not for a master, and it does change the picture.

## Audio

Off by default. `"audio": true` lays the same ranges onto an audio track at
the same times, with a fade at every cut so the joins do not click, and a fade
as long as the dissolve on clips that dissolve. Clip 0 cuts in, matching its
picture.

Many downloads are video-only streams with no audio track at all — check
before promising a soundtrack.

## Checking the result

Render parameters are not evidence. Read frames out of the finished file:

```bash
python -c "from moviepy import VideoFileClip; import numpy as np; c=VideoFileClip('reel.mp4'); print([(round(t,1), round(float(c.get_frame(t).mean())/255,3)) for t in np.arange(0.1,c.duration,0.5)]); c.close()"
```

A frame under about 0.05 mean brightness means a range straddled a dip — use
`guard` and it cannot happen. Beyond that, actually look at frames from each
clip: a crop can cut the school name off a sign, or centre on the wrong
building, and no number will tell you.

## Things it cannot do

- **Add depth of field.** Shallow focus is decided at the lens. Faking it on
  wide drone footage looks wrong.
- **Change the lighting.** It can be graded, not relit.
- **Compose music.** Audio can be cut and placed, not created.
- **Remove a watermark.** A burned-in logo survives cropping and grading, and
  is most visible in `fit` shots.

Say so rather than implying otherwise. Footage rights belong to whoever shot
it; credit and permission are the user's to arrange.

## Files

| | |
|---|---|
| `scripts/scan_source.py` | find cut-safe stretches |
| `scripts/build_reel.py` | CLI, spec parsing, validation, encoding |
| `scripts/reel_grade.py` | the look engine |
| `scripts/reel_timeline.py` | camera moves, framing, placement, transitions |
| `tests/test_reel.py` | `python tests/test_reel.py` — no pytest, no network |
| `examples/` | a worked spec |

```bash
pip install -r requirements.txt
```

moviepy bundles its own ffmpeg, so ffmpeg does **not** need to be on PATH.
