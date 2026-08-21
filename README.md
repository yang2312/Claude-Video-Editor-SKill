# Cinematic Reel

A Claude Code skill that turns a long video into a short graded one.

Point it at footage, hand it a list of time ranges, and it cuts them out,
reframes each to your delivery aspect with a camera move of its own, grades
them, and joins them with the transition each junction asks for. It was built
to make a 15-second 9:16 promo reel out of a 3-minute 16:9 drone video, and it
still does that in about a minute.

```bash
python scripts/scan_source.py -i source.mp4 --min-length 2.0
python scripts/build_reel.py --spec highlights.json
```

## Install

Copy the directory into `.claude/skills/cinematic-reel/` in your project, or
into `~/.claude/skills/` to have it everywhere.

```bash
pip install -r requirements.txt
```

`moviepy` ships its own ffmpeg binary, so ffmpeg does **not** need to be
installed or on PATH.

## What it does

```json
{
  "source": "source.mp4",
  "output": "reel.mp4",
  "aspect": "9:16",
  "crossfade": 0.35,
  "crf": 25,
  "clips": [
    {"image": "1995-gate.png", "duration": 2.8, "fit": true,
     "grade": "vintage", "motion": "zoom-in", "zoom": [1.0, 1.10]},
    {"start": 15.5, "end": 17.6, "grade": "faded", "transition": "flash"},
    {"start": 152.8, "end": 154.1, "grade": "vivid", "motion": "pan-right"}
  ]
}
```

Three levers do the work:

**Grade** — `vintage`, `faded`, `warm`, `vivid`, or eight knobs of your own
(`saturation`, `temperature`, `contrast`, `lift`, `gamma`, `vignette`,
`grain`, `softness`). Running a grade *arc* across a reel — desaturated at the
top, saturated at the end — says "time passed" without a caption.

**Motion** — `zoom-in`, `zoom-out`, `pan-left`, `pan-right`, `auto`. Cropping
16:9 to 9:16 throws away two thirds of the width; a moving window spends that
slack instead, which lets one locked-off drone shot read as two angles. It is
also the only camera move a photograph has.

**Transition** — `crossfade` reads as time passing and costs a beat; `cut` is
the only thing that makes a section feel fast; `flash` is a white bloom that
reads as *somewhere else now*, which is what carries a jump between two eras.

## Scan first

```bash
python scripts/scan_source.py -i source.mp4
```

An edited source hides two kinds of boundary. A **hard cut** is a one-frame
spike. A **dip transition** ramps down to black and back over a second or
more and never produces a spike at all — so a detector that only looks for
spikes reports a clean stretch that is not clean, and the reel gets a black
flash in the middle of a clip.

You cannot see a dip by looking at two still frames either, because it is a
ramp: both ends look fine. Scan, then pick from the clean stretches.

## Performance

The renderer is roughly **four times faster** than the straightforward version
it replaces, on the same footage and with the same output, single-threaded and
allocating nothing per frame. Measured on a 4.3s 1080×1920 reel, 8-core
laptop:

| | before | after |
|---|---|---|
| grade | 294 ms/frame | 50 ms/frame |
| frame generation | 486 ms/frame | 121 ms/frame |
| whole render | 77.8 s | 19.2 s |

Where it came from, in order of size:

**The grade collapses into one matrix.** Saturation is a channel mix,
temperature a channel scale, contrast and lift affine maps. Four linear maps
compose into one, so all four become a single 3×3 matrix and an offset,
computed once per look instead of four passes per frame. Gamma is pointwise
and monotone, so it becomes a 256-entry table.

**Broadcasting across the innermost axis is pathological.** Multiplying a
(H, W, 3) frame by a (H, W, 1) vignette mask took 20 ms; storing the mask
already expanded to (H, W, 3) took 2.7 ms for 17 MB more memory. Adding a
length-3 offset vector to an (N, 3) buffer took 10 ms where the same value as
a scalar took 1.7 ms.

**The timeline is hand-written, not composited.** A `CompositeVideoClip`
treats every moment as a stack of layers to alpha-blend, so a reel where 95%
of frames have one visible shot still pays for masks and RGBA conversion on
all of them. Owning placement makes the common case `return shot.frame(t)`,
and the white flash — which touches thirteen frames of a fifteen-second reel —
stops charging the other four hundred for a float32 conversion.

**Grain slides instead of being redrawn.** White noise has no autocorrelation
at any non-zero lag, so eight overlapping windows onto one buffer are as
independent as eight separate draws, at a quarter of the memory.

Things that were measured and *rejected*: folding the tone curve and vignette
into one 2-D lookup table (41 ms against 20 ms — gathers are slower than
arithmetic), and BICUBIC upscaling as the default (6% faster, but it changes
the picture, so it is a flag instead).

## Tests

```bash
python tests/test_reel.py
```

27 tests, no pytest, no network, no fixture files in the repo — a synthetic
source video is generated on first run. The suite checks the compiled grade
against a plainly-written version of the eight-step chain it replaces, because
that equivalence is the entire performance argument.

## What it cannot do

Add depth of field, relight a scene, compose music, or remove a burned-in
watermark. Those are decided at the lens, on the day, or by a composer. It
says so rather than pretending.

## Licence

MIT.
