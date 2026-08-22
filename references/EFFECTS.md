# The effect catalogue

Every effect this skill can apply, in one place, with the exact key that turns
it on. If you cannot remember what exists, read the index and stop there —
everything below it is detail.

`python scripts/build_reel.py --effects` prints the same index out of the code
itself, so it cannot drift out of date. `--grades` prints the presets with
their numbers.

Each entry lists its search terms in both English and Vietnamese, so that
looking for "nhoè chuyển động" or for "chromatic aberration" lands on the same
key.

---

## Index

| Effect | Where it goes | Value | One line |
|---|---|---|---|
| `zoom-in` | clip `motion` | — | pushes in; the default forward move |
| `zoom-out` | clip `motion` | — | pulls back; reveals context |
| `pan-left` | clip `motion` | — | slides left across the frame |
| `pan-right` | clip `motion` | — | slides right |
| `zoom` | clip key | `1.10` or `[1.0, 1.3]` | how far the move travels |
| `pan` | clip key | `0.0`–`1.0` | how much of the slack a pan crosses |
| `anchor` | clip key | `[0.5, 0.4]` | where the crop sits when it is not moving |
| `ease` | clip key | `smooth`/`impact` | how a move spends its time |
| `spill` | clip key | `[t, b, l, r]` | which block breaks an out-of-bounds border |
| `crossfade` | `transition` | — | dissolve; reads as time passing |
| `cut` | `transition` | — | hard join; the only way to feel fast |
| `flash` | `transition` | — | white bloom over the join |
| `invert` | `transition` | — | two frames of inverted colour |
| `invert-r` `invert-g` `invert-b` | `transition` | — | the same on one channel |
| `shake` | `transition` | — | decaying jitter with a brightness pop |
| `shutter-shake` | `transition` | — | the same with the shutter open, so it smears |
| `film-roll` | `transition` | — | the strip yanked through the gate |
| `out-of-bounds` | `transition` | — | a bordered frame with one block breaking out |
| `speed` | clip key | `0.5`–`2.0` | slow motion or fast motion |
| `shutter` | clip key | `0`–`360` | motion blur under this renderer's move |
| `shutter_samples` | clip key | `2`+ | how smooth that blur is |
| `stutter` | clip key | `8`, `12` | holds the shot at a lower frame rate |
| `hold` | reel key | seconds | how long a still photograph stays up |
| `saturation` | grade knob | `0.0`–`1.5` | colour intensity |
| `temperature` | grade knob | `-1`–`1` | cool blue to warm amber |
| `contrast` | grade knob | around `1.0` | separation between dark and light |
| `lift` | grade knob | `0.0`–`0.2` | milky, raised blacks |
| `gamma` | grade knob | around `1.0` | midtone brightness |
| `black` `white` | grade knobs | `0.0`–`1.0` | input levels — the "extract" look |
| `glow` | grade knob | `0.0`–`1.0` | light spilling out of the highlights |
| `glow_threshold` | grade knob | `0.0`–`1.0` | where that spill starts |
| `rgb_split` | grade knob | pixels | chromatic aberration |
| `vignette` | grade knob | `0.0`–`1.0` | corner falloff |
| `grain` | grade knob | `0.0`–`0.2` | film noise |
| `softness` | grade knob | `0.0`–`1.0` | blend toward a blurred copy |
| `fit` | clip key | `true`/`false` | letterbox instead of cropping |
| `resample` | reel key | `quality`/`fast` | sharpness traded for draft speed |

Presets that bundle grade knobs: `vintage`, `faded`, `warm`, `vivid`,
`opium`, `dirty`, `none`.

---

## Camera moves

A move is what separates a shot from a screenshot. Every clip gets one; on
footage it rides on top of whatever the camera already did.

### `zoom-in` / `zoom-out`

*push in, pull out, ken burns, phóng to, thu nhỏ, đẩy máy*

```json
{"motion": "zoom-in", "zoom": [1.0, 1.25]}
```

`zoom` takes a pair to say exactly where the move starts and ends, or a single
number for "this far, in the direction `motion` names". A push reads as
interest — it says *look closer*. A pull reads as release, which is why it
belongs on a last clip more often than anywhere else.

Past about `1.4` on 1080p footage the crop starts to soften: there are no
longer enough source pixels to fill the delivery frame.

### `pan-left` / `pan-right`

*slide, track, lia máy, trượt ngang*

```json
{"motion": "pan-right", "pan": 0.5}
```

A pan needs somewhere to go, and there are two ways to get it. **A delivery
aspect narrower than the source gives it for free**: a 16:9 source cut to 9:16
uses barely a third of the width, so more than 1300 pixels of a 1920-wide
frame are slack and a pan at `zoom: 1.0` travels the whole way across.
**Matching aspects give none**, and there `zoom` above 1 is what creates it —
a pan at `zoom: 1.0` from 16:9 to 16:9 does nothing at all.

`pan` is the fraction of that slack the move crosses: `1.0` travels edge to
edge, `0.3` is a drift.

At 9:16 this is the most useful move available, and the one that rescues a
wide composition the crop would otherwise cut in half — a sign too long to
fit reads end to end if the frame walks along it.

### `ease`

*easing, impact zoom, punch, snap zoom, giật zoom, zoom nhanh*

```json
{"motion": "zoom-out", "zoom": [1.9, 1.0], "ease": "impact", "shutter": 240}
```

How a move spends its time, which is the whole difference between a camera
pushed and a camera *thrown*.

`smooth` is the default: slow at both ends, what an operator on a slider does.
`impact` puts **41% of the travel in the first tenth of the clip** and 67% in
the first fifth, then settles for the rest — what a hand does when it snaps a
zoom out and lets go.

Pair `impact` with a `shutter`. Without one, the first three frames each land
somewhere far from the last and the eye reads dropped frames rather than
speed; that is the same failure motion blur exists to fix, just concentrated.

It applies to pans as well as zooms, because it changes the clock, not the
geometry.

### `anchor`

*framing, composition, đặt khung, căn khung*

```json
{"anchor": [0.5, 0.35]}
```

Where the crop sits when nothing is moving, as a fraction of the source. The
default `[0.5, 0.5]` is dead centre; `0.35` vertically favours the top, which
is usually right for architecture and for anything with a horizon in it.

---

## Transitions and punctuation

How a clip joins the one before it. Set on the *incoming* clip. Clip 0 has
nothing before it, so anything but `cut` is refused there.

### `crossfade`

*dissolve, fade, chuyển cảnh mềm, hoà tan*

The default. Reads as time passing, and costs a beat — a half-second dissolve
takes half a second out of both clips. A reel that dissolves throughout has
exactly one tempo.

Its length is a reel-level key rather than a per-clip one: `{"crossfade": 0.5}`.
Set it to `0` for a reel of hard cuts.

### `cut`

*hard cut, cắt thẳng*

Nothing between the two frames. The only thing that makes a section feel fast.
Shortening clips *and* switching to cuts is what accelerates a back half.

### `flash`

*white flash, bloom, loé sáng*

A white bloom straddling the join. Reads as *somewhere else now*, which is
what carries a jump between two eras or two places.

### `invert`, `invert-r`, `invert-g`, `invert-b`

*inverted colour, negative, đảo màu, âm bản*

Two frames of inverted picture after the join. White reads as **light**;
inversion reads as a **fault in the picture**, which is why it lands harder
than a flash between very short cuts. The per-channel variants invert only
red, green or blue, so the fault has a colour — alternate them across a burst
(`invert`, then `invert-r`, then `invert-b`) or a run of them turns into a
strobe.

### `shake`

*camera shake, impact, rung lắc, giật hình*

About a third of a second of decaying positional jitter with a brightness pop.
Decay is what separates a shake from a vibration: it has to land. The offset
comes from the frame's own number, so a re-render is identical and two shakes
in one reel do not move in step.

This is the whole of what a "flashy shake preset" is — a transform and a
brightness curve, nothing else.

### `shutter-shake`

*shutter shake, impact shake, rung nhoè, giật mạnh*

A `shake` with the shutter left open across it. A plain shake steps: every
frame is sharp, somewhere else than the last, and the eye reads a sequence of
stills. This one averages the jolt in progress, which is what a real camera
records — the difference between a camera knocked and a camera hit.

Costs four framing passes on about eleven frames. Nothing, at that count.

### `film-roll`

*film roll, reel change, projector, cuộn phim, chuyển cảnh phim*

The strip yanked through the gate. A projector normally hides the mechanism:
it holds one frame still, pulls the next down behind a closed shutter, opens
again. Yank the strip by hand and you see what it hides — the picture sliding
up, and the black frame-line between exposures crossing the gate.

The picture scrolls by three whole frame-heights, a whole number so the strip
lands registered instead of parked halfway. The sideways weave and the dip in
the lamp are part of it: without them this reads as a scrolling web page.

It straddles the junction rather than following it, so the section ending and
the section arriving are on the same strip. That is what makes it a **section
break** — use it between movements, not between shots.

### `out-of-bounds`

*out of bounds, frame break, phá khung, vượt khung*

The picture drops into a bordered frame, and one block of it keeps going past
the border.

**Read this before using it.** The real effect masks a *subject* — a person, a
car — so that the subject crosses the border while the background stays
inside. That needs per-object segmentation, which this module does not have.
What it has is a rectangle: the block it names keeps its full-strength picture
and its full width, and everything outside the border is dimmed and
desaturated into a backdrop.

On a shot where something real occupies that block — a near roofline, a tree,
a building edge crossing the lower frame — it reads correctly, because the eye
takes the break as evidence of depth rather than tracing the outline. On a
shot where the block cuts through empty ground it reads as what it is, a
rectangle. **Choose the shot, not the knob.**

It lasts 1.3 seconds: the frame draws in, holds long enough to be understood
as a frame, and releases. Anything shorter is a flicker rather than an idea.
The border belongs to the shot arriving, so unlike a shake it never appears
on the shot it is leaving.

`spill` is the block, as four fractions — top, bottom, left, right — and it is
the whole difference between the effect reading and not:

```json
{"transition": "out-of-bounds", "spill": [0.50, 1.0, 0.36, 0.92]}
```

The default `[0.55, 1.0, 0.27, 0.75]` is the bottom of centre, where a
foreground usually is. Move it over whatever is actually nearest the camera in
*this* shot.

---

## Time

### `speed`

*slow motion, fast motion, quay chậm, tua nhanh*

```json
{"speed": 0.7}
```

Below 1 slows, above 1 speeds up. Footage only — a photograph has no time axis
for it to act on. Audio taken from the clip is stretched with it.

A slowed clip is where a burst pays off. Without something longer and slower
after a run of fast cuts, the fast cuts were setup with no punchline.

### `shutter` and `shutter_samples`

*motion blur, shutter angle, nhoè chuyển động, mờ chuyển động*

```json
{"motion": "zoom-in", "zoom": 1.35, "shutter": 180}
```

A frame is not an instant. A real shutter is open for a fraction of it, and
everything moving during that fraction is smeared. Without this, a fast push
across four frames reads as a slideshow of sharp frames — the most common tell
of an unfinished edit.

`180` is the film convention, open for half of each frame. `360` is fully open
and smears. `0` is off.

It blurs **this renderer's move**, not the camera's. The footage already
carries whatever the operator did; what has no blur is the crop window sliding
on top of it, so that is what gets sampled. A clip with no move gets a warning
rather than wasted work.

`shutter_samples` (default 3) is how many sub-frames the exposure is built
from. Each one costs a full framing pass, which makes this the most expensive
thing in the catalogue — measured 39 ms to 161 ms per frame at 3 samples. It
is affordable because it belongs on clips three to nine frames long, where a
handful of frames carry it.

### `stutter`

*posterize time, frame rate, choppy, giật khung, khựng hình*

```json
{"stutter": 12}
```

Holds each sample for a whole step, so the shot plays at 8 or 12 frames a
second while the reel around it stays at 30. Applied to the source time and
the move together — a stutter that let the camera keep gliding would read as a
dropped frame rather than as a choice.

### `hold`

*still duration, thời lượng ảnh tĩnh*

A reel-level key: how long a photograph stays on screen. Default 3.2s.

---

## Colour

Thirteen knobs. Any of them can be set alone, or on top of a preset:

```json
{"grade": {"preset": "vintage", "grain": 0.02}}
```

Everything in this section except `glow` and `rgb_split` is **free**: the
linear knobs collapse into one 3×3 matrix and the pointwise ones into one
256-entry table, both computed once per look rather than once per frame.

### `saturation`

*colour, vibrance, độ bão hoà, màu sắc*

`1.0` untouched, `0.0` greyscale, above 1 richer. Desaturating uses Rec. 709
weights, the same ones broadcast uses, so skin and foliage stay where the eye
expects them.

A grade *arc* — desaturated at the top, saturated at the end — says "time
passed" without a caption.

### `temperature`

*white balance, warm, cool, nhiệt độ màu, ám vàng, ám xanh*

`-1` to `1`. Negative cools toward blue, positive warms toward amber.

### `contrast`, `lift`, `gamma`

*độ tương phản, đen mờ, độ sáng trung gian*

`contrast` around 1.0 separates dark from light. `lift` raises the black point
for the milky shadows of aged film. `gamma` below 1 brightens midtones, above
1 deepens them.

### `black` / `white`

*input levels, extract, crush, ép sáng tối, tăng tương phản mạnh*

```json
{"grade": {"black": 0.18, "white": 0.88, "saturation": 0.4}}
```

Input levels. Everything below `black` is crushed to zero, everything above
`white` is blown out. Pulling the two together is the **extract** look: high
contrast, detail only where the light actually was.

Costs nothing — it folds into the same table gamma already uses.

### `glow` and `glow_threshold`

*bloom, halation, haze, loang sáng, quầng sáng*

```json
{"grade": {"glow": 0.45, "glow_threshold": 0.55}}
```

Light spilling out of the highlights. `glow_threshold` is where the spill
starts, as a fraction of full brightness — lower it and more of the picture
blooms.

Genuinely spatial, so it costs real time; the bloom is built at quarter
resolution to keep that bounded. It is best on backlit and hazy footage, where
it amplifies something already in the picture instead of inventing it.

### `rgb_split`

*chromatic aberration, glitch, tách kênh màu, lệch màu*

Separation between the red and blue channels, in pixels. Two or three pixels
reads as a cheap lens; more reads as a fault. Spatial, so it costs time.

---

## Texture and optics

### `vignette`

*corner falloff, tối góc*

`0.0`–`1.0`. Darkens the corners, which pulls the eye to the middle. The mask
is built once per size and shared by every clip using that look.

### `grain`

*film noise, texture, nhiễu hạt, hạt phim*

`0.0`–`0.2`. Luminance noise. Real grain moves, so the field cycles through
eight phases — past the point where a viewer can see the loop. Between `0.10`
and `0.15` is the short-form texture; `0.03` is a film emulation.

### `softness`

*diffusion, blur, làm mềm, mờ nhẹ*

`0.0`–`1.0` blend toward a blurred copy. Old lenses were not sharp. Applied
during framing, where the image is still a PIL image and a blur is cheap.

---

## Presets

| Preset | What it is | Sits under |
|---|---|---|
| `none` | untouched | anything already graded |
| `vintage` | desaturated, warm, lifted, soft, vignetted | archive footage, openings |
| `faded` | a gentler vintage | the middle of an arc |
| `warm` | slight amber lift, barely there | daylight footage that needs help |
| `vivid` | rich and contrasty | the payoff clip, the landing |
| `opium` | near-monochrome, crushed, blooming | three-frame cuts, impact beats |
| `dirty` | noisy, split, blown | stuttered frame rates, bursts |

`opium` and `dirty` are heavy on purpose: a subtle grade under a three-frame
cut does not register at all.

---

## Combinations that work

**The slow open** — extract levels, a long push, no blur.

```json
{"start": 9.6, "end": 12.6, "motion": "zoom-in", "zoom": [1.02, 1.14],
 "grade": {"black": 0.14, "white": 0.9, "saturation": 0.42, "contrast": 1.1}}
```

**Backlit haze** — glow on footage that already has sun in it.

```json
{"motion": "pan-right", "zoom": 1.18, "pan": 0.45,
 "grade": {"preset": "warm", "glow": 0.42, "glow_threshold": 0.5}}
```

**The punch** — a short clip, a hard push, blur under it, hard cut in.

```json
{"start": 139.4, "end": 140.9, "transition": "cut",
 "motion": "zoom-in", "zoom": [1.0, 1.42], "shutter": 180}
```

**The glitch beat** — inverted join, stutter, dirty grade.

```json
{"transition": "invert-r", "stutter": 12, "grade": "dirty",
 "motion": "zoom-in", "zoom": 1.3, "shutter": 240}
```

**The landing** — slowed, pulled back, graded back into colour.

```json
{"speed": 0.8, "motion": "zoom-out", "zoom": [1.16, 1.0], "grade": "vivid"}
```

---

## Not in the catalogue

Three things from the short-form vocabulary are **not** reachable here,
because they need per-object segmentation and tracking rather than pixel
maths:

- masking a person out of the background
- locking a subject in frame while the world moves around it
- a silhouette glow traced around a body

An editor with a modern NLE has those. Faking them with pixel maths looks
wrong, so this does not try.

**Out-of-bounds is the half case.** [`out-of-bounds`](#out-of-bounds) draws the
border and breaks a rectangle of picture through it, which is everything about
the effect except the part that needs a subject mask. Its entry says plainly
what that costs.
