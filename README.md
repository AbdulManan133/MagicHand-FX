# Hand Particles

Real-time hand-controlled particle system using your webcam.
Glowing particle clouds follow your hands on a dark background, split per hand, and react to every gesture.
Hide your hands → particles burst wide and fall across the floor.
Show your hands again → they reform around each hand instantly.

---

## Files

| File | Description |
|------|-------------|
| `main_back.py` | **Full-featured version** — skeleton overlay, snap/clap, optional **fist shockwaves** (`s`), gravity tilt, backgrounds, screen shake, ground-tint colors |
| `main.py` | Lightweight version — trails, black-hole pinch, wall physics, floor scatter |

Run the full version:

```bash
python main_back.py
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## All launch options (`main_back.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | Webcam index. Try `1` if your default doesn't open |
| `--width` | `960` | Output window width (see note below) |
| `--height` | `540` | Output window height |
| `--particles` | `3000` | Number of particles (lower if the CPU struggles) |
| `--track-width` | `640` | Hand-tracking frame width (lower = faster MediaPipe) |
| `--track-height` | `360` | Hand-tracking frame height (lower = faster MediaPipe) |
| `--detect-every` | `2` | Run MediaPipe every N frames (larger N = less CPU; hand pose updates less often) |
| `--demo` | off | No webcam — two animated fake hands, tests the window |
| `--vispy` | off | GPU markers via VisPy (requires `pip install PyQt5 vispy`); default is an OpenCV window |

**OpenCV rendering:** particle glow is composited at about **62%** of `--width` × `--height`, then scaled to the full window for performance. The live camera preview still uses your chosen resolution.

---

## Controls

### Keyboard (`main_back.py`)

| Key | Action |
|-----|--------|
| `b` | Cycle background mode (black → gradient → wave → circles) |
| `s` | Toggle **fist shockwaves** — when on, a closed fist emits expanding rings that **push** particles; the extra **ring glow** is drawn only in the **OpenCV** window (simulation still runs with `--vispy`) |
| `q` or `Esc` | Quit |

`main.py` may still expose extra keys (for example a color palette); this table matches **`main_back.py`**.

### Hand gestures

| Gesture | Action |
|---------|--------|
| Move hand | Particle cloud follows your hand |
| Two hands visible | Splits into two clouds — each hand owns its particles (left = orange, right = blue) |
| Pinch and hold (thumb + index close) | Black-hole mode — particles spiral inward, dark core, thin glowing ring |
| Fist (quick close) | Scatter pulse — particles burst outward, then draw back when you open your hand |
| Fist + **shockwaves on** (`s`) | While the fist stays closed, periodic **shock rings** push nearby particles and add a matching glow (OpenCV view) |
| Open hand after fist | Particles gather back to the hand(s) |
| Snap (thumb + index very close, quick) | Shockwave — nearby particles blast outward |
| Clap (both hands come together fast) | Big explosion — screen shake + particles blast in all directions |
| Tilt hand sideways | Gravity shifts to match hand orientation — particles fall in that direction |
| Hide one hand | Only particles near that hand scatter to the floor; the other hand keeps its cloud |
| Hide both hands | All particles burst wide and fall across the full floor width with wall bounces |
| Last hand to leave the screen | Free / floor particles use **that hand’s color** (orange = left, blue = right), so the “ground” cloud matches whoever vanished last |
| Show hands again | Particles reform around each hand |

---

## Visual features

- **Hand skeleton overlay** — wireframe drawn on every detected hand (OpenCV path)
- **Two fixed particle colors** — left ≈ orange, right ≈ blue (BGR `[255,150,100]` / `[100,150,255]`), matched to the skeleton tint; **no palette cycling**
- **Ground / free-particle tint** — when a hand is gone, particles that aren’t tied to a hand use the color of the **last hand that dropped off tracking**; scattered “floor” bits follow the same rule
- **Shooting-star trails** — velocity-aligned streaks in the OpenCV glow layer
- **Proximity dimming** — particles dim near the hand (suction / absorption look); VisPy and OpenCV paths both use it where applicable
- **Pinch black hole** — darker core + bright ring while pinch is held (OpenCV composite)
- **Layered bloom** — OpenCV path stacks Gaussian-blurred passes for a soft glow (tuned for speed)
- **Screen shake** — triggered by clap explosions, decays smoothly
- **Smart scatter** — when only one hand disappears, only particles near that hand fly off; the other cloud is unaffected

---

## Background modes (press `b`)

1. **black** — pure black (default, best contrast)
2. **gradient** — dark vignette overlay
3. **wave** — animated wave pattern
4. **circles** — animated concentric ring pattern

---

## Performance tips

- If lagging: lower particles and skip some detection frames, e.g. `--particles 1800 --width 640 --height 480 --detect-every 3` (or `4`)
- Smaller `--track-width` / `--track-height` speeds up MediaPipe with slightly rougher landmarks
- If the camera won't open: `--camera 1` or close Zoom / Teams / other apps using the camera
- Test without a webcam: `--demo`
- **VisPy** (`--vispy`) offloads marker drawing to the GPU but still runs the same Python simulation; try it if the OpenCV compositor is the bottleneck

---

## Debug log

If the app closes instantly with no message, open **`magic_hand_debug.log`** in the project directory (same folder as `main_back.py`).

Every run appends a timestamped trace. Full Python tracebacks are written there.

---

## Requirements

See `requirements.txt`:

| Package | Notes |
|---------|--------|
| `opencv-python` | Camera + default window + CPU glow composite |
| `mediapipe` | Hand landmarks |
| `numpy` | Particle simulation |
| `vispy` | Optional; GPU markers when you pass `--vispy` |
| `PyQt5` | Optional; common VisPy backend on Windows (or set `VISPY_BACKEND` if you use another) |
