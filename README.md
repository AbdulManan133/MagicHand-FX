# Hand Particles

Real-time hand-controlled particle simulation with webcam tracking.

The app supports:
- dual-hand particle ownership
- independent left/right color switching
- gesture-triggered effects (shield, pinch, snap, clap, gravity tilt)
- optional demo mode (no webcam)

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point that launches `magic_hand.py` |
| `magic_hand.py` | Main full-feature app (webcam + demo + gestures + effects) |
| `demo.py` | Demo-only launcher (no webcam path) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Full app:

```bash
python main.py
```

Demo mode from full app:

```bash
python main.py --demo
```

Demo-only launcher:

```bash
python demo.py
```

## Launch options (`magic_hand.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | Webcam index. Try `1` if needed |
| `--width` | `960` | Output window width |
| `--height` | `540` | Output window height |
| `--particles` | `3000` | Particle count |
| `--track-width` | `640` | Tracking frame width |
| `--track-height` | `360` | Tracking frame height |
| `--detect-every` | `2` | Run detection every N frames |
| `--demo` | off | Use fake hands (no webcam) |
| `--vispy` | off | Use VisPy GPU markers |

## Keyboard controls

| Key | Action |
|-----|--------|
| `e` | Cycle **left-hand** color |
| `r` | Cycle **right-hand** color |
| `s` | Toggle peace-shockwaves |
| `q` or `Esc` | Quit |

Notes:
- Left and right colors are independent.
- The app prevents both hands from selecting the same color at once.
- Color list includes 10+ variants, including `red` and brighter `black`.

## Gesture highlights

| Gesture | Effect |
|---------|--------|
| Move hand | Particles follow hand |
| Open 4 fingers | Shield layer around hand |
| Index raised + pointing | Shoots orb projectiles from fingertip |
| Pinch | Black-hole pull effect |
| Snap | Local shockwave |
| Clap | Explosion + screen shake |
| Tilt hand | Gravity direction shifts |
| Hide hands | Scatter/fall behavior, then reform on return |

## Visual notes

- Deep blue gradient background with soft cyan glow
- Multi-layer bloom and particle trails
- Per-hand skeleton overlay
- Ground/free particles tint to last tracked hand side

## Performance tips

- Lower load if needed: `--particles 1800 --width 640 --height 480 --detect-every 3`
- Lower tracking size for speed: reduce `--track-width` / `--track-height`
- If webcam fails: try `--camera 1` and close apps using the camera

## Debug log

If the app closes instantly, open `magic_hand_debug.log` in the project folder.

## Requirements

From `requirements.txt`:

- `opencv-python`
- `mediapipe`
- `numpy`
- `vispy` (optional)
- `PyQt5` (optional)
