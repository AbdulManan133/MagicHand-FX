import argparse

import os

import sys

import time

import traceback



import cv2

import mediapipe as mp

import numpy as np



_DEBUG_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "magic_hand_debug.log")





def _log(msg: str) -> None:

    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg + "\n"

    try:

        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:

            f.write(line)

    except OSError:

        pass





def _announce(msg: str) -> None:

    """Print to stdout; stderr may be invalid on Windows without a console."""

    print(msg, flush=True)

    try:

        if sys.stderr and sys.stderr.fileno() >= 0:

            print(msg, file=sys.stderr, flush=True)

    except (OSError, AttributeError):

        pass

    _log(msg)





def _global_excepthook(exc_type, exc, tb):

    _log("FATAL: " + "".join(traceback.format_exception(exc_type, exc, tb)))

    sys.__excepthook__(exc_type, exc, tb)





sys.excepthook = _global_excepthook





def _try_vispy_backend() -> bool:

    """VisPy needs PyQt5, GLFW, Pyglet, etc. or it exits with no window."""

    import vispy.app as va



    if name := os.environ.get("VISPY_BACKEND"):

        try:

            va.use_app(name)

            return True

        except Exception:

            pass

    for b in ("pyqt5", "PyQt5", "pyside2", "glfw", "pyglet"):

        try:

            va.use_app(b)

            return True

        except Exception:

            continue

    return False





def landmarks_to_points(landmark_list, tw: int, th: int, rw: int, rh: int) -> np.ndarray:

    sx, sy = rw / float(tw), rh / float(th)

    return np.array(

        [[p.x * tw * sx, p.y * th * sy] for p in landmark_list],

        dtype=np.float32,

    )





def parse_hands(results, tw: int, th: int, rw: int, rh: int) -> dict:

    """Return {'Left': ndarray(21,2), 'Right': ...}  (may be empty)."""

    out = {}

    if not results.multi_hand_landmarks:

        return out

    mh = getattr(results, "multi_handedness", None)

    try:

        n_lbl = len(mh) if mh is not None else 0

    except TypeError:

        n_lbl = 0

    for i, hand_lms in enumerate(results.multi_hand_landmarks):

        try:

            if mh is not None and n_lbl > i:

                lab = mh[i].classification[0].label

            else:

                lab = "Left" if i == 0 else "Right"

        except (IndexError, AttributeError, TypeError):

            lab = "Left" if i == 0 else "Right"

        out[lab] = landmarks_to_points(hand_lms.landmark, tw, th, rw, rh)

    return out





def hand_center(pts: np.ndarray) -> np.ndarray:

    if pts is None or len(pts) < 21:

        return pts.mean(0).astype(np.float32)

    tips = pts[[4, 8, 12, 16, 20]]

    return (0.62 * tips.mean(0) + 0.38 * pts[0]).astype(np.float32)




def hand_pinch_fist(pts: np.ndarray):

    if pts is None or len(pts) < 21:

        return False, False

    wrist, tips = pts[0], pts[[4, 8, 12, 16, 20]]

    pinch = float(np.linalg.norm(pts[4] - pts[8])) < 42.0

    # Guard against false fist when index is clearly raised/pointing.
    idx_ext = _finger_extended(pts, 8, 6, 5, wrist)
    mid_ext = _finger_extended(pts, 12, 10, 9, wrist)
    ring_ext = _finger_extended(pts, 16, 14, 13, wrist)
    pinky_ext = _finger_extended(pts, 20, 18, 17, wrist)
    extended_count = int(idx_ext) + int(mid_ext) + int(ring_ext) + int(pinky_ext)

    base_fist = float(np.mean(np.linalg.norm(tips - wrist, axis=1))) < 118.0
    fist = bool(base_fist and (not idx_ext) and extended_count <= 1)

    return pinch, fist





def hand_snap(pts: np.ndarray):

    """Detect snap: thumb + index very close."""

    if pts is None or len(pts) < 21:

        return False

    thumb_tip = pts[4]

    index_tip = pts[8]

    snap_dist = float(np.linalg.norm(thumb_tip - index_tip))

    return snap_dist < 35.0





def hand_gravity_tilt(pts: np.ndarray):

    """Return (is_tilted, tilt_angle) based on hand orientation."""

    if pts is None or len(pts) < 21:

        return False, 0.0

    wrist = pts[0]

    mid_base = pts[9]

    direction = mid_base - wrist

    angle = np.arctan2(direction[1], direction[0])

    is_tilted = abs(angle) > np.pi / 6

    return is_tilted, angle


def hand_four_fingers_open_score(pts: np.ndarray) -> float:

    """Return openness score in [0..1] for index/middle/ring/pinky extension."""

    if pts is None or len(pts) < 21:

        return 0.0

    wrist = pts[0]

    # (tip, pip, mcp) for index/middle/ring/pinky
    chains = ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17))

    extended = 0

    for tip_i, pip_i, mcp_i in chains:

        v1 = pts[tip_i] - pts[pip_i]   # PIP -> TIP
        v2 = pts[mcp_i] - pts[pip_i]   # PIP -> MCP
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-3 or n2 < 1e-3:
            continue
        # Straight finger => vectors are near opposite at the PIP joint.
        straightness = float(np.dot(v1, v2) / (n1 * n2))
        d_tip = float(np.linalg.norm(pts[tip_i] - wrist))
        d_mcp = float(np.linalg.norm(pts[mcp_i] - wrist))
        long_enough = d_tip > d_mcp * 1.22
        is_extended = (straightness < -0.52) and long_enough
        if is_extended:
            extended += 1

    # Hard gate: shield requires ring + pinky up, so peace sign cannot trigger it.
    ring_ext = _finger_extended(pts, 16, 14, 13, wrist)
    pinky_ext = _finger_extended(pts, 20, 18, 17, wrist)
    if not (ring_ext and pinky_ext):
        return 0.0

    if extended < 4:
        return 0.0

    # Keep full open palm out of shield, but avoid over-strict thumb folding.
    palm_center = (pts[5] + pts[9] + pts[13] + pts[17]) * 0.25
    palm_width = float(np.linalg.norm(pts[5] - pts[17])) + 1e-6

    thumb_tip_to_palm = float(np.linalg.norm(pts[4] - palm_center))
    thumb_joint_to_palm = float(np.linalg.norm(pts[2] - palm_center)) + 1e-6
    thumb_open_ratio = thumb_tip_to_palm / thumb_joint_to_palm

    # Side spread of thumb relative to palm width.
    thumb_lateral_ratio = float(np.linalg.norm(pts[4] - pts[5])) / palm_width

    # Depth-agnostic backup metric: thumb distance from wrist compared to thumb base.
    thumb_to_wrist_ratio = float(np.linalg.norm(pts[4] - wrist)) / (float(np.linalg.norm(pts[2] - wrist)) + 1e-6)

    # Only clearly open thumb should be hard-blocked as 5-finger open palm.
    if (thumb_open_ratio > 1.56) or (thumb_lateral_ratio > 0.97) or (thumb_to_wrist_ratio > 1.62):
        return 0.0

    # Otherwise score remains high enough for practical 4-finger shield activation.
    score = 1.0
    if thumb_open_ratio > 1.36:
        score -= 0.20
    if thumb_lateral_ratio > 0.84:
        score -= 0.18
    if thumb_to_wrist_ratio > 1.44:
        score -= 0.12

    return float(np.clip(score, 0.0, 1.0))


def _finger_extended(pts: np.ndarray, tip_i: int, pip_i: int, mcp_i: int, wrist: np.ndarray) -> bool:

    v1 = pts[tip_i] - pts[pip_i]
    v2 = pts[mcp_i] - pts[pip_i]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-3 or n2 < 1e-3:
        return False
    straightness = float(np.dot(v1, v2) / (n1 * n2))
    d_tip = float(np.linalg.norm(pts[tip_i] - wrist))
    d_mcp = float(np.linalg.norm(pts[mcp_i] - wrist))
    return (straightness < -0.50) and (d_tip > d_mcp * 1.18)


def hand_index_raised_score(pts: np.ndarray) -> float:

    """High score when only index is raised; low when other fingers are also open."""

    if pts is None or len(pts) < 21:
        return 0.0

    wrist = pts[0]
    idx_ext = _finger_extended(pts, 8, 6, 5, wrist)
    mid_ext = _finger_extended(pts, 12, 10, 9, wrist)
    ring_ext = _finger_extended(pts, 16, 14, 13, wrist)
    pinky_ext = _finger_extended(pts, 20, 18, 17, wrist)

    # Finger should also point away from wrist direction.
    idx_tip = pts[8]
    idx_pip = pts[6]
    idx_long = float(np.linalg.norm(idx_tip - wrist)) > float(np.linalg.norm(idx_pip - wrist)) * 1.25

    score = 1.0 if (idx_ext and idx_long) else 0.0
    if mid_ext:
        score -= 0.34
    if ring_ext:
        score -= 0.34
    if pinky_ext:
        score -= 0.34

    # Penalize very open thumb (looks like full open hand).
    palm_center = (pts[5] + pts[9] + pts[13] + pts[17]) * 0.25
    thumb_tip_to_palm = float(np.linalg.norm(pts[4] - palm_center))
    thumb_joint_to_palm = float(np.linalg.norm(pts[2] - palm_center))
    if thumb_tip_to_palm > (thumb_joint_to_palm * 1.35):
        score -= 0.18

    return float(np.clip(score, 0.0, 1.0))


def hand_index_direction(pts: np.ndarray, fallback: np.ndarray) -> np.ndarray:

    """Normalized shooting direction based on index finger axis (PIP->TIP)."""

    if pts is None or len(pts) < 21:
        v = fallback.astype(np.float32)
        n = float(np.linalg.norm(v))
        return (v / n) if n > 1e-3 else np.array([1.0, 0.0], dtype=np.float32)

    tip = pts[8].astype(np.float32)
    pip = pts[6].astype(np.float32)
    mcp = pts[5].astype(np.float32)
    # Blend local finger direction with base-to-tip direction for stability.
    v = 0.72 * (tip - pip) + 0.28 * (tip - mcp)
    n = float(np.linalg.norm(v))
    if n < 1e-3:
        v = fallback.astype(np.float32)
        n = float(np.linalg.norm(v))
    return (v / max(n, 1e-3)).astype(np.float32)


def hand_peace_sign_score(pts: np.ndarray) -> float:

    """High score when index+middle are up and ring+pinky are down (V sign)."""

    if pts is None or len(pts) < 21:
        return 0.0

    wrist = pts[0]
    idx_ext = _finger_extended(pts, 8, 6, 5, wrist)
    mid_ext = _finger_extended(pts, 12, 10, 9, wrist)
    ring_ext = _finger_extended(pts, 16, 14, 13, wrist)
    pinky_ext = _finger_extended(pts, 20, 18, 17, wrist)

    # Distance between index and middle tips helps distinguish V sign.
    v_gap = float(np.linalg.norm(pts[8] - pts[12]))
    palm_w = float(np.linalg.norm(pts[5] - pts[17])) + 1e-3
    gap_ok = v_gap > palm_w * 0.20

    # Hard gate: both index and middle must be clearly extended.
    if not (idx_ext and mid_ext):
        return 0.0

    score = 0.90
    if gap_ok:
        score += 0.15
    if ring_ext:
        score -= 0.45
    if pinky_ext:
        score -= 0.45

    # Penalize very open thumb to avoid accidental 4-5 finger hand poses.
    palm_center = (pts[5] + pts[9] + pts[13] + pts[17]) * 0.25
    thumb_tip_to_palm = float(np.linalg.norm(pts[4] - palm_center))
    thumb_joint_to_palm = float(np.linalg.norm(pts[2] - palm_center))
    if thumb_tip_to_palm > (thumb_joint_to_palm * 1.45):
        score -= 0.14

    return float(np.clip(score, 0.0, 1.0))


def hand_thumb_out_score(pts: np.ndarray) -> float:

    """High score when thumb is out while other fingers are mostly folded."""

    if pts is None or len(pts) < 21:
        return 0.0

    wrist = pts[0]
    idx_ext = _finger_extended(pts, 8, 6, 5, wrist)
    mid_ext = _finger_extended(pts, 12, 10, 9, wrist)
    ring_ext = _finger_extended(pts, 16, 14, 13, wrist)
    pinky_ext = _finger_extended(pts, 20, 18, 17, wrist)
    extended_non_thumb = int(idx_ext) + int(mid_ext) + int(ring_ext) + int(pinky_ext)

    palm_center = (pts[5] + pts[9] + pts[13] + pts[17]) * 0.25
    thumb_tip_to_palm = float(np.linalg.norm(pts[4] - palm_center))
    thumb_joint_to_palm = float(np.linalg.norm(pts[2] - palm_center)) + 1e-6
    thumb_open_ratio = thumb_tip_to_palm / thumb_joint_to_palm

    thumb_to_wrist_ratio = float(np.linalg.norm(pts[4] - wrist)) / (float(np.linalg.norm(pts[2] - wrist)) + 1e-6)
    palm_width = float(np.linalg.norm(pts[5] - pts[17])) + 1e-6
    thumb_lateral_ratio = float(np.linalg.norm(pts[4] - pts[5])) / palm_width

    thumb_out = (thumb_open_ratio > 1.25) or (thumb_to_wrist_ratio > 1.30) or (thumb_lateral_ratio > 0.72)
    if not thumb_out:
        return 0.0

    score = 1.0
    if idx_ext:
        score -= 0.26
    if mid_ext:
        score -= 0.26
    if ring_ext:
        score -= 0.26
    if pinky_ext:
        score -= 0.26
    if extended_non_thumb >= 3:
        score = 0.0
    return float(np.clip(score, 0.0, 1.0))





class WaterFlow:

    def __init__(self, width: int, height: int, n: int = 30000):

        self.w = width

        self.h = height

        self.n = n

        self.focal = float(min(width, height)) * 0.92



        cx, cy = width * 0.5, height * 0.5

        self.hand_L = np.array([cx - 120, cy], dtype=np.float32)

        self.hand_R = np.array([cx + 120, cy], dtype=np.float32)

        self.prev_L = self.hand_L.copy()

        self.prev_R = self.hand_R.copy()

        self.vel_L = np.zeros(2, dtype=np.float32)

        self.vel_R = np.zeros(2, dtype=np.float32)

        self.smooth_speed = 0.0



        self.pos = np.zeros((n, 2), dtype=np.float32)

        self.pos[:] = (self.hand_L + self.hand_R) * 0.5

        self.pos += np.random.normal(0, 32.0, (n, 2)).astype(np.float32)

        self.prev_pos = self.pos.copy()

        self.vel = np.random.normal(0, 0.5, (n, 2)).astype(np.float32)

        self.z = np.random.uniform(0.0, 300.0, n).astype(np.float32)

        self.vel_z = np.zeros(n, dtype=np.float32)



        self.seed = np.random.uniform(0.0, np.pi * 2.0, n).astype(np.float32)

        self.u = np.zeros(n, dtype=np.float32)

        self.head_theta = np.random.uniform(0.0, np.pi * 2.0, n).astype(np.float32)

        self.head_r_norm = np.sqrt(np.random.uniform(0.0, 1.0, n)).astype(np.float32)



        n_left = n // 2

        self.mask_left = np.zeros(n, dtype=bool)

        self.mask_left[:n_left] = True

        self.mask_right = ~self.mask_left



        self.scatter_mode = False

        self.prev_fist = False

        self.prev_n_hands = 0

        self.prev_has_L = False  # Track which hands were open

        self.prev_has_R = False



        # Last hand that left tracking: 'L' or 'R' — free / ground particles use that hand's BGR

        self._ground_tint_side = "L"



        self.t = 0.0

        self.fps = 0.0

        self.last_time = time.time()

        self.pinch_active = False

        self._bh_smooth = 0.0

        self._render_target_xy = np.zeros((n, 2), dtype=np.float32)

        self._has_render_targets = False



        self.bg_mode = 0

        self.bg_modes = ["black", "gradient", "wave", "circles"]



        # ===== HAND COMBAT & ENERGY BALL SYSTEM =====

        self.show_skeleton = True  # Always show skeleton for energy ball effect

        self.skeleton_line_width = 4  # Thick skeleton lines

        

        # Track which hand owns each particle (0=none, 1=left, 2=right)

        self.particle_owner = np.zeros(n, dtype=np.uint8)

        # Particle colors tied to hand

        self.particle_hand_color = np.zeros((n, 3), dtype=np.uint8)

        self.element_variants = [

            {"name": "ember", "bgr": [50, 120, 255]},

            {"name": "frost", "bgr": [100, 150, 255]},

            {"name": "toxic", "bgr": [110, 255, 130]},

            {"name": "magenta", "bgr": [255, 110, 220]},

            {"name": "gold", "bgr": [80, 210, 255]},

            {"name": "mint", "bgr": [170, 255, 170]},

            {"name": "cobalt", "bgr": [255, 140, 70]},

            {"name": "rose", "bgr": [180, 120, 255]},

            {"name": "aqua", "bgr": [255, 230, 120]},

            {"name": "lime", "bgr": [90, 255, 200]},

            {"name": "sunset", "bgr": [120, 180, 255]},

            {"name": "violet", "bgr": [255, 120, 180]},

            {"name": "red", "bgr": [0, 0, 255]},

            {"name": "black", "bgr": [34, 34, 34]},

        ]

        self.element_mode_idx_L = 0
        self.element_mode_idx_R = 1

        

        # Hand positions for energy ball throwing

        self.hand_L_skeleton = None

        self.hand_R_skeleton = None

        self.hand_throw_power_L = 0.0  # How fast left hand is moving

        self.hand_throw_power_R = 0.0  # How fast right hand is moving

        self.shield_active_L = False

        self.shield_active_R = False

        self.shield_strength_L = 0.0

        self.shield_strength_R = 0.0

        self.shield_confidence_L = 0.0

        self.shield_confidence_R = 0.0

        self.shield_particles_max = 760
        self.shield_pos_L = np.zeros((self.shield_particles_max, 2), dtype=np.float32)
        self.shield_pos_R = np.zeros((self.shield_particles_max, 2), dtype=np.float32)
        self.shield_prev_L = np.zeros((self.shield_particles_max, 2), dtype=np.float32)
        self.shield_prev_R = np.zeros((self.shield_particles_max, 2), dtype=np.float32)
        self.shield_vel_L = np.zeros((self.shield_particles_max, 2), dtype=np.float32)
        self.shield_vel_R = np.zeros((self.shield_particles_max, 2), dtype=np.float32)
        self.shield_phase_L = np.random.uniform(0.0, np.pi * 2.0, self.shield_particles_max).astype(np.float32)
        self.shield_phase_R = np.random.uniform(0.0, np.pi * 2.0, self.shield_particles_max).astype(np.float32)
        self.shield_seeded_L = False
        self.shield_seeded_R = False

        

        # Environmental forces

        self.wind_force = np.zeros(2, dtype=np.float32)

        self.wind_angle = 0.0

        self.vortex_strength = 0.0

        

        # ===== NEW GESTURE EFFECTS =====

        # Snap gesture

        self.snap_active_L = False

        self.snap_active_R = False

        self.snap_cooldown_L = 0.0

        self.snap_cooldown_R = 0.0

        

        # Clap gesture

        self.clap_active = True

        self.clap_cooldown = 0.0

        

        # Collision explosions

        self.explosion_particles = np.zeros(n, dtype=bool)

        self.explosion_age = np.zeros(n, dtype=np.float32)

        

        # Vortex & effects

        self.vortex_intensity_L = 0.0

        self.vortex_intensity_R = 0.0

        self.gravity_direction = np.array([0.0, 1.0], dtype=np.float32)

        self.time_slow = 0.0

        self.screen_shake = 0.0

        

        # Hand combat: when hands are close, particles get pulled to center

        self.hand_distance = 999.0

        self.combat_intensity = 0.0  # How intense the hand fight is
        # ===== FIST SHOCKWAVE SYSTEM =====
        # Each wave: [cx, cy, radius, alpha, born_time]
        self.fist_waves = []
        self.fist_wave_timer_L = 0.0   # countdown to next wave from left fist
        self.fist_wave_timer_R = 0.0   # countdown to next wave from right fist
        self.fist_core_L = 0.0         # smooth 0-1 energy buildup for left fist glow
        self.fist_core_R = 0.0         # smooth 0-1 energy buildup for right fist glow
        self.fist_center_L = np.array([0.0, 0.0], dtype=np.float32)
        self.fist_center_R = np.array([0.0, 0.0], dtype=np.float32)
        self.fist_L = False
        self.fist_R = False
        # Press 's' to arm: only then do closed-fist shockwaves spawn, push, and render
        self.fist_shockwaves_armed = False
        self.peace_conf_L = 0.0
        self.peace_conf_R = 0.0
        self.peace_active_L = False
        self.peace_active_R = False

        # ===== INDEX ORB ATTACK SYSTEM =====
        self.index_conf_L = 0.0
        self.index_conf_R = 0.0
        self.index_active_L = False
        self.index_active_R = False
        self.index_tip_L = self.hand_L.copy()
        self.index_tip_R = self.hand_R.copy()
        self.orb_cooldown_L = 0.0
        self.orb_cooldown_R = 0.0
        self.orbs = []
        self.orb_max = 56
        self.thumb_conf_L = 0.0
        self.thumb_conf_R = 0.0
        self.thumb_out_active_L = False
        self.thumb_out_active_R = False
        self.prev_thumb_out_L = False
        self.prev_thumb_out_R = False

    def _side_bgr_u8(self, side: str) -> np.ndarray:

        idx = self.element_mode_idx_L if side == "L" else self.element_mode_idx_R

        return np.array(self.element_variants[idx]["bgr"], dtype=np.uint8)

    def _side_bgr_tuple(self, side: str):

        bgr = self._side_bgr_u8(side)

        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _side_variant_name(self, side: str) -> str:

        idx = self.element_mode_idx_L if side == "L" else self.element_mode_idx_R

        return str(self.element_variants[idx]["name"])

    def cycle_element_mode(self, side: str = None):

        if side == "L":
            next_idx = (self.element_mode_idx_L + 1) % len(self.element_variants)
            guard = 0
            while next_idx == self.element_mode_idx_R and guard < len(self.element_variants):
                next_idx = (next_idx + 1) % len(self.element_variants)
                guard += 1
            self.element_mode_idx_L = next_idx
            return
        if side == "R":
            next_idx = (self.element_mode_idx_R + 1) % len(self.element_variants)
            guard = 0
            while next_idx == self.element_mode_idx_L and guard < len(self.element_variants):
                next_idx = (next_idx + 1) % len(self.element_variants)
                guard += 1
            self.element_mode_idx_R = next_idx
            return
        self.cycle_element_mode("L")
        self.cycle_element_mode("R")

    def _spawn_attack_orb(self, side: str, start_xy: np.ndarray, direction_xy: np.ndarray):

        if len(self.orbs) >= self.orb_max:
            self.orbs.pop(0)

        v = direction_xy.astype(np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-3:
            v = np.array([1.0, 0.0], dtype=np.float32)
            n = 1.0
        v = v / n
        self.orbs.append(
            {
                "x": float(start_xy[0]),
                "y": float(start_xy[1]),
                "vx": float(v[0] * 24.0),
                "vy": float(v[1] * 24.0),
                "life": 1.05,
                "age": 0.0,
                "seed": float(np.random.uniform(0.0, np.pi * 2.0)),
                "side": side,
            }
        )

    def _update_attack_orbs(self, dt: float):

        if not self.orbs:
            return

        alive = []
        for ob in self.orbs:
            ob["age"] += dt
            # Keep course; tiny drag avoids excessive speed while preserving pointer aim.
            ob["vx"] *= 0.998
            ob["vy"] *= 0.998

            ob["x"] += ob["vx"]
            ob["y"] += ob["vy"]
            ob["life"] -= dt

            if ob["x"] < -20 or ob["x"] > self.w + 20 or ob["y"] < -20 or ob["y"] > self.h + 20:
                continue
            if ob["life"] <= 0.0:
                continue

            center = np.array([ob["x"], ob["y"]], dtype=np.float32)
            d = np.linalg.norm(self.pos - center[None, :], axis=1)
            hit = d < 54.0
            if np.any(hit):
                delta = self.pos[hit] - center[None, :]
                mag = np.linalg.norm(delta, axis=1, keepdims=True)
                mag = np.maximum(mag, 0.1)
                force = (1.0 - np.clip(mag[:, 0] / 54.0, 0.0, 1.0)) * 6.2
                self.vel[hit] += (delta / mag) * force[:, None]

            alive.append(ob)

        self.orbs = alive




    def draw_hand_skeleton(self, frame, hand_pts, color=(200, 200, 100), hand_id=0):

        """Draw hand skeleton wireframe with thick glowing lines."""

        if hand_pts is None or len(hand_pts) < 21:

            return

        # Full finger connections (MediaPipe hand structure)

        connections = [

            (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb

            (0, 5), (5, 6), (6, 7), (7, 8),    # Index

            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle

            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring

            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky

            (5, 9), (9, 13), (13, 17),         # Palm

        ]

        # Draw thicker glowing skeleton lines

        for (start, end) in connections:

            pt1 = tuple(hand_pts[start].astype(int))

            pt2 = tuple(hand_pts[end].astype(int))

            cv2.line(frame, pt1, pt2, color, 4)

            # Add glow effect with thinner layer

            cv2.line(frame, pt1, pt2, tuple([int(c * 0.6) for c in color]), 2)

        # Draw joints

        for pt in hand_pts:

            cv2.circle(frame, tuple(pt.astype(int)), 5, color, -1)

    def draw_hand_particle_shield(self, frame: np.ndarray, hand_pts: np.ndarray, side: str, strength: float):

        """Render a clean, fast particle membrane that covers the hand area."""

        if hand_pts is None or len(hand_pts) < 21 or strength < 0.05:

            return

        h, w = frame.shape[:2]

        hull = cv2.convexHull(hand_pts.astype(np.int32))

        if hull is None or len(hull) < 3:

            return

        x, y, bw, bh = cv2.boundingRect(hull)
        pad = int(16 + 10 * strength)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        if x1 <= x0 or y1 <= y0:
            return

        hull_local = hull.copy()
        hull_local[:, 0, 0] -= x0
        hull_local[:, 0, 1] -= y0

        roi = frame[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull_local, 255)

        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return

        # Per-hand persistent micro-particles, constrained to hand ROI.
        if side == "L":
            spos = self.shield_pos_L
            sprev = self.shield_prev_L
            svel = self.shield_vel_L
            sphase = self.shield_phase_L
            seeded = self.shield_seeded_L
        else:
            spos = self.shield_pos_R
            sprev = self.shield_prev_R
            svel = self.shield_vel_R
            sphase = self.shield_phase_R
            seeded = self.shield_seeded_R

        area = xs.size
        base_count = int(np.clip(area / 17.0, 220, self.shield_particles_max))
        p_count = int(base_count * (0.65 + 0.50 * strength))
        # Adaptive budget: keep shield dense, but protect frame rate if needed.
        fps_factor = 1.0
        if self.fps > 1.0:
            fps_factor = float(np.clip((self.fps - 20.0) / 18.0, 0.62, 1.0))
        p_count = int(p_count * fps_factor)
        p_count = max(180, min(p_count, self.shield_particles_max))

        if not seeded:
            seed_idx = np.random.randint(0, xs.size, p_count)
            spos[:p_count, 0] = xs[seed_idx].astype(np.float32) + x0
            spos[:p_count, 1] = ys[seed_idx].astype(np.float32) + y0
            sprev[:p_count] = spos[:p_count]
            svel[:p_count] = 0.0
            if side == "L":
                self.shield_seeded_L = True
            else:
                self.shield_seeded_R = True

        pts = spos[:p_count]
        prv = sprev[:p_count]
        vel = svel[:p_count]
        phase = sphase[:p_count]

        center = hand_center(hand_pts).astype(np.float32)
        prv[:] = pts

        to_center = center[None, :] - pts
        d = np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-3
        dir_c = to_center / d
        tang = np.column_stack([-dir_c[:, 1], dir_c[:, 0]])

        noise = np.random.normal(0.0, 0.50 + 0.35 * strength, (p_count, 2)).astype(np.float32)
        vel[:] = vel * 0.79 + dir_c * (0.28 + 0.30 * strength) + tang * (0.56 + 0.34 * strength) + noise * 0.19
        spd = np.linalg.norm(vel, axis=1, keepdims=True)
        fast = spd[:, 0] > 7.5
        if np.any(fast):
            vel[fast] = vel[fast] / np.maximum(spd[fast], 1e-3) * 7.5
        pts[:] = pts + vel

        lx = (pts[:, 0] - x0).astype(np.int32)
        ly = (pts[:, 1] - y0).astype(np.int32)
        in_bounds = (lx >= 0) & (lx < rw) & (ly >= 0) & (ly < rh)
        inside = np.zeros(p_count, dtype=bool)
        inside[in_bounds] = mask[ly[in_bounds], lx[in_bounds]] > 0
        outside = ~inside
        if np.any(outside):
            resp_idx = np.random.randint(0, xs.size, int(outside.sum()))
            pts[outside, 0] = xs[resp_idx].astype(np.float32) + x0
            pts[outside, 1] = ys[resp_idx].astype(np.float32) + y0
            vel[outside] *= 0.15

        # Draw as dense, protective particle armor (not flat transparent light).
        layer = np.zeros((rh, rw, 3), dtype=np.float32)
        bgr = np.array(self._side_bgr_tuple(side), dtype=np.float32)

        lx = np.clip((pts[:, 0] - x0).astype(np.int32), 0, rw - 1)
        ly = np.clip((pts[:, 1] - y0).astype(np.int32), 0, rh - 1)
        idx = ly * rw + lx

        twinkle = (0.60 + 0.40 * np.sin(self.t * 10.0 + phase)).astype(np.float32)
        speed_bright = np.clip(np.linalg.norm(pts - prv, axis=1) * 0.45, 0.25, 1.0).astype(np.float32)
        w0 = twinkle * speed_bright * (0.95 + 0.60 * strength)

        size = rw * rh
        layer[:, :, 0] += np.bincount(idx, weights=bgr[0] * w0, minlength=size).reshape(rh, rw)
        layer[:, :, 1] += np.bincount(idx, weights=bgr[1] * w0, minlength=size).reshape(rh, rw)
        layer[:, :, 2] += np.bincount(idx, weights=bgr[2] * w0, minlength=size).reshape(rh, rw)

        # Short streaks from previous positions add a protective flowing feel.
        mx = np.clip(((pts[:, 0] + prv[:, 0]) * 0.5 - x0).astype(np.int32), 0, rw - 1)
        my = np.clip(((pts[:, 1] + prv[:, 1]) * 0.5 - y0).astype(np.int32), 0, rh - 1)
        mid_idx = my * rw + mx
        w1 = w0 * 0.52
        layer[:, :, 0] += np.bincount(mid_idx, weights=bgr[0] * w1, minlength=size).reshape(rh, rw)
        layer[:, :, 1] += np.bincount(mid_idx, weights=bgr[1] * w1, minlength=size).reshape(rh, rw)
        layer[:, :, 2] += np.bincount(mid_idx, weights=bgr[2] * w1, minlength=size).reshape(rh, rw)

        # Subtle membrane fill keeps the hand fully covered while particles remain dominant.
        soft = cv2.GaussianBlur(mask, (0, 0), 3.2).astype(np.float32) / 255.0
        membrane = soft * (0.10 + 0.14 * strength)
        layer += membrane[:, :, None] * bgr[None, None, :]

        glow = cv2.GaussianBlur(layer, (0, 0), 1.7 + 1.2 * strength)
        comp = np.clip(layer * 0.92 + glow * 1.08, 0, 255).astype(np.uint8)
        alpha = float(0.56 + 0.28 * strength)
        cv2.addWeighted(comp, alpha, roi, 1.0, 0.0, dst=roi)



    def check_particle_collisions(self):

        """Particle-particle collisions (disabled for speed)."""

        pass



    def apply_environmental_forces(self, n_hands):

        """Environmental forces disabled for speed."""

        pass



    def gesture_fountain_effect(self):

        """Thumbs up → fountain spray effect."""

        if self.gesture_start_pos is None:

            return

        center = self.gesture_start_pos.get("thumb", np.array([self.w * 0.5, self.h * 0.5]))

        n_spray = min(150, self.n // 20)

        idx = np.random.choice(self.n, n_spray, replace=False)

        for i in idx:

            angle = np.random.uniform(0, 2 * np.pi)

            speed = np.random.uniform(15, 35)

            self.pos[i] = center.copy()

            self.vel[i, 0] = np.cos(angle) * speed

            self.vel[i, 1] = -np.abs(np.sin(angle)) * speed



    def apply_snap_shockwave(self, hand_pos, is_left=True):

        """Shockwave from snap gesture - scatter particles outward."""

        dist_to_particles = np.linalg.norm(self.pos - hand_pos[np.newaxis, :], axis=1)

        nearby = dist_to_particles < 180.0

        if np.any(nearby):

            delta = self.pos[nearby] - hand_pos[np.newaxis, :]

            magnitude = np.linalg.norm(delta, axis=1, keepdims=True)

            magnitude = np.maximum(magnitude, 0.1)

            direction = delta / magnitude

            shockwave_power = 22.0 * (1.0 - np.clip(magnitude[:, 0] / 180.0, 0, 1))

            self.vel[nearby] += direction * (shockwave_power[:, np.newaxis])

            self.screen_shake = max(self.screen_shake, 3.0)



    def apply_clap_explosion(self, center_pos):

        """Massive explosion from hand clap."""

        dist_to_particles = np.linalg.norm(self.pos - center_pos[np.newaxis, :], axis=1)

        close = dist_to_particles < 250.0

        if np.any(close):

            delta = self.pos[close] - center_pos[np.newaxis, :]

            magnitude = np.linalg.norm(delta, axis=1, keepdims=True)

            magnitude = np.maximum(magnitude, 0.1)

            direction = delta / magnitude

            blast_power = 45.0 * (1.0 - np.clip(magnitude[:, 0] / 250.0, 0, 0.9))

            self.vel[close] += direction * (blast_power[:, np.newaxis])

            self.explosion_particles[close] = True

            self.explosion_age[close] = 0.0

            self.screen_shake = max(self.screen_shake, 8.0)



    def apply_vortex_forces(self):

        """Swirling vortex around each hand."""

        if self.vortex_intensity_L > 0.01:

            delta_L = self.pos - self.hand_L[np.newaxis, :]

            dist_L = np.linalg.norm(delta_L, axis=1, keepdims=True)

            dist_L = np.maximum(dist_L, 10.0)

            # Perpendicular direction (rotate 90 degrees)

            perp_L = np.column_stack([-delta_L[:, 1], delta_L[:, 0]])

            perp_L = perp_L / dist_L

            vortex_acc_L = perp_L * (self.vortex_intensity_L * 0.4)

            self.vel += vortex_acc_L

        

        if self.vortex_intensity_R > 0.01:

            delta_R = self.pos - self.hand_R[np.newaxis, :]

            dist_R = np.linalg.norm(delta_R, axis=1, keepdims=True)

            dist_R = np.maximum(dist_R, 10.0)

            perp_R = np.column_stack([-delta_R[:, 1], delta_R[:, 0]])

            perp_R = perp_R / dist_R

            vortex_acc_R = perp_R * (self.vortex_intensity_R * 0.4)

            self.vel += vortex_acc_R



    def apply_magnetic_repulsion(self):

        """Particles repelled when hands are far apart."""

        dist_hands = np.linalg.norm(self.hand_R - self.hand_L)

        if dist_hands > 300.0:

            # Hands far = repulsion

            left_owned = self.particle_owner == 1

            right_owned = self.particle_owner == 2

            

            # Left particles repelled from right hand

            if np.any(left_owned):

                delta_L = self.pos[left_owned] - self.hand_R[np.newaxis, :]

                mag_L = np.linalg.norm(delta_L, axis=1, keepdims=True)

                mag_L = np.maximum(mag_L, 1.0)

                dir_L = delta_L / mag_L

                repel_L = 0.08 * (1.0 - np.clip(mag_L[:, 0] / 400.0, 0, 1))

                self.vel[left_owned] += dir_L * (repel_L[:, np.newaxis])

            

            # Right particles repelled from left hand

            if np.any(right_owned):

                delta_R = self.pos[right_owned] - self.hand_L[np.newaxis, :]

                mag_R = np.linalg.norm(delta_R, axis=1, keepdims=True)

                mag_R = np.maximum(mag_R, 1.0)

                dir_R = delta_R / mag_R

                repel_R = 0.08 * (1.0 - np.clip(mag_R[:, 0] / 400.0, 0, 1))

                self.vel[right_owned] += dir_R * (repel_R[:, np.newaxis])



    def apply_gravity_shift(self):

        """Apply shifted gravity based on hand tilt."""

        # Gravity component in shifted direction

        gravity_mag = 3.8

        self.vel[:, 0] += self.gravity_direction[0] * gravity_mag

        self.vel[:, 1] += self.gravity_direction[1] * gravity_mag



    def render_background(self, frame: np.ndarray) -> np.ndarray:

        """Render a richer deep-blue gradient backdrop for better contrast."""

        h, w = frame.shape[:2]

        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]

        # Vertical navy-to-blue gradient.
        b = 72.0 + 58.0 * (1.0 - y)
        g = 28.0 + 24.0 * (1.0 - y)
        r = 12.0 + 8.0 * (1.0 - y)

        overlay = np.zeros((h, w, 3), dtype=np.float32)
        overlay[:, :, 0] = b
        overlay[:, :, 1] = g
        overlay[:, :, 2] = r

        # Soft cyan bloom near top center for a nicer blue atmosphere.
        glow = np.exp(-((x * 1.25) ** 2 + ((y - 0.22) * 2.6) ** 2)).astype(np.float32)
        overlay[:, :, 0] += glow * 32.0
        overlay[:, :, 1] += glow * 18.0

        overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
        return cv2.addWeighted(frame, 0.70, overlay_u8, 0.30, 0)



    def update(self, hands: dict):

        now = time.time()

        dt = max(0.006, min(0.040, now - self.last_time))

        self.last_time = now

        self.fps = 0.9 * self.fps + 0.1 / dt

        self.t += dt



        self.prev_pos[:] = self.pos



        has_L = "Left" in hands and hands["Left"] is not None

        has_R = "Right" in hands and hands["Right"] is not None

        n_hands = int(has_L) + int(has_R)

        pn = self.prev_n_hands

        pL = self.prev_has_L

        pR = self.prev_has_R

        if n_hands < pn:

            if pn == 2 and n_hands == 1:

                if has_L and not has_R:

                    self._ground_tint_side = "R"

                elif has_R and not has_L:

                    self._ground_tint_side = "L"

            elif pn == 1 and n_hands == 0:

                if pL and not pR:

                    self._ground_tint_side = "L"

                elif pR and not pL:

                    self._ground_tint_side = "R"



        pinch_any = False

        fist_any = False

        snap_L = False

        snap_R = False

        tilt_L = False

        tilt_R = False

        gravity_angle_L = 0.0

        gravity_angle_R = 0.0

        peace_score_L = 0.0

        peace_score_R = 0.0

        shield_L = False

        shield_R = False

        

        if has_L:

            p, f = hand_pinch_fist(hands["Left"])

            pinch_any = pinch_any or p

            fist_any = fist_any or f

            peace_score_L = hand_peace_sign_score(hands["Left"])

            snap_L = hand_snap(hands["Left"])

            tilt_L, gravity_angle_L = hand_gravity_tilt(hands["Left"])

            shield_L = hand_four_fingers_open_score(hands["Left"])
            self.index_tip_L = hands["Left"][8].astype(np.float32)
            idx_dir_L = hand_index_direction(hands["Left"], np.array([1.0, 0.0], dtype=np.float32))
        else:
            idx_dir_L = np.array([1.0, 0.0], dtype=np.float32)

        if has_R:

            p, f = hand_pinch_fist(hands["Right"])

            pinch_any = pinch_any or p

            fist_any = fist_any or f

            peace_score_R = hand_peace_sign_score(hands["Right"])

            snap_R = hand_snap(hands["Right"])

            tilt_R, gravity_angle_R = hand_gravity_tilt(hands["Right"])

            shield_R = hand_four_fingers_open_score(hands["Right"])
            self.index_tip_R = hands["Right"][8].astype(np.float32)
            idx_dir_R = hand_index_direction(hands["Right"], np.array([-1.0, 0.0], dtype=np.float32))
        else:
            idx_dir_R = np.array([-1.0, 0.0], dtype=np.float32)

        # Color changes are controlled by keyboard now: E for left, R for right.
        self.thumb_conf_L = 0.0
        self.thumb_conf_R = 0.0
        self.thumb_out_active_L = False
        self.thumb_out_active_R = False
        self.prev_thumb_out_L = False
        self.prev_thumb_out_R = False

        idx_score_L = hand_index_raised_score(hands["Left"]) if has_L else 0.0
        idx_score_R = hand_index_raised_score(hands["Right"]) if has_R else 0.0
        self.index_conf_L = self.index_conf_L * 0.80 + idx_score_L * 0.20
        self.index_conf_R = self.index_conf_R * 0.80 + idx_score_R * 0.20
        if self.index_active_L:
            self.index_active_L = self.index_conf_L > 0.48
        else:
            self.index_active_L = self.index_conf_L > 0.82
        if self.index_active_R:
            self.index_active_R = self.index_conf_R > 0.48
        else:
            self.index_active_R = self.index_conf_R > 0.82

        self.orb_cooldown_L = max(0.0, self.orb_cooldown_L - dt)
        self.orb_cooldown_R = max(0.0, self.orb_cooldown_R - dt)
        if has_L and self.index_active_L and self.orb_cooldown_L <= 0.0:
            self._spawn_attack_orb("L", self.index_tip_L.copy(), idx_dir_L)
            self.orb_cooldown_L = 0.12
        if has_R and self.index_active_R and self.orb_cooldown_R <= 0.0:
            self._spawn_attack_orb("R", self.index_tip_R.copy(), idx_dir_R)
            self.orb_cooldown_R = 0.12

        self.peace_conf_L = self.peace_conf_L * 0.80 + (peace_score_L if has_L else 0.0) * 0.20
        self.peace_conf_R = self.peace_conf_R * 0.80 + (peace_score_R if has_R else 0.0) * 0.20
        if self.peace_active_L:
            self.peace_active_L = self.peace_conf_L > 0.50
        else:
            self.peace_active_L = self.peace_conf_L > 0.84
        if self.peace_active_R:
            self.peace_active_R = self.peace_conf_R > 0.50
        else:
            self.peace_active_R = self.peace_conf_R > 0.84
        peace_any = self.peace_active_L or self.peace_active_R

        # Prevent overlap with peace sign only; index can coexist with open-hand shield pose.
        target_conf_L = float(shield_L) if (has_L and (not self.peace_active_L)) else 0.0
        target_conf_R = float(shield_R) if (has_R and (not self.peace_active_R)) else 0.0
        self.shield_confidence_L = self.shield_confidence_L * 0.80 + target_conf_L * 0.20
        self.shield_confidence_R = self.shield_confidence_R * 0.80 + target_conf_R * 0.20

        if self.shield_active_L:
            self.shield_active_L = self.shield_confidence_L > 0.40
        else:
            self.shield_active_L = self.shield_confidence_L > 0.76
        if self.shield_active_R:
            self.shield_active_R = self.shield_confidence_R > 0.40
        else:
            self.shield_active_R = self.shield_confidence_R > 0.76

        self.shield_strength_L = self.shield_strength_L * 0.86 + (self.shield_confidence_L if self.shield_active_L else 0.0) * 0.14
        self.shield_strength_R = self.shield_strength_R * 0.86 + (self.shield_confidence_R if self.shield_active_R else 0.0) * 0.14



        self.pinch_active = bool(pinch_any and n_hands > 0)

        self._bh_smooth = 0.88 * self._bh_smooth + 0.12 * (1.0 if self.pinch_active else 0.0)

        

        # Enhanced time-slow with pinch

        self.time_slow = 0.6 * self._bh_smooth if self.pinch_active else 0.0

        

        # Snap shockwave handling

        if snap_L and not self.snap_active_L and self.snap_cooldown_L <= 0:

            self.apply_snap_shockwave(hand_center(hands["Left"]), is_left=True)

            self.snap_active_L = True

            self.snap_cooldown_L = 0.4

        if snap_R and not self.snap_active_R and self.snap_cooldown_R <= 0:

            self.apply_snap_shockwave(hand_center(hands["Right"]), is_left=False)

            self.snap_active_R = True

            self.snap_cooldown_R = 0.4

        

        self.snap_cooldown_L -= dt

        self.snap_cooldown_R -= dt

        if not snap_L:

            self.snap_active_L = False

        if not snap_R:

            self.snap_active_R = False

        

        # Clap detection & explosion

        if n_hands == 2:

            hand_dist = np.linalg.norm(self.hand_R - self.hand_L)

            relative_vel = np.linalg.norm(self.vel_L - self.vel_R)

            if hand_dist < 60.0 and relative_vel > 12.0 and not self.clap_active and self.clap_cooldown <= 0:

                clap_center = (self.hand_L + self.hand_R) * 0.5

                self.apply_clap_explosion(clap_center)

                self.clap_active = True

                self.clap_cooldown = 0.6

        

        self.clap_cooldown -= dt

        if self.clap_active and (n_hands < 2 or np.linalg.norm(self.hand_R - self.hand_L) > 100.0):

            self.clap_active = False

        

        # Gravity shift based on hand tilt

        if tilt_L or tilt_R:

            angle_to_use = gravity_angle_L if tilt_L else gravity_angle_R

            self.gravity_direction = np.array([np.sin(angle_to_use), np.cos(angle_to_use)], dtype=np.float32)

        else:

            self.gravity_direction = np.array([0.0, 1.0], dtype=np.float32)

        

        # Vortex intensity based on hand speed

        hand_speed_L = np.linalg.norm(self.vel_L)

        hand_speed_R = np.linalg.norm(self.vel_R)

        self.vortex_intensity_L = 0.9 * self.vortex_intensity_L + 0.1 * min(hand_speed_L * 0.3, 4.0)

        self.vortex_intensity_R = 0.9 * self.vortex_intensity_R + 0.1 * min(hand_speed_R * 0.3, 4.0)

        

        # Decay screen shake

        self.screen_shake *= 0.92



        # Smooth hand targets

        if n_hands == 2:

            mode = "dual"

            cL = hand_center(hands["Left"])

            cR = hand_center(hands["Right"])

            self.prev_L[:] = self.hand_L

            self.prev_R[:] = self.hand_R

            self.hand_L = 0.72 * self.hand_L + 0.28 * cL

            self.hand_R = 0.72 * self.hand_R + 0.28 * cR

            self.vel_L = self.hand_L - self.prev_L

            self.vel_R = self.hand_R - self.prev_R

            c_single = (self.hand_L + self.hand_R) * 0.5

        elif n_hands == 1:

            mode = "single"

            pts = hands["Left"] if has_L else hands["Right"]

            c = hand_center(pts)

            self.prev_L[:] = self.hand_L

            self.hand_L = 0.72 * self.hand_L + 0.28 * c

            self.hand_R = self.hand_L.copy()

            self.vel_L = self.hand_L - self.prev_L

            self.vel_R = self.vel_L

            c_single = self.hand_L.copy()

            cL = cR = c_single

        else:

            mode = "none"

            cL = self.hand_L

            cR = self.hand_R

            c_single = (self.hand_L + self.hand_R) * 0.5



        raw_speed = float(np.linalg.norm(self.vel_L) + np.linalg.norm(self.vel_R)) * 0.5

        self.smooth_speed = 0.82 * self.smooth_speed + 0.18 * raw_speed



        # ===== HAND COMBAT & ENERGY BALL SYSTEM =====

        self.hand_throw_power_L = np.linalg.norm(self.vel_L)

        self.hand_throw_power_R = np.linalg.norm(self.vel_R)

        

        # Calculate hand distance for combat intensity

        if n_hands == 2:

            self.hand_distance = float(np.linalg.norm(self.hand_R - self.hand_L))

            self.combat_intensity = max(0, 1.0 - self.hand_distance / 400)

        else:

            self.hand_distance = 999.0

            self.combat_intensity = 0.0

        

        # ===== VECTORIZED HAND COMBAT & ENERGY BALL SYSTEM =====

        # Compute all particle distances at once (vectorized)

        delta_L = self.pos - self.hand_L[np.newaxis, :]

        delta_R = self.pos - self.hand_R[np.newaxis, :]

        dist_L = np.linalg.norm(delta_L, axis=1)  # (n,)

        dist_R = np.linalg.norm(delta_R, axis=1)  # (n,)

        

        closer_to_L = dist_L < dist_R

        

        if n_hands == 2:

            # Both hands: pull particles to closer hand

            self.particle_owner[:] = np.where(closer_to_L, 1, 2).astype(np.uint8)

            self.particle_hand_color[closer_to_L] = self._side_bgr_u8("L")

            self.particle_hand_color[~closer_to_L] = self._side_bgr_u8("R")

            

            # Vectorized force application

            safe_dist_L = np.maximum(dist_L, 0.1)

            safe_dist_R = np.maximum(dist_R, 0.1)

            

            # Attraction + throwing for left-assigned particles

            pull_L = 0.15 * np.maximum(0, 1.0 - dist_L / 300)

            throw_L = self.hand_throw_power_L * 0.02

            dir_L = delta_L / safe_dist_L[:, np.newaxis]

            self.vel[closer_to_L] += dir_L[closer_to_L] * (pull_L[closer_to_L, np.newaxis] + throw_L)

            

            # Attraction + throwing for right-assigned particles

            pull_R = 0.15 * np.maximum(0, 1.0 - dist_R / 300)

            throw_R = self.hand_throw_power_R * 0.02

            dir_R = delta_R / safe_dist_R[:, np.newaxis]

            self.vel[~closer_to_L] += dir_R[~closer_to_L] * (pull_R[~closer_to_L, np.newaxis] + throw_R)

        

        elif n_hands == 1:

            # Single hand: all particles absorb to that hand

            if has_L:

                self.particle_owner[:] = 1

                self.particle_hand_color[:] = self._side_bgr_u8("L")

                safe_dist = np.maximum(dist_L, 0.1)

                pull = 0.15 * np.maximum(0, 1.0 - dist_L / 300)

                throw = self.hand_throw_power_L * 0.02

                dir_L = delta_L / safe_dist[:, np.newaxis]

                self.vel += dir_L * (pull[:, np.newaxis] + throw)

            else:

                self.particle_owner[:] = 2

                self.particle_hand_color[:] = self._side_bgr_u8("R")

                safe_dist = np.maximum(dist_R, 0.1)

                pull = 0.15 * np.maximum(0, 1.0 - dist_R / 300)

                throw = self.hand_throw_power_R * 0.02

                dir_R = delta_R / safe_dist[:, np.newaxis]

                self.vel += dir_R * (pull[:, np.newaxis] + throw)

        

        else:

            self.particle_owner[:] = 0

            gb = self._side_bgr_u8(self._ground_tint_side)

            self.particle_hand_color[:] = gb

        

        # When one hand disappears, keep ownership of particles properly

        if n_hands == 1:

            if has_L and not has_R:

                # Right hand disappeared: convert right-owned particles to left ownership

                reassign_to_left = (self.particle_owner == 2)

                self.particle_owner[reassign_to_left] = 1

                self.particle_hand_color[reassign_to_left] = self._side_bgr_u8("L")

                # Scattered particles (owner=0) stay free

            elif has_R and not has_L:

                # Left hand disappeared: convert left-owned particles to right ownership

                reassign_to_right = (self.particle_owner == 1)

                self.particle_owner[reassign_to_right] = 2

                self.particle_hand_color[reassign_to_right] = self._side_bgr_u8("R")

                # Scattered particles (owner=0) stay free

        elif n_hands == 0:

            pass



        # ===== SMART HAND CLOSING: Only scatter particles CLOSE to the closing hand =====

        scatter_radius = 150  # Only scatter particles within this radius of hand

        

        # Both hands disappear

        if self.prev_n_hands > 0 and n_hands == 0:

            self.vel *= 0.90
            n_burst = min(self.n, 2200)
            idx = np.random.choice(self.n, n_burst, replace=False)
            phi = np.random.uniform(-np.pi, 0.0, n_burst).astype(np.float32)
            mag = np.random.uniform(24.0, 72.0, n_burst).astype(np.float32)
            bias_x = np.random.uniform(-self.w * 0.48, self.w * 0.48, n_burst).astype(np.float32)
            self.vel[idx, 0] = np.cos(phi) * mag + bias_x * 0.18
            self.vel[idx, 1] = np.abs(np.sin(phi)) * mag * 0.9 + np.random.uniform(6.0, 22.0, n_burst).astype(np.float32)

        # One hand closes but other is still open

        elif self.prev_has_L and not has_L and has_R:

            # Left hand closed: scatter only particles CLOSE to left hand

            dist_to_L = np.linalg.norm(self.pos - self.hand_L[np.newaxis, :], axis=1)

            close_to_L = dist_to_L < scatter_radius

            if np.any(close_to_L):

                n_scatter = int(close_to_L.sum())

                phi = np.random.uniform(-np.pi, 0.0, n_scatter).astype(np.float32)

                mag = np.random.uniform(18.0, 50.0, n_scatter).astype(np.float32)

                bias_x = np.random.uniform(-self.w * 0.3, 0, n_scatter).astype(np.float32)

                vx = np.cos(phi) * mag + bias_x * 0.15

                vy = np.abs(np.sin(phi)) * mag * 0.85 + np.random.uniform(4.0, 16.0, n_scatter).astype(np.float32)

                self.vel[close_to_L, 0] = vx

                self.vel[close_to_L, 1] = vy

                # Clear ownership so scattered particles stay free, not re-attracted

                self.particle_owner[close_to_L] = 0

                self.particle_hand_color[close_to_L] = self._side_bgr_u8("L")

                self._ground_tint_side = "L"

        elif self.prev_has_R and not has_R and has_L:

            # Right hand closed: scatter only particles CLOSE to right hand

            dist_to_R = np.linalg.norm(self.pos - self.hand_R[np.newaxis, :], axis=1)

            close_to_R = dist_to_R < scatter_radius

            if np.any(close_to_R):

                n_scatter = int(close_to_R.sum())

                phi = np.random.uniform(-np.pi, 0.0, n_scatter).astype(np.float32)

                mag = np.random.uniform(18.0, 50.0, n_scatter).astype(np.float32)

                bias_x = np.random.uniform(0, self.w * 0.3, n_scatter).astype(np.float32)

                vx = np.cos(phi) * mag + bias_x * 0.15

                vy = np.abs(np.sin(phi)) * mag * 0.85 + np.random.uniform(4.0, 16.0, n_scatter).astype(np.float32)

                self.vel[close_to_R, 0] = vx

                self.vel[close_to_R, 1] = vy

                # Clear ownership so scattered particles stay free, not re-attracted

                self.particle_owner[close_to_R] = 0

                self.particle_hand_color[close_to_R] = self._side_bgr_u8("R")

                self._ground_tint_side = "R"



        if n_hands >= 1:

            gb = self._side_bgr_u8(self._ground_tint_side)

            free = self.particle_owner == 0

            if np.any(free):

                self.particle_hand_color[free] = gb



        self.prev_n_hands = n_hands

        self.prev_has_L = has_L

        self.prev_has_R = has_R



        # ===== PEACE-SIGN SHOCKWAVE: per-hand detection & wave management =====
        self.fist_L = False
        self.fist_R = False
        if has_L:
            self.fist_L = bool(peace_any and self.peace_active_L)
            if self.fist_L:
                self.fist_center_L[:] = hand_center(hands["Left"])
        if has_R:
            self.fist_R = bool(peace_any and self.peace_active_R)
            if self.fist_R:
                self.fist_center_R[:] = hand_center(hands["Right"])

        if self.fist_shockwaves_armed:
            # Smooth energy-buildup glow (ramps up while fist held)
            self.fist_core_L = self.fist_core_L * 0.85 + (1.0 if self.fist_L else 0.0) * 0.15
            self.fist_core_R = self.fist_core_R * 0.85 + (1.0 if self.fist_R else 0.0) * 0.15

            # Emit a new wave every 0.25 s while fist is held
            WAVE_INTERVAL = 0.25
            WAVE_SPEED = 320.0   # px/s
            WAVE_LIFETIME = 1.2     # s before fully faded
            if self.fist_L and n_hands > 0:
                self.fist_wave_timer_L -= dt
                if self.fist_wave_timer_L <= 0.0:
                    self.fist_waves.append({
                        'cx': float(self.fist_center_L[0]),
                        'cy': float(self.fist_center_L[1]),
                        'r':  0.0,
                        'alpha': 1.0,
                        'born': self.t,
                        'speed': WAVE_SPEED,
                        'life': WAVE_LIFETIME,
                        'side': 'L',
                    })
                    self.fist_wave_timer_L = WAVE_INTERVAL
            else:
                self.fist_wave_timer_L = 0.0   # reset so next fist fires immediately

            if self.fist_R and n_hands > 0:
                self.fist_wave_timer_R -= dt
                if self.fist_wave_timer_R <= 0.0:
                    self.fist_waves.append({
                        'cx': float(self.fist_center_R[0]),
                        'cy': float(self.fist_center_R[1]),
                        'r':  0.0,
                        'alpha': 1.0,
                        'born': self.t,
                        'speed': WAVE_SPEED,
                        'life': WAVE_LIFETIME,
                        'side': 'R',
                    })
                    self.fist_wave_timer_R = WAVE_INTERVAL
            else:
                self.fist_wave_timer_R = 0.0

            # Advance all active waves and repel particles inside ring
            RING_WIDTH = 30.0   # px — thickness of the push zone
            alive = []
            for w in self.fist_waves:
                age   = self.t - w['born']
                w['r'] = age * w['speed']
                w['alpha'] = max(0.0, 1.0 - age / w['life'])
                if w['alpha'] <= 0.0:
                    continue
                # Push particles within the ring band outward
                cx_w, cy_w = w['cx'], w['cy']
                wc = np.array([cx_w, cy_w], dtype=np.float32)
                d2p = self.pos - wc[np.newaxis, :]
                dist2p = np.linalg.norm(d2p, axis=1)
                inner  = w['r'] - RING_WIDTH * 0.5
                outer  = w['r'] + RING_WIDTH * 0.5
                in_ring = (dist2p > inner) & (dist2p < outer) & (dist2p > 0.1)
                if np.any(in_ring):
                    push_dir = d2p[in_ring] / dist2p[in_ring, np.newaxis]
                    # Falloff inside ring band (strongest at centre)
                    band_pos = np.abs(dist2p[in_ring] - w['r']) / (RING_WIDTH * 0.5 + 1e-3)
                    push_mag = (1.0 - band_pos) * 8.0 * w['alpha']
                    self.vel[in_ring] += push_dir * push_mag[:, np.newaxis]
                alive.append(w)
            self.fist_waves = alive
        else:
            self.fist_waves = []
            self.fist_wave_timer_L = 0.0
            self.fist_wave_timer_R = 0.0
            self.fist_core_L *= 0.88
            self.fist_core_R *= 0.88

        # Peace-sign scatter (only when at least one hand visible)

        if fist_any and not self.prev_fist and n_hands > 0:

            ref = c_single if mode != "dual" else (self.hand_L + self.hand_R) * 0.5

            self.vel *= 0.82
            n_burst = min(self.n, 2400)
            idx = np.random.choice(self.n, n_burst, replace=False)
            phi = np.random.uniform(0, 2 * np.pi, n_burst).astype(np.float32)
            r_mag = np.random.uniform(13.0, 44.0, n_burst).astype(np.float32)
            spin = np.random.uniform(-10.0, 10.0, n_burst).astype(np.float32)
            self.vel[idx, 0] = np.cos(phi) * r_mag - np.sin(phi) * spin * 0.38
            self.vel[idx, 1] = np.sin(phi) * r_mag + np.cos(phi) * spin * 0.38
            self.vel[idx] += np.random.normal(0, 3.2, (n_burst, 2)).astype(np.float32)

            self.scatter_mode = True



        if not fist_any and self.scatter_mode:

            self.scatter_mode = False



        self.prev_fist = fist_any



        if n_hands == 0:

            self._has_render_targets = False

            self.vel[:, 1] += 3.8                                                # stronger downward gravity

            self.vel[:, 0] += np.random.normal(0, 0.55, self.n).astype(np.float32)  # subtle horizontal drift

            self.vel *= 0.984

            spd = np.linalg.norm(self.vel, axis=1, keepdims=True)

            fast = spd[:, 0] > 68.0

            if np.any(fast):

                self.vel[fast] = self.vel[fast] / spd[fast] * 68.0

            self.pos += self.vel

            bounce_l = self.pos[:, 0] < 2.0

            bounce_r = self.pos[:, 0] > self.w - 2.0

            self.vel[bounce_l, 0] = np.abs(self.vel[bounce_l, 0]) * 0.7

            self.vel[bounce_r, 0] = -np.abs(self.vel[bounce_r, 0]) * 0.7

            self.pos[:, 0] = np.clip(self.pos[:, 0], 2.0, self.w - 2.0)

            # Floor: low bounce + random horizontal spread so they don't all stack

            ground = self.h - 8.0

            hit = self.pos[:, 1] >= ground

            if np.any(hit):

                n_hit = int(hit.sum())

                self.pos[hit, 1] = ground

                self.vel[hit, 1] *= -0.20

                self.vel[hit, 0] *= 0.94

                self.vel[hit, 0] += np.random.normal(0, 1.8, n_hit).astype(np.float32)

            self.hand_L += np.array([0.0, 0.35], np.float32)

            self.hand_R += np.array([0.0, 0.35], np.float32)

            self.hand_L[1] = np.minimum(self.hand_L[1], self.h * 0.92)

            self.hand_R[1] = np.minimum(self.hand_R[1], self.h * 0.92)

        elif self.scatter_mode:

            self._has_render_targets = False

            accel = np.random.normal(0, 0.10, (self.n, 2)).astype(np.float32)

            # If one hand is in shield pose during scatter mode, pull particles toward shield.
            shield_pull_active = (n_hands == 1) and (
                (has_L and self.shield_active_L and self.shield_strength_L > 0.12)
                or (has_R and self.shield_active_R and self.shield_strength_R > 0.12)
            )

            if shield_pull_active:
                shield_center = self.hand_L if has_L else self.hand_R
                s_strength = self.shield_strength_L if has_L else self.shield_strength_R
                delta_s = shield_center[np.newaxis, :] - self.pos
                dist_s = np.linalg.norm(delta_s, axis=1, keepdims=True) + 1e-3
                dir_s = delta_s / dist_s
                tang_s = np.column_stack([-dir_s[:, 1], dir_s[:, 0]])
                # Attraction cone around the hand shield: stronger near center, fades by radius.
                pull_radius = 340.0
                falloff = np.clip(1.0 - dist_s[:, 0] / pull_radius, 0.0, 1.0).astype(np.float32)
                pull_mag = (0.95 + 1.55 * s_strength) * falloff
                swirl_mag = (0.22 + 0.35 * s_strength) * falloff
                accel += dir_s * pull_mag[:, np.newaxis]
                accel += tang_s * swirl_mag[:, np.newaxis]
                # Less random shake while shield gathers particles.
                accel *= 0.82

            self.vel = self.vel * (0.905 if not shield_pull_active else 0.875) + accel

            spd = np.linalg.norm(self.vel, axis=1, keepdims=True)

            fast = spd[:, 0] > (36.0 if not shield_pull_active else 31.5)

            if np.any(fast):

                cap = 36.0 if not shield_pull_active else 31.5
                self.vel[fast] = self.vel[fast] / spd[fast] * cap

            self.pos += self.vel

            oob_x = (self.pos[:, 0] < 1.0) | (self.pos[:, 0] > self.w - 1.0)

            oob_y = (self.pos[:, 1] < 1.0) | (self.pos[:, 1] > self.h - 1.0)

            self.vel[oob_x, 0] *= -0.55

            self.vel[oob_y, 1] *= -0.55

            self.pos[:, 0] = np.clip(self.pos[:, 0], 1.0, self.w - 1.0)

            self.pos[:, 1] = np.clip(self.pos[:, 1], 1.0, self.h - 1.0)

            em = 22.0

            near_l = self.pos[:, 0] < em

            near_r = self.pos[:, 0] > self.w - em

            if np.any(near_l):

                self.vel[near_l, 0] += (em - self.pos[near_l, 0]) * 0.35

            if np.any(near_r):

                self.vel[near_r, 0] -= (self.pos[near_r, 0] - (self.w - em)) * 0.35

        else:

            if mode == "dual":

                target_pos = self.pos.copy()  # Start with current positions (free particles stay where they are)

                left_owned = self.particle_owner == 1

                right_owned = self.particle_owner == 2

                target_pos[left_owned] = self.hand_L

                target_pos[right_owned] = self.hand_R

                

                hvel_carry = np.zeros((self.n, 2), dtype=np.float32)

                hvel_carry[left_owned] = self.vel_L

                hvel_carry[right_owned] = self.vel_R

            elif mode == "single":

                target_pos = self.pos.copy()  # Start with current positions

                owned = self.particle_owner > 0  # Only owned particles get attracted

                target_pos[owned] = c_single

                

                hvel_carry = np.zeros((self.n, 2), dtype=np.float32)

                hvel_carry[owned] = self.vel_L

            else:

                target_pos = np.tile(c_single[None, :], (self.n, 1))

                hvel_carry = np.tile(self.vel_L[None, :], (self.n, 1))



            self._render_target_xy[:] = target_pos

            self._has_render_targets = True



            delta = target_pos - self.pos

            dist      = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-4

            dist_flat = dist[:, 0]

            dir_h     = delta / dist

            perp      = np.column_stack([-dir_h[:, 1], dir_h[:, 0]])

            dist_norm = np.clip(dist_flat / 350.0, 0.0, 1.0)



            # ===== PREMIUM GATHERING PHYSICS =====



            # --- Three layered orbital shells (permanent per-particle assignment) ---

            # Inner (0): fast clockwise  ·  Mid (1): medium counter-clockwise

            # Outer (2): slow clockwise  ->  braided galaxy look

            shell     = (self.seed / (np.pi * 2.0 / 3.0)).astype(np.int32) % 3

            orbit_dir = np.where(shell == 1, -1.0, 1.0).astype(np.float32)

            base_spd  = np.array([0.72, 0.44, 0.26], dtype=np.float32)[shell]

            shell_r   = np.array([0.38, 0.68, 1.00], dtype=np.float32)[shell]

            shell_k   = np.array([0.22, 0.15, 0.09], dtype=np.float32)[shell]



            # --- Breathing pulsation (whole cloud inhales/exhales ~every 3.6 s) ---

            breath     = 1.0 + 0.22 * np.sin(self.t * 1.75)

            head_scale = 1.0 - 0.62 * self._bh_smooth * float(self.pinch_active)

            HEAD_R     = (90.0 + self.smooth_speed * 2.2) * head_scale * breath



            # Per-particle orbital radius with organic radial wobble

            r_base   = self.head_r_norm * HEAD_R * shell_r

            r_wobble = r_base * (1.0 + 0.09 * np.sin(self.t * 5.2 + self.seed * 2.1))



            # Orbital angle - each shell spins its own direction + gentle harmonic

            harmonic = 0.10 * np.sin(self.t * 4.0 + self.seed * 1.6)

            ang      = self.head_theta + self.t * base_spd * orbit_dir + harmonic

            tx = target_pos[:, 0] + r_wobble * np.cos(ang)

            ty = target_pos[:, 1] + r_wobble * np.sin(ang)

            tgt_disk = np.column_stack([tx, ty]).astype(np.float32)



            bh_mult    = 1.0 + 0.5 * self._bh_smooth * float(self.pinch_active)

            disk_accel = (tgt_disk - self.pos) * (shell_k * bh_mult)[:, np.newaxis]



            # --- Inward pull: strong swoop from far, soft when already close ---

            pull_base = np.clip(dist_flat / 180.0, 0.04, 1.3) * 0.44

            far_bonus = np.maximum(0.0, (dist_flat - HEAD_R * 1.4) / (HEAD_R + 1e-3)) * 0.55

            accel     = dir_h * (pull_base + far_bonus)[:, np.newaxis] + disk_accel



            # --- Tangential swoop: far particles arc in along their shell orbit ---

            swoop = np.maximum(0.0, (dist_flat - HEAD_R * 0.9) / (HEAD_R + 1e-3)) * 0.28

            accel += perp * (orbit_dir * swoop)[:, np.newaxis]



            # --- Wave ripple: expanding ring briefly pushes particles outward ---

            ripple_T   = 4.0

            rph        = (self.t % ripple_T) / ripple_T

            ripple_r   = rph * HEAD_R * 2.4

            r_width    = max(HEAD_R * 0.30, 1.0)

            ripple_w   = np.exp(-((dist_flat - ripple_r) / r_width) ** 2)

            ripple_env = float(np.sin(rph * np.pi) ** 2)

            accel     -= dir_h * (ripple_w * ripple_env * 0.38)[:, np.newaxis]



            # --- Pinch black hole ---

            if self.pinch_active:

                bh    = self._bh_smooth

                inv_r = 1.0 / np.clip(dist, 12.0, 900.0)

                accel += dir_h * (inv_r * (420.0 + 380.0 * bh))

                accel += perp * (0.20 + 0.30 * bh) * (1.0 + 0.8 * bh)



            # --- Mild flow-field turbulence ---

            px  = self.pos[:, 0] * (2.6 / self.w)

            py  = self.pos[:, 1] * (2.6 / self.h)

            nx_ = (np.sin(px * np.pi + py * 0.70 * np.pi + self.t * 1.05) *

                   np.cos(py * 0.85 * np.pi + self.t * 0.80)).astype(np.float32)

            ny_ = (np.cos(px * 0.90 * np.pi + self.t * 1.20) *

                   np.sin(py * np.pi + px * 0.45 * np.pi + self.t * 0.65)).astype(np.float32)

            turb = 0.055 + 0.035 * (1.0 - dist_norm)

            if self.pinch_active:

                turb = 0.09 + 0.06 * (1.0 - dist_norm)

            accel[:, 0] += nx_ * turb

            accel[:, 1] += ny_ * turb

            accel += hvel_carry * 0.28



            # Side walls

            edge = 26.0

            k_w  = 0.034 * (1.0 + 0.5 * self._bh_smooth * float(self.pinch_active))

            accel[:, 0] += k_w * np.maximum(0.0, edge - self.pos[:, 0])

            accel[:, 0] -= k_w * np.maximum(0.0, self.pos[:, 0] - (self.w - edge))



            self.apply_vortex_forces()

            self.apply_magnetic_repulsion()

            self.apply_gravity_shift()



            self.vel = self.vel * 0.82 + accel

            spd  = np.linalg.norm(self.vel, axis=1, keepdims=True)

            cap  = 32.0 + self.smooth_speed * 1.1

            fast = spd[:, 0] > cap

            if np.any(fast):

                self.vel[fast] = self.vel[fast] / spd[fast] * cap

            self.pos += self.vel



            d_far = np.linalg.norm(self.pos - target_pos, axis=1)

            far   = d_far > min(self.w, self.h) * 1.05

            if np.any(far):

                k = int(far.sum())

                self.pos[far] = target_pos[far] + np.random.normal(0, 24.0, (k, 2))

                self.vel[far] = np.random.normal(0, 0.5, (k, 2))

                self.z[far]   = np.random.uniform(0, 280, k)

                self.u[far]   = 0.0



        # Skip environmental forces/collisions for speed (can re-enable if desired)

        # self.check_particle_collisions()

        # self.apply_environmental_forces(n_hands)

        # Global shield attraction: whenever shield is active, pull particles toward it.
        shield_L_on = bool(has_L and self.shield_strength_L > 0.18)
        shield_R_on = bool(has_R and self.shield_strength_R > 0.18)
        if shield_L_on or shield_R_on:
            if shield_L_on and shield_R_on:
                dL = np.linalg.norm(self.pos - self.hand_L[np.newaxis, :], axis=1)
                dR = np.linalg.norm(self.pos - self.hand_R[np.newaxis, :], axis=1)
                pick_L = dL <= dR

                if np.any(pick_L):
                    delta_L = self.hand_L[np.newaxis, :] - self.pos[pick_L]
                    dist_L = np.linalg.norm(delta_L, axis=1, keepdims=True) + 1e-3
                    dir_L = delta_L / dist_L
                    fall_L = np.power(np.clip(1.0 - dist_L[:, 0] / 430.0, 0.0, 1.0), 1.35)
                    pull_L = (1.10 + 1.90 * self.shield_strength_L) * fall_L
                    self.vel[pick_L] += dir_L * pull_L[:, np.newaxis]

                if np.any(~pick_L):
                    delta_R = self.hand_R[np.newaxis, :] - self.pos[~pick_L]
                    dist_R = np.linalg.norm(delta_R, axis=1, keepdims=True) + 1e-3
                    dir_R = delta_R / dist_R
                    fall_R = np.power(np.clip(1.0 - dist_R[:, 0] / 430.0, 0.0, 1.0), 1.35)
                    pull_R = (1.10 + 1.90 * self.shield_strength_R) * fall_R
                    self.vel[~pick_L] += dir_R * pull_R[:, np.newaxis]
            else:
                center = self.hand_L if shield_L_on else self.hand_R
                s_strength = self.shield_strength_L if shield_L_on else self.shield_strength_R
                delta = center[np.newaxis, :] - self.pos
                dist = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-3
                direction = delta / dist
                # Strong falloff keeps effect local and stable while still obvious.
                fall = np.power(np.clip(1.0 - dist[:, 0] / 440.0, 0.0, 1.0), 1.28)
                pull = (1.18 + 2.15 * s_strength) * fall
                self.vel += direction * pull[:, np.newaxis]

            # Mild damping improves smoothness during strong attraction.
            self.vel *= 0.985

        self._update_attack_orbs(dt)



        z_tgt = 55.0 + 125.0 * np.sin(self.seed + self.t * 0.78) + 75.0 * np.sin(

            self.seed * 2.2 + self.t * 1.25

        )

        self.vel_z = self.vel_z * 0.87 + (z_tgt - self.z) * 0.024

        self.z = np.clip(self.z + self.vel_z, 1.0, 420.0)



    def _project(self, pos, z):

        s = self.focal / (self.focal + z + 60.0)

        x = (pos[:, 0] - self.w * 0.5) * s + self.w * 0.5

        y = (pos[:, 1] - self.h * 0.5) * s + self.h * 0.5

        return x, y, s



    def _fist_display_rgb(self, side: str):

        """R,G,B in display order — fixed left/right particle colors (BGR [B,G,R] → display RGB)."""

        bgr = self._side_bgr_u8(side)

        return float(bgr[2]), float(bgr[1]), float(bgr[0])



    def get_marker_data(self):

        """Positions and colors for VisPy (smooth GPU markers)."""

        x, y, sc = self._project(self.pos, self.z)

        z_vis = (self.z / 420.0) * 40.0

        pos3 = np.column_stack(
            [x.astype(np.float32), (self.h - y).astype(np.float32), z_vis.astype(np.float32)]
        ).astype(np.float32)



        depth_b = np.clip(sc * 1.48, 0.35, 1.58)

        body_b = np.clip(1.12 - self.u * 0.58, 0.35, 1.18)

        head_bst = np.where(self.u < 0.32, 1.45, 1.0)

        

        # Calculate proximity to hands EARLY for global dimming

        dist_to_L = np.linalg.norm(self.pos - self.hand_L[np.newaxis, :], axis=1)

        dist_to_R = np.linalg.norm(self.pos - self.hand_R[np.newaxis, :], axis=1)

        dist_to_closest_hand = np.minimum(dist_to_L, dist_to_R)

        

        # EXTREME proximity dimming - particles near hands become nearly invisible

        proximity_dim = np.power(np.clip(dist_to_closest_hand / 150.0, 0, 1), 1.2)  # Power curve for aggressive dimming

        proximity_dim = np.clip(proximity_dim, 0.02, 1.0)  # Never go above 2% brightness when very close

        

        glow = depth_b * head_bst * body_b * proximity_dim  # Apply proximity dimming to glow itself



        ph = self.particle_hand_color.astype(np.float32)

        r = ph[:, 2] * glow

        g = ph[:, 1] * glow

        b = ph[:, 0] * glow

        

        if self._has_render_targets and self.pinch_active:

            d = np.linalg.norm(self.pos - self._render_target_xy, axis=1)

            hz = 30.0 + 105.0 * self._bh_smooth

            sink = np.clip(d / np.maximum(hz, 1e-3), 0, 1).astype(np.float32)

            # Even darker in pinch mode - aggressive darkening

            dim = 0.01 + 0.10 * np.power(sink, 0.8)

            r, g, b = r * dim, g * dim, b * dim

        a = np.clip(0.48 + 0.22 * depth_b, 0.4, 0.95)

        face = np.column_stack(

            [

                np.clip(r / 255.0, 0, 1),

                np.clip(g / 255.0, 0, 1),

                np.clip(b / 255.0, 0, 1),

                a,

            ]

        ).astype(np.float32)



        base = 5.5 + 8.0 * np.clip(sc, 0.4, 1.35)

        sizes = np.clip(base, 3.5, 20.0).astype(np.float32)

        return pos3, face, sizes



    def get_frame_bgr(self, rw: int, rh: int, hands: dict = None) -> np.ndarray:

        """High-res layered bloom + velocity trail (shooting-star) + pinch dark core."""

        if hands is None:

            hands = {}

        # Enhanced visuals: Better trail effect for particle glow

        n_st = 5  # Slightly lower for better FPS

        fr = np.linspace(0.0, 0.94, n_st, dtype=np.float32)

        spd = np.linalg.norm(self.vel, axis=1, keepdims=True)

        u = np.divide(

            self.vel,

            np.maximum(spd, 1e-3),

            out=np.zeros_like(self.vel),

            where=(spd > 1e-3),

        )

        tail_len = np.clip(spd * 2.3 + 13.0, 10.0, 72.0)

        pw = self.pos[:, None, :] - u[:, None, :] * (tail_len[:, None, :] * fr[None, :, None])

        zv = np.broadcast_to(self.z[:, None], (self.n, n_st))

        pwf = pw.reshape(-1, 2)

        zf = zv.reshape(-1)

        xf, yf, scf = self._project(pwf, zf)

        moving = (spd[:, 0] > 2.2).astype(np.float32)

        wt_trail = np.broadcast_to(((1.0 - fr) ** 0.75)[None, :], (self.n, n_st)).copy()  # Smoother falloff

        wt_trail *= (0.28 + 0.72 * moving[:, None])

        wt_trail = np.maximum(wt_trail, 0.06 * moving[:, None])

        w_flat = wt_trail.reshape(-1)



        sx, sy = rw / float(self.w), rh / float(self.h)

        xi = np.clip((xf * sx).astype(np.int32), 0, rw - 1)

        yi = np.clip((yf * sy).astype(np.int32), 0, rh - 1)

        size = rw * rh

        idx = yi * rw + xi



        ph = self.particle_hand_color.astype(np.float32)

        r0, g0, b0 = ph[:, 2].copy(), ph[:, 1].copy(), ph[:, 0].copy()

        

        r_flat = np.repeat(r0, n_st) * w_flat

        g_flat = np.repeat(g0, n_st) * w_flat

        b_flat = np.repeat(b0, n_st) * w_flat

        

        # Proximity dimming — particles fade as they get close to the hand (absorbed look)

        if self._has_render_targets:

            dist_to_hand = np.linalg.norm(self.pos - self._render_target_xy, axis=1)

            # Dim to ~15% at 0 px distance, full brightness by 280 px

            proximity_dim = np.clip(dist_to_hand / 280.0, 0.15, 1.0)

            proximity_dim_flat = np.repeat(proximity_dim, n_st)

            r_flat *= proximity_dim_flat

            g_flat *= proximity_dim_flat

            b_flat *= proximity_dim_flat

        

        # Enhanced bloom for better particle visibility - increased intensity

        glow_f = np.clip(scf * 1.85, 0.5, 2.0)

        r_flat *= glow_f

        g_flat *= glow_f

        b_flat *= glow_f



        if self._has_render_targets and self.pinch_active:

            d_head = np.linalg.norm(self.pos - self._render_target_xy, axis=1)

            hz = 32.0 + 108.0 * self._bh_smooth

            sink = np.clip(d_head / np.maximum(hz, 1e-3), 0, 1).astype(np.float32)

            dim = (0.09 + 0.91 * np.power(sink, 0.48)).astype(np.float32)

            dim_flat = np.repeat(dim, n_st)

            r_flat *= dim_flat

            g_flat *= dim_flat

            b_flat *= dim_flat

            ring = np.exp(-(((d_head - hz * 0.52) / np.maximum(hz * 0.20, 1.0)) ** 2)).astype(np.float32)

            ring_flat = np.repeat(ring, n_st) * w_flat

            r_flat += ring_flat * 42.0

            g_flat += ring_flat * 48.0

            b_flat += ring_flat * 55.0



        ov = np.zeros((rh, rw, 3), dtype=np.float32)

        ov[:, :, 2] = np.bincount(idx, weights=r_flat, minlength=size).reshape(rh, rw)

        ov[:, :, 1] = np.bincount(idx, weights=g_flat, minlength=size).reshape(rh, rw)

        ov[:, :, 0] = np.bincount(idx, weights=b_flat, minlength=size).reshape(rh, rw)



        if self.fist_shockwaves_armed:
            # ===== FIST SHOCKWAVES & CORE (fast: scalar depth + one bincount pass per channel) =====
            sx_w = rw / float(self.w)
            sy_w = rh / float(self.h)
            hw, hh = float(self.w) * 0.5, float(self.h) * 0.5
    
            n_sw = 3
            fr_sw = np.linspace(0.0, 0.85, n_sw, dtype=np.float32)
            wt_sw = np.power(1.0 - fr_sw, 0.72).astype(np.float32)
    
            ii_chunks = []
            wr_chunks = []
            wg_chunks = []
            wb_chunks = []
    
            def _project_flat_z(px, py, z_blob):
                """Same perspective as _project; z constant → one scalar scale (no per-call overhead)."""
                s = self.focal / (self.focal + z_blob + 60.0)
                x = (px - hw) * s + hw
                y = (py - hh) * s + hh
                gfw = float(np.clip(s * 1.85, 0.5, 2.0))
                return x, y, gfw
    
            for wv in self.fist_waves:
                if wv['alpha'] <= 0.005:
                    continue
                r_w = wv['r']
                if r_w < 4.0:
                    continue
                alpha = float(wv['alpha'])
                side = wv.get("side", "L")
                rc, gc, bc = self._fist_display_rgb(side)
                cx, cy = float(wv['cx']), float(wv['cy'])
                z_ring = 95.0 + 55.0 * (1.0 - alpha)
    
                for r_scale, shell_w in ((0.98, 0.92), (1.02, 0.88)):
                    r_eff = r_w * r_scale
                    n_pts = max(40, min(120, int(r_eff * sx_w * 2.8)))
                    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False, dtype=np.float32)
                    cos_a = np.cos(angles)
                    sin_a = np.sin(angles)
                    jitter = np.random.normal(0, r_eff * 0.01, n_pts).astype(np.float32)
                    r_ring = r_eff + jitter
    
                    for ki in range(n_sw):
                        fk = float(fr_sw[ki])
                        wt = float(wt_sw[ki])
                        rk = r_ring * (1.0 - 0.12 * fk)
                        px = cx + rk * cos_a
                        py = cy + rk * sin_a
                        x, y, gfw = _project_flat_z(px, py, z_ring)
                        xi = np.clip((x * sx_w).astype(np.int32), 0, rw - 1)
                        yi = np.clip((y * sy_w).astype(np.int32), 0, rh - 1)
                        amp = alpha * shell_w * wt * (0.42 + 0.58 * alpha) * 1.45 * gfw
                        ii_chunks.append(yi * rw + xi)
                        wr_chunks.append(np.full(n_pts, rc * amp, dtype=np.float32))
                        wg_chunks.append(np.full(n_pts, gc * amp, dtype=np.float32))
                        wb_chunks.append(np.full(n_pts, bc * amp, dtype=np.float32))
    
            for core_val, fist_cx, fist_cy, side in [
                (self.fist_core_L, float(self.fist_center_L[0]), float(self.fist_center_L[1]), "L"),
                (self.fist_core_R, float(self.fist_center_R[0]), float(self.fist_center_R[1]), "R"),
            ]:
                if core_val < 0.02:
                    continue
                cr, cg, cb = self._fist_display_rgb(side)
                pulse = 1.0 + 0.22 * float(np.sin(self.t * 14.0))
                spread = 5.0 + 11.0 * core_val * pulse
                n_core = 72
                z_core = 48.0
                for ki in range(n_sw):
                    fk = float(fr_sw[ki])
                    wt = float(wt_sw[ki]) * float(np.exp(-fk * 0.4))
                    rad = spread * (0.4 + 0.6 * fk)
                    px = fist_cx + np.random.normal(0.0, rad, n_core).astype(np.float32)
                    py = fist_cy + np.random.normal(0.0, rad, n_core).astype(np.float32)
                    x, y, gfw = _project_flat_z(px, py, z_core)
                    xi = np.clip((x * sx_w).astype(np.int32), 0, rw - 1)
                    yi = np.clip((y * sy_w).astype(np.int32), 0, rh - 1)
                    amp = core_val * wt * 1.12 * (0.55 + 0.45 * core_val) * gfw
                    ii_chunks.append(yi * rw + xi)
                    wr_chunks.append(np.full(n_core, cr * amp, dtype=np.float32))
                    wg_chunks.append(np.full(n_core, cg * amp, dtype=np.float32))
                    wb_chunks.append(np.full(n_core, cb * amp, dtype=np.float32))
    
            if ii_chunks:
                ii_all = np.concatenate(ii_chunks)
                wr_all = np.concatenate(wr_chunks)
                wg_all = np.concatenate(wg_chunks)
                wb_all = np.concatenate(wb_chunks)
                ov[:, :, 2] += np.bincount(ii_all, weights=wr_all, minlength=size).reshape(rh, rw)
                ov[:, :, 1] += np.bincount(ii_all, weights=wg_all, minlength=size).reshape(rh, rw)
                ov[:, :, 0] += np.bincount(ii_all, weights=wb_all, minlength=size).reshape(rh, rw)
    

        core = cv2.GaussianBlur(ov, (0, 0), 1.5)

        mid = cv2.GaussianBlur(ov, (0, 0), 3.0)

        far = cv2.GaussianBlur(ov, (0, 0), 5.5)

        comp = ov * 0.6 + core * 1.4 + mid * 0.6 + far * 0.3  # Multi-layer bloom for quality

        comp = np.clip(comp, 0, 255).astype(np.uint8)

        frame = cv2.resize(comp, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

        


        frame = self.render_background(frame)
        
        # Attack orbs: soft particle clusters (less bright, more organic).
        if self.orbs:
            orb_layer = np.zeros((self.h, self.w, 3), dtype=np.float32)
            for ob in self.orbs:
                side = ob.get("side", "L")
                base = self._side_bgr_u8(side).astype(np.float32)
                cx, cy = float(ob["x"]), float(ob["y"])
                seed = float(ob.get("seed", 0.0))
                life = float(np.clip(ob.get("life", 0.0), 0.0, 1.0))
                age = float(ob.get("age", 0.0))

                # Core cloud
                core_col = tuple(float(v) * (0.24 + 0.18 * life) for v in base)
                cv2.circle(orb_layer, (int(cx), int(cy)), 3, core_col, -1, cv2.LINE_AA)

                # Ring of tiny particles around core
                t = self.t * 7.0 + seed + age * 2.0
                n_s = 10
                for i in range(n_s):
                    a = t + i * (2.0 * np.pi / n_s)
                    rr = 3.0 + 1.8 * np.sin(t * 0.8 + i * 0.7)
                    px = int(cx + np.cos(a) * rr)
                    py = int(cy + np.sin(a) * rr)
                    amp = 0.18 + 0.24 * (0.5 + 0.5 * np.sin(a + seed))
                    col = tuple(float(v) * amp for v in base)
                    cv2.circle(orb_layer, (px, py), 1, col, -1, cv2.LINE_AA)

                # Subtle trailing dust in travel direction
                vx, vy = float(ob["vx"]), float(ob["vy"])
                vn = np.hypot(vx, vy)
                if vn > 1e-3:
                    tx, ty = -vx / vn, -vy / vn
                    for k in (1, 2, 3):
                        px = int(cx + tx * (k * 3.0))
                        py = int(cy + ty * (k * 3.0))
                        col = tuple(float(v) * (0.10 + 0.05 * (4 - k)) for v in base)
                        cv2.circle(orb_layer, (px, py), 1, col, -1, cv2.LINE_AA)

            orb_blur = cv2.GaussianBlur(orb_layer, (0, 0), 1.2)
            orb_comp = np.clip(orb_layer * 0.85 + orb_blur * 0.95, 0, 255).astype(np.uint8)
            cv2.addWeighted(orb_comp, 0.82, frame, 1.0, 0.0, dst=frame)

        

        # Draw hand skeleton

        if "Left" in hands:

            self.draw_hand_skeleton(frame, hands["Left"], self._side_bgr_tuple("L"), hand_id=1)

        if "Right" in hands:

            self.draw_hand_skeleton(frame, hands["Right"], self._side_bgr_tuple("R"), hand_id=2)

        # Shield render is intentionally on top of skeleton to fully cover the hand area.

        if "Left" in hands and self.shield_strength_L > 0.05:

            self.draw_hand_particle_shield(frame, hands["Left"], "L", self.shield_strength_L)

        if "Right" in hands and self.shield_strength_R > 0.05:

            self.draw_hand_particle_shield(frame, hands["Right"], "R", self.shield_strength_R)

        



        # Apply screen shake

        if self.screen_shake > 0.5:

            h, w = frame.shape[:2]

            shake_amount = int(self.screen_shake * 3.0)

            dy = np.random.randint(-shake_amount, shake_amount + 1)

            dx = np.random.randint(-shake_amount, shake_amount + 1)

            if dy != 0 or dx != 0:

                M = np.float32([[1, 0, dx], [0, 1, dy]])

                frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        
        # Status hints for index trigger confidence.
        cv2.putText(
            frame,
            f"index L:{self.index_conf_L:.2f} R:{self.index_conf_R:.2f} | color L:{self._side_variant_name('L')} R:{self._side_variant_name('R')} | orbs:{len(self.orbs)}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (225, 230, 235),
            1,
            cv2.LINE_AA,
        )

        return frame





def _fake_hand_landmarks(cx: float, cy: float, phase: float) -> np.ndarray:

    """21 points like MediaPipe layout — enough for hand_center() and pinch/fist heuristics."""

    pts = np.zeros((21, 2), dtype=np.float32)

    pts[0] = (cx, cy + 45)

    for k in range(1, 21):

        a = phase + k * 0.35

        r = 22.0 + (k % 5) * 4.0

        pts[k] = (cx + r * np.cos(a), cy + r * np.sin(a) * 0.85)

    pts[4] = pts[3] + (8, -15)

    pts[8] = pts[7] + (6, -40)

    pts[4] += (np.cos(phase) * 5, np.sin(phase) * 5)

    pts[8] += (np.cos(phase) * 5, np.sin(phase) * 5)

    return pts





def run_demo(width: int, height: int, particles: int, rw: int, rh: int) -> None:

    """No camera — proves OpenCV window + physics work on this PC."""

    _announce("DEMO mode (no webcam). Close window or press q.")

    flow = WaterFlow(width=width, height=height, n=particles)

    t0 = time.time()

    win = "Hand particles DEMO"

    try:

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(win, width, height)

    except cv2.error as e:

        _announce(f"OpenCV cannot create a window: {e}\nTry: pip install --upgrade opencv-python")

        return

    while True:

        t = time.time() - t0

        x1 = width * 0.34 + 130 * np.sin(t * 1.05)

        y1 = height * 0.48 + 70 * np.cos(t * 0.88)

        x2 = width * 0.66 + 130 * np.cos(t * 0.95)

        y2 = height * 0.50 + 70 * np.sin(t * 0.92)

        hands = {

            "Left": _fake_hand_landmarks(x1, y1, t * 1.7),

            "Right": _fake_hand_landmarks(x2, y2, t * 1.5 + 1.0),

        }

        flow.update(hands)

        out = flow.get_frame_bgr(rw, rh, hands)

        cv2.putText(

            out,

            f"Peace shockwaves: {'ON' if flow.fist_shockwaves_armed else 'OFF'} (s) | Left color (e) Right color (r)",

            (12, 26),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (120, 220, 255) if flow.fist_shockwaves_armed else (140, 140, 140),

            1,

            cv2.LINE_AA,

        )

        cv2.putText(

            out,

            "DEMO - e=left color r=right color | raise index finger to shoot | q=quit",

            (12, height - 14),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (200, 200, 100),

            1,

            cv2.LINE_AA,

        )

        cv2.imshow(win, out)

        k = cv2.waitKey(1) & 0xFF

        if k == ord("b"):

            flow.bg_mode = (flow.bg_mode + 1) % len(flow.bg_modes)

        if k == ord("s"):

            flow.fist_shockwaves_armed = not flow.fist_shockwaves_armed

        if k == ord("e"):

            flow.cycle_element_mode("L")

        if k == ord("r"):

            flow.cycle_element_mode("R")

        if k == ord("q") or k == 27:

            break

        try:

            vp = getattr(cv2, "WND_PROP_VISIBLE", None)

            if vp is not None and cv2.getWindowProperty(win, vp) < 1:

                break

        except (cv2.error, AttributeError):

            break

    cv2.destroyAllWindows()

    _log("demo exit ok")





def run(camera_id, width, height, particles, track_w, track_h, detect_every, use_opencv: bool):

    if sys.platform == "win32":

        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

        if not cap.isOpened():

            cap.release()

            cap = cv2.VideoCapture(camera_id)

    else:

        cap = cv2.VideoCapture(camera_id)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if use_opencv:

        _announce(

            "Hand particles: OpenCV window. Press q to quit.\n"

            "Tip: python main.py --demo  (no webcam test)  |  --vispy needs PyQt5"

        )



    if not cap.isOpened():

        _announce(

            "ERROR: Cannot open webcam.\n"

            "  Try:  python main.py --camera 1\n"

            "  Or:   python main.py --demo\n"

            "  (Close Zoom/Teams/other apps using the camera.)"

        )

        sys.exit(1)



    # Windows often returns fail on the first few reads — warm up the device.

    ok = False

    for _ in range(45):

        ok, test = cap.read()

        if ok and test is not None and test.size > 0:

            break

    if not ok:

        _announce(

            "ERROR: Webcam opened but no frames arrived.\n"

            "  Try:  python main.py --camera 1 --width 640 --height 480\n"

            "  Or:   python main.py --demo"

        )

        cap.release()

        sys.exit(1)



    flow = WaterFlow(width=width, height=height, n=particles)

    detect_every = max(1, detect_every)

    frame_id = 0

    cached_hands = {}

    hold = 0



    rw = max(400, int(width * 0.62))

    rh = max(225, int(height * 0.62))



    want_vispy = not use_opencv

    vispy_ok = want_vispy and _try_vispy_backend()

    if want_vispy and not vispy_ok:

        _announce(

            "VisPy has no GUI backend — falling back to OpenCV.\n"

            "  Optional: pip install PyQt5 vispy"

        )



    markers = canvas = None

    quit_flag = [False]

    visapp = None

    if vispy_ok:

        from vispy import app as visapp

        from vispy import scene



        canvas = scene.SceneCanvas(

            keys="interactive",

            show=True,

            title="Hand particles (VisPy)",

            size=(width, height),

            bgcolor="black",

            vsync=True,

        )

        view = canvas.central_widget.add_view()

        view.camera = scene.PanZoomCamera(aspect=None)

        view.camera.rect = (0, 0, width, height)

        view.camera.interactive = False



        markers = scene.visuals.Markers(parent=view.scene)

        markers.set_data(

            np.zeros((particles, 3), np.float32),

            face_color=np.ones((particles, 4), np.float32),

            edge_color=np.zeros((particles, 4), np.float32),

            edge_width=0,

            size=6,

            symbol="disc",

            antialias=True,

        )



        @canvas.events.key_press.connect

        def on_key(ev):

            if ev.text == "q":

                quit_flag[0] = True

            if ev.text == "s":

                flow.fist_shockwaves_armed = not flow.fist_shockwaves_armed

            if ev.text == "e":

                flow.cycle_element_mode("L")

            if ev.text == "r":

                flow.cycle_element_mode("R")



    win_ocv = "Hand particles"

    if not vispy_ok:

        try:

            cv2.namedWindow(win_ocv, cv2.WINDOW_NORMAL)

            cv2.resizeWindow(win_ocv, width, height)

        except cv2.error as e:

            _announce(f"OpenCV cannot open a display window: {e}")

            cap.release()

            sys.exit(1)



    _log("entering MediaPipe + main loop")

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(

        static_image_mode=False,

        max_num_hands=4,

        model_complexity=0,

        min_detection_confidence=0.42,

        min_tracking_confidence=0.38,

    ) as hands:

        while True:

            if vispy_ok and quit_flag[0]:

                break

            ok, frame = cap.read()

            if not ok or frame is None or frame.size == 0:

                _announce("ERROR: Webcam read failed — check cable / another app using camera.")

                break

            frame = cv2.flip(frame, 1)

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)



            frame_id += 1

            if (frame_id % detect_every) == 0 or hold <= 0:

                small = cv2.resize(frame, (track_w, track_h), interpolation=cv2.INTER_LINEAR)

                res = hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

                det = parse_hands(res, track_w, track_h, width, height)

                if det:

                    cached_hands = det

                    hold = 8

                else:

                    hold -= 1

                    if hold <= 0:

                        cached_hands = {}



            flow.update(cached_hands)

            if vispy_ok:

                pos3, face, sizes = flow.get_marker_data()

                markers.set_data(

                    pos3,

                    face_color=face,

                    size=sizes,

                    edge_width=0,

                    symbol="disc",

                    antialias=True,

                )

                try:

                    canvas.title = "  ·  ".join(

                        [

                            f"{len(cached_hands)} hand(s)",

                            f"{flow.fps:.0f} fps",

                            "SW ON" if flow.fist_shockwaves_armed else "SW off",

                        ]

                    )

                except Exception:

                    pass

                canvas.update()

                visapp.process_events()

            else:

                out = flow.get_frame_bgr(rw, rh, cached_hands)

                cv2.putText(

                    out,

                    f"{len(cached_hands)} hand(s) | {flow.fps:.0f} fps | SW:{'ON' if flow.fist_shockwaves_armed else 'OFF'} | L:{flow._side_variant_name('L')} R:{flow._side_variant_name('R')}",

                    (12, 26),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (210, 210, 210),

                    1,

                    cv2.LINE_AA,

                )

                cv2.putText(

                    out,

                    "e=left color r=right color | raise index finger to shoot particle orbs | q=quit",

                    (12, height - 14),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.42,

                    (170, 170, 170),

                    1,

                    cv2.LINE_AA,

                )

                cv2.imshow(win_ocv, out)

                k = cv2.waitKey(1) & 0xFF

                if k == ord("b"):

                    flow.bg_mode = (flow.bg_mode + 1) % len(flow.bg_modes)

                if k == ord("s"):

                    flow.fist_shockwaves_armed = not flow.fist_shockwaves_armed

                if k == ord("e"):

                    flow.cycle_element_mode("L")

                if k == ord("r"):

                    flow.cycle_element_mode("R")

                if k == ord("q") or k == 27:

                    break

                try:

                    vp = getattr(cv2, "WND_PROP_VISIBLE", None)

                    if vp is not None and cv2.getWindowProperty(win_ocv, vp) < 1:

                        _log("window closed by user (X)")

                        break

                except (cv2.error, AttributeError):

                    break



    cap.release()

    if not vispy_ok:

        cv2.destroyAllWindows()

    _log("main loop exit")





def parse_args():

    p = argparse.ArgumentParser(description="Hand particles with VisPy (two-hand split, auto gravity)")

    p.add_argument("--camera", type=int, default=0)

    p.add_argument("--width", type=int, default=960, help="Use 1280 only if your camera can keep up")

    p.add_argument("--height", type=int, default=540)

    p.add_argument("--particles", type=int, default=3000)

    p.add_argument("--track-width", type=int, default=640)

    p.add_argument("--track-height", type=int, default=360)

    p.add_argument("--detect-every", type=int, default=2)

    p.add_argument(

        "--demo",

        action="store_true",

        help="No webcam — shows two fake hands so you can verify the window works",

    )

    p.add_argument(

        "--vispy",

        action="store_true",

        help="Use VisPy GPU markers (requires: pip install PyQt5 vispy). Default is OpenCV window.",

    )

    return p.parse_args()





if __name__ == "__main__":

    _log("========== session start ==========")

    args = parse_args()

    rw = max(400, int(args.width * 0.62))

    rh = max(225, int(args.height * 0.62))

    _announce(

        f"If nothing appears or the app vanishes instantly, open this log:\n  {_DEBUG_LOG}"

    )

    try:

        if args.demo:

            run_demo(args.width, args.height, args.particles, rw, rh)

        else:

            run(

                camera_id=args.camera,

                width=args.width,

                height=args.height,

                particles=args.particles,

                track_w=args.track_width,

                track_h=args.track_height,

                detect_every=args.detect_every,

                use_opencv=not args.vispy,

            )

    except KeyboardInterrupt:

        _announce("Stopped (Ctrl+C).")

    except SystemExit:

        raise

    except Exception:

        _announce(traceback.format_exc())

        raise

