from vis_nav_game import Player, Action
import pygame
import cv2
import numpy as np
import json
import os
import time

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
HEADING_STEP = 90.0
DATA_DIR = "trajectory_data"
os.makedirs(DATA_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
#  PLAYER
# ──────────────────────────────────────────────────────────────────────────────
class KeyboardPlayerPyGame(Player):

    def __init__(self):
        super().__init__()

        self.fpv = None
        self.screen = None

        self.turning = None
        self.mode = "auto"
        self.heading = 0.0

        self.trajectory_id = int(time.time())
        self.frame_idx = 0
        self.prev_action = None
        self.trajectory_log = []

        self._manual_action = None
        self._quit_requested = False

        # ✅ Frame pacing controls
        self.last_save_time = 0
        self.SAVE_INTERVAL = 0.2   # 5 FPS (good for mapping)
        self.FRAME_DIFF_THRESH = 2.0  # very strict duplicate filter

        # ✅ Optional visual difference filter
        self.prev_frame_gray = None

        # ✅ Per-trajectory folder
        self.traj_dir = os.path.join(DATA_DIR, f"traj_{self.trajectory_id}")
        os.makedirs(self.traj_dir, exist_ok=True)

        # Floor detection params
        self.FLOOR_H_LOW,  self.FLOOR_H_HIGH = 0,   140
        self.FLOOR_S_LOW,  self.FLOOR_S_HIGH = 0,   91
        self.FLOOR_V_LOW,  self.FLOOR_V_HIGH = 204, 255

        self.SCAN_FRAC = 0.40
        self.ZONE_SPLIT = (0.33, 0.67)
        self.GOOD_FLOOR_THRESH = 0.15
        self.LOW_FLOOR_THRESH = 0.05
        self.CENTER_FORWARD_THRESH = 0.25
        self.CENTER_MARGIN = 0.05

        print("="*60)
        print("  Autonomous + Manual Override Player")
        print("  SPACE  → toggle AUTO / MANUAL mode")
        print("  ← / →  → manual LEFT / RIGHT")
        print("  ↑      → manual FORWARD")
        print("  ↓      → manual BACKWARD")
        print("  ESC    → quit + save trajectory")
        print("="*60)

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self):
        pygame.init()
        self.frame_idx = 0
        self.trajectory_log = []
        self.heading = 0.0
        self.turning = None
        self.prev_action = None
        self.last_save_time = 0
        self.prev_frame_gray = None
        print("Reset complete")

    # ─────────────────────────────────────────────────────────────────────────
    def _update_heading(self, action):
        if action == Action.LEFT:
            self.heading = (self.heading - HEADING_STEP) % 360
        elif action == Action.RIGHT:
            self.heading = (self.heading + HEADING_STEP) % 360

    # ─────────────────────────────────────────────────────────────────────────
    # TRAJECTORY LOGGING (FIXED)
    # ─────────────────────────────────────────────────────────────────────────
    def _save_frame_and_log(self, action):
        if self.fpv is None:
            return

        now = time.time()

        # ⏱ Enforce minimum interval (smooth sampling)
        if now - self.last_save_time < self.SAVE_INTERVAL:
            return

        gray = cv2.cvtColor(self.fpv, cv2.COLOR_BGR2GRAY)

        # 🔍 Only skip if visually identical (not just similar)
        if self.prev_frame_gray is not None:
            diff = np.mean(cv2.absdiff(gray, self.prev_frame_gray))

            # Much lower threshold → only remove true duplicates
            if diff < 2.0:
                return

        self.prev_frame_gray = gray

        fname = f"{self.frame_idx}.jpg"
        fpath = os.path.join(self.traj_dir, fname)
        cv2.imwrite(fpath, self.fpv)

        action_name = action.name if isinstance(action, Action) else str(action)

        node = {
            "step": self.frame_idx,
            "image": fname,
            "action": [action_name]
        }

        self.trajectory_log.append(node)

        self.prev_action = action
        self.last_save_time = now
        self.frame_idx += 1
        
    def _flush_trajectory(self):
        out = os.path.join(self.traj_dir, "trajectory.json")
        with open(out, "w") as f:
            json.dump(self.trajectory_log, f, indent=2)

        print(f"\n[SAVE] Trajectory → {out} ({len(self.trajectory_log)} frames)")

    # ─────────────────────────────────────────────────────────────────────────
    def get_floor_mask(self, frame):
        h, w = frame.shape[:2]
        scan_top = int(h * (1.0 - self.SCAN_FRAC))
        roi = frame[scan_top:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lo = np.array([self.FLOOR_H_LOW, self.FLOOR_S_LOW, self.FLOOR_V_LOW], dtype=np.uint8)
        hi = np.array([self.FLOOR_H_HIGH, self.FLOOR_S_HIGH, self.FLOOR_V_HIGH], dtype=np.uint8)

        mask = cv2.inRange(hsv, lo, hi)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[scan_top:, :] = mask
        return full_mask

    # ─────────────────────────────────────────────────────────────────────────
    def compute_zone_scores(self, mask):
        h, w = mask.shape
        l_end = int(w * self.ZONE_SPLIT[0])
        r_start = int(w * self.ZONE_SPLIT[1])

        def frac(region):
            return np.count_nonzero(region) / region.size if region.size > 0 else 0

        return (
            frac(mask[:, :l_end]),
            frac(mask[:, l_end:r_start]),
            frac(mask[:, r_start:])
        )

    # ─────────────────────────────────────────────────────────────────────────
    def decide(self, left, center, right, mask):
        total_floor = np.count_nonzero(mask) / mask.size

        if total_floor < self.LOW_FLOOR_THRESH:
            self.turning = "LEFT"
            return Action.LEFT

        if self.turning is not None:
            if center > self.CENTER_FORWARD_THRESH:
                self.turning = None
                return Action.FORWARD
            return Action.LEFT if self.turning == "LEFT" else Action.RIGHT

        if total_floor < self.GOOD_FLOOR_THRESH:
            self.turning = "LEFT" if left > right else "RIGHT"
            return Action.LEFT if left > right else Action.RIGHT

        if center > self.CENTER_FORWARD_THRESH:
            return Action.FORWARD

        if abs(left - right) < self.CENTER_MARGIN:
            return Action.FORWARD

        if left > right:
            self.turning = "LEFT"
            return Action.LEFT
        else:
            self.turning = "RIGHT"
            return Action.RIGHT

    # ─────────────────────────────────────────────────────────────────────────
    def _pump_events(self):
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self._flush_trajectory()
                self._quit_requested = True
                return

            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_ESCAPE:
                self._flush_trajectory()
                self._quit_requested = True
                return

            if event.key == pygame.K_SPACE:
                self.mode = "manual" if self.mode == "auto" else "auto"
                self.turning = None
                self._manual_action = None
                print(f"MODE → {self.mode}")
                continue

            if event.key == pygame.K_UP:
                self._manual_action = Action.FORWARD
                if self.mode == "auto":
                    self.mode = "manual"
            elif event.key == pygame.K_LEFT:
                self._manual_action = Action.LEFT
            elif event.key == pygame.K_RIGHT:
                self._manual_action = Action.RIGHT
            elif event.key == pygame.K_DOWN:
                self._manual_action = Action.BACKWARD

    # ─────────────────────────────────────────────────────────────────────────
    def act(self):
        if self.fpv is None:
            return Action.IDLE

        self._pump_events()

        if self._quit_requested:
            pygame.quit()
            return Action.QUIT

        if self.mode == "manual":
            action = self._manual_action if self._manual_action else Action.IDLE
            self._manual_action = None

            self._update_heading(action)
            self._save_frame_and_log(action)
            return action

        mask = self.get_floor_mask(self.fpv)
        left, center, right = self.compute_zone_scores(mask)
        action = self.decide(left, center, right, mask)

        self._update_heading(action)
        self._save_frame_and_log(action)

        return action

    # ─────────────────────────────────────────────────────────────────────────
    def see(self, fpv):
        if fpv is None:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        rgb = fpv[:, :, ::-1]
        surf = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")

        self.screen.blit(surf, (0, 0))
        pygame.display.update()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import vis_nav_game as vng
    vng.play(the_player=KeyboardPlayerPyGame())