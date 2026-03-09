"""
Two-Wheel Barrel Robot – Self-balancing + Waypoint Navigation
=============================================================
A cascaded PID controller keeps the inverted-pendulum robot upright
while a high-level waypoint planner drives it along a path.

Controls (in MuJoCo viewer):
  - The robot balances automatically and follows waypoints
  - Press Space to pause / resume
  - Close the viewer window to exit
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math


# ──────────────────────────────────────────────────────────────────
# PID Controller
# ──────────────────────────────────────────────────────────────────
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = 10.0

    def compute(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement
        self.integral = np.clip(self.integral + error * dt,
                                -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


# ──────────────────────────────────────────────────────────────────
# Waypoint Planner  (generates a looping figure-8 / rectangle path)
# ──────────────────────────────────────────────────────────────────
class WaypointPlanner:
    """Supplies target (x, y) positions in sequence."""

    def __init__(self, path: str = "circle", radius: float = 3.0):
        self.waypoints = self._generate(path, radius)
        self.index = 0
        self.reach_threshold = 0.4  # metres

    # --- built-in path generators -----------------------------------
    @staticmethod
    def _generate(kind: str, r: float):
        if kind == "circle":
            n = 24
            return [(r * math.cos(2 * math.pi * i / n),
                     r * math.sin(2 * math.pi * i / n)) for i in range(n)]
        elif kind == "figure8":
            n = 48
            pts = []
            for i in range(n):
                t = 2 * math.pi * i / n
                x = r * math.sin(t)
                y = r * math.sin(t) * math.cos(t)
                pts.append((x, y))
            return pts
        elif kind == "square":
            s = r
            return [(s, 0), (s, s), (0, s), (-s, s),
                    (-s, 0), (-s, -s), (0, -s), (s, -s)]
        elif kind == "star":
            pts = []
            for i in range(10):
                angle = math.pi / 2 + 2 * math.pi * i / 10
                rad = r if i % 2 == 0 else r * 0.4
                pts.append((rad * math.cos(angle), rad * math.sin(angle)))
            return pts
        else:
            return [(2, 0), (2, 2), (-2, 2), (-2, -2), (2, -2)]

    @property
    def current(self):
        return self.waypoints[self.index]

    def advance_if_reached(self, x: float, y: float) -> bool:
        tx, ty = self.current
        if math.hypot(tx - x, ty - y) < self.reach_threshold:
            self.index = (self.index + 1) % len(self.waypoints)
            return True
        return False


# ──────────────────────────────────────────────────────────────────
# Robot Controller
# ──────────────────────────────────────────────────────────────────
class BarrelRobotController:
    """Cascaded PID:  position-loop → pitch-loop → wheel torques
       + yaw (steering) loop for turning toward waypoints."""

    def __init__(self, model, data):
        self.m = model
        self.d = data

        # --- PID gains (tuned for this model) ---
        self.pitch_pid = PIDController(kp=50.0, ki=3.0, kd=6.0)
        self.pos_pid   = PIDController(kp=0.8, ki=0.05, kd=1.5)
        self.yaw_pid   = PIDController(kp=8.0, ki=0.1, kd=2.0)

        # actuator ids
        self.left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
        self.right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")

        # body id
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # navigation target
        self.target_xy = (0.0, 0.0)

    # ---- state helpers (read from freejoint qpos / qvel) -----------
    def _quat(self):
        """Return (w, x, y, z) quaternion of the chassis."""
        return self.d.qpos[3:7]

    def pitch(self):
        w, x, y, z = self._quat()
        return np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))

    def pitch_rate(self):
        return self.d.qvel[4]  # wy

    def yaw(self):
        w, x, y, z = self._quat()
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def yaw_rate(self):
        return self.d.qvel[5]  # wz

    def xy(self):
        return self.d.qpos[0], self.d.qpos[1]

    def forward_speed(self):
        """Speed projected onto the robot's heading direction."""
        vx, vy = self.d.qvel[0], self.d.qvel[1]
        h = self.yaw()
        return vx * math.cos(h) + vy * math.sin(h)

    # ---- control step -----------------------------------------------
    def step(self, dt: float):
        px, py = self.xy()
        tx, ty = self.target_xy

        # --- heading error → yaw control ---
        desired_yaw = math.atan2(ty - py, tx - px)
        yaw_err = desired_yaw - self.yaw()
        # wrap to [-π, π]
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
        yaw_cmd = self.yaw_pid.compute(-yaw_err, dt)

        # --- distance error → position (forward) control ---
        dist = math.hypot(tx - px, ty - py)
        # position PID drives toward the waypoint
        self.pos_pid.setpoint = 0.0
        target_pitch = self.pos_pid.compute(-dist, dt)
        target_pitch = np.clip(target_pitch, -0.15, 0.15)

        # --- pitch (balance) loop ---
        self.pitch_pid.setpoint = target_pitch
        balance_cmd = -self.pitch_pid.compute(self.pitch(), dt)
        balance_cmd = np.clip(balance_cmd, -15.0, 15.0)

        # --- mix balance + steering ---
        yaw_cmd = np.clip(yaw_cmd, -5.0, 5.0)
        left_torque  = np.clip(balance_cmd - yaw_cmd, -15.0, 15.0)
        right_torque = np.clip(balance_cmd + yaw_cmd, -15.0, 15.0)

        self.d.ctrl[self.left_id]  = left_torque
        self.d.ctrl[self.right_id] = right_torque

    def reset_pids(self):
        self.pitch_pid.reset()
        self.pos_pid.reset()
        self.yaw_pid.reset()


# ──────────────────────────────────────────────────────────────────
# Visualisation helpers (trace + target markers via MuJoCo viewer)
# ──────────────────────────────────────────────────────────────────
def draw_waypoints(viewer, waypoints, current_idx):
    """Draw coloured spheres at waypoint positions."""
    for i, (wx, wy) in enumerate(waypoints):
        rgba = [0.2, 0.9, 0.2, 0.6] if i == current_idx else [0.4, 0.4, 0.4, 0.3]
        viewer.user_scn.ngeom += 1
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                [0.12, 0, 0], [wx, wy, 0.12],
                                np.eye(3).flatten().astype(np.float64),
                                np.array(rgba, dtype=np.float32))


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    # ---- load model ------------------------------------------------
    model = mujoco.MjModel.from_xml_path("/home/quannh/working/intern_vin/two_wheel_car_balancing/balancing_robot.xml")
    data  = mujoco.MjData(model)

    # ---- initial pose: upright, slightly above ground ---------------
    data.qpos[2] = 0.155              # height ≈ wheel radius
    data.qpos[3:7] = [1, 0, 0, 0]    # identity quaternion (upright)
    mujoco.mj_forward(model, data)

    # ---- controller + planner --------------------------------------
    ctrl = BarrelRobotController(model, data)
    # Choose a path: "circle", "figure8", "square", or "star"
    planner = WaypointPlanner(path="figure8", radius=3.0)
    ctrl.target_xy = planner.current

    dt = model.opt.timestep

    print("=" * 62)
    print("  Two-Wheel Barrel Robot – Navigation Demo")
    print("=" * 62)
    print(f"  Path : figure-8  |  Waypoints : {len(planner.waypoints)}")
    print(f"  Pitch PID : Kp={ctrl.pitch_pid.kp}  Ki={ctrl.pitch_pid.ki}  Kd={ctrl.pitch_pid.kd}")
    print(f"  Pos   PID : Kp={ctrl.pos_pid.kp}  Ki={ctrl.pos_pid.ki}  Kd={ctrl.pos_pid.kd}")
    print(f"  Yaw   PID : Kp={ctrl.yaw_pid.kp}  Ki={ctrl.yaw_pid.ki}  Kd={ctrl.yaw_pid.kd}")
    print("-" * 62)
    print("  Close the viewer to quit.  Space = pause.\n")

    # ---- simulation loop -------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_n = 0
        log_interval = 2000

        while viewer.is_running():
            t0 = time.time()

            # high-level: advance waypoint
            px, py = ctrl.xy()
            if planner.advance_if_reached(px, py):
                ctrl.target_xy = planner.current

            ctrl.target_xy = planner.current
            ctrl.step(dt)

            mujoco.mj_step(model, data)

            # draw waypoint markers
            viewer.user_scn.ngeom = 0
            draw_waypoints(viewer, planner.waypoints, planner.index)

            viewer.sync()

            step_n += 1
            if step_n % log_interval == 0:
                pitch_deg = np.degrees(ctrl.pitch())
                yaw_deg   = np.degrees(ctrl.yaw())
                print(f"  step {step_n:7d} | pitch {pitch_deg:+6.2f}° | "
                      f"yaw {yaw_deg:+6.1f}° | pos ({px:+.2f},{py:+.2f}) | "
                      f"wp#{planner.index}")

            # fall detection → auto-reset
            if abs(ctrl.pitch()) > math.radians(45):
                print(f"\n  ⚠  Robot fell! ({np.degrees(ctrl.pitch()):.1f}°) — resetting…")
                time.sleep(1.5)
                mujoco.mj_resetData(model, data)
                data.qpos[2] = 0.155
                data.qpos[3:7] = [1, 0, 0, 0]
                mujoco.mj_forward(model, data)
                ctrl.reset_pids()
                planner.index = 0
                ctrl.target_xy = planner.current
                step_n = 0
                print("  Reset done.\n")

            # real-time pacing
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    print("\nSimulation finished.")


if __name__ == "__main__":
    main()
