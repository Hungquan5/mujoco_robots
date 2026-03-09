
"""
Two-Wheel Legged Robot - Balancing + Jumping + Navigation
==========================================================
A wheeled-legged robot (similar to Ascento / Handle) with:
  - 2 driven wheels
  - 2 hip joints  (thigh rotation)
  - 2 knee joints (shin rotation)

Control architecture:
  1. Leg posture PD   - keeps hip/knee at desired angles (stance height)
  2. Pitch balance PID - drives wheels to keep body upright
  3. Position PID      - leans body toward waypoint
  4. Yaw PID           - differential wheel torque for steering
  5. Jump state machine - CROUCH -> THRUST -> FLIGHT -> LAND

The robot periodically jumps while navigating waypoints.
Close the MuJoCo viewer window to exit.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import time

# ──────────────────────────────────────────────────────────────────────
#  PID Controller
# ──────────────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, setpoint=0.0, ilimit=10.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self._prev_err = 0.0
        self._integral = 0.0
        self._ilimit = ilimit

    def compute(self, measurement, dt):
        err = self.setpoint - measurement
        self._integral = np.clip(self._integral + err * dt,
                                 -self._ilimit, self._ilimit)
        deriv = (err - self._prev_err) / dt if dt > 0 else 0.0
        self._prev_err = err
        return self.kp * err + self.ki * self._integral + self.kd * deriv

    def reset(self):
        self._prev_err = 0.0
        self._integral = 0.0


# ──────────────────────────────────────────────────────────────────────
#  Jump State Machine
# ──────────────────────────────────────────────────────────────────────
class JumpState:
    IDLE    = 0   # normal standing / moving
    CROUCH  = 1   # bending legs to store energy
    THRUST  = 2   # rapidly extending legs to launch
    FLIGHT  = 3   # airborne – tuck legs
    LAND    = 4   # absorb impact, return to stance


class JumpController:
    """Finite-state-machine that orchestrates a jump sequence.
    Crouch: lower hip, bend knee deep to store energy.
    Thrust: extend knee rapidly to push off ground."""

    # Timing for each phase (seconds)
    CROUCH_DUR = 0.30    # time to get into deep squat
    THRUST_DUR = 0.12    # rapid knee extension
    FLIGHT_DUR = 0.50    # max time in air before forcing landing
    LAND_DUR   = 0.80    # absorb and recover

    # Leg targets per phase                  hip      knee
    CROUCH_POSE = ( 0.80, -1.40)   # deep squat: hip forward, knee bent hard
    THRUST_POSE = ( 0.10, -0.10)   # legs nearly straight: knee extends fast!
    FLIGHT_POSE = ( 0.30, -0.50)   # tucked in air
    LAND_POSE   = ( 0.45, -0.80)   # soft bent to absorb impact

    # Leg PD gains per phase              hip_kp  hip_kd  knee_kp  knee_kd
    GAINS = {
        JumpState.IDLE:   (40.0,   4.0,  30.0,   3.0),
        JumpState.CROUCH: (60.0,   8.0,  50.0,   6.0),   # strong + damped to reach pose
        JumpState.THRUST: (80.0,   2.0, 100.0,   1.0),   # very high knee kp, minimal damping = explosive
        JumpState.FLIGHT: (30.0,   5.0,  25.0,   4.0),
        JumpState.LAND:   (50.0,  25.0,  40.0,  20.0),   # high damping to absorb
    }

    def __init__(self):
        self.state = JumpState.IDLE
        self.phase_start = 0.0
        self.jump_count = 0
        self.land_end_time = 0.0   # when landing phase finished
        self.needs_pid_reset = False  # flag for controller to reset PIDs

    def request_jump(self, sim_time: float):
        """Initiate a jump if currently idle."""
        if self.state == JumpState.IDLE:
            self.state = JumpState.CROUCH
            self.phase_start = sim_time
            self.needs_pid_reset = False

    def update(self, sim_time: float, body_height: float, vz: float):
        """Advance the state machine. Returns (des_hip, des_knee, gains_tuple, is_jumping) or None."""
        elapsed = sim_time - self.phase_start

        if self.state == JumpState.IDLE:
            # Post-landing stabilisation: use high damping gains for 0.5s after landing
            post_land_dur = sim_time - self.land_end_time
            if self.land_end_time > 0 and post_land_dur < 0.5:
                blend = post_land_dur / 0.5  # 0->1
                # Interpolate between LAND gains and IDLE gains
                land_g = self.GAINS[JumpState.LAND]
                idle_g = self.GAINS[JumpState.IDLE]
                gains = tuple(l * (1 - blend) + i * blend for l, i in zip(land_g, idle_g))
                return (0.35, -0.70, gains, False)
            return None  # let normal controller handle it

        elif self.state == JumpState.CROUCH:
            if elapsed >= self.CROUCH_DUR:
                self.state = JumpState.THRUST
                self.phase_start = sim_time
            return (*self.CROUCH_POSE, self.GAINS[JumpState.CROUCH], True)

        elif self.state == JumpState.THRUST:
            if elapsed >= self.THRUST_DUR:
                self.state = JumpState.FLIGHT
                self.phase_start = sim_time
            return (*self.THRUST_POSE, self.GAINS[JumpState.THRUST], True)

        elif self.state == JumpState.FLIGHT:
            # Transition to LAND when we start falling back or max time
            if (elapsed > 0.10 and vz < -0.3) or elapsed >= self.FLIGHT_DUR:
                self.state = JumpState.LAND
                self.phase_start = sim_time
                self.needs_pid_reset = True  # reset balance PIDs for clean landing
            return (*self.FLIGHT_POSE, self.GAINS[JumpState.FLIGHT], True)

        elif self.state == JumpState.LAND:
            if elapsed >= self.LAND_DUR:
                self.state = JumpState.IDLE
                self.jump_count += 1
                self.land_end_time = sim_time
            return (*self.LAND_POSE, self.GAINS[JumpState.LAND], True)

        return None

    @property
    def is_jumping(self):
        return self.state != JumpState.IDLE

    @property
    def state_name(self):
        return ["IDLE", "CROUCH", "THRUST", "FLIGHT", "LAND"][self.state]


# ──────────────────────────────────────────────────────────────────────
#  Waypoint Planner
# ──────────────────────────────────────────────────────────────────────
class WaypointPlanner:
    def __init__(self, path="figure8", radius=3.0):
        self.waypoints = self._gen(path, radius)
        self.index = 0
        self.threshold = 0.5

    @staticmethod
    def _gen(kind, r):
        if kind == "circle":
            n = 24
            return [(r * math.cos(2 * math.pi * i / n),
                     r * math.sin(2 * math.pi * i / n)) for i in range(n)]
        elif kind == "figure8":
            n = 48
            return [(r * math.sin(2 * math.pi * i / n),
                     r * math.sin(2 * math.pi * i / n) *
                     math.cos(2 * math.pi * i / n)) for i in range(n)]
        elif kind == "square":
            s = r
            return [(s, 0), (s, s), (0, s), (-s, s),
                    (-s, 0), (-s, -s), (0, -s), (s, -s)]
        elif kind == "star":
            pts = []
            for i in range(10):
                a = math.pi / 2 + 2 * math.pi * i / 10
                rd = r if i % 2 == 0 else r * 0.4
                pts.append((rd * math.cos(a), rd * math.sin(a)))
            return pts
        else:
            return [(2, 0), (2, 2), (-2, 2), (-2, -2), (2, -2)]

    @property
    def current(self):
        return self.waypoints[self.index]

    def advance(self, x, y):
        tx, ty = self.current
        if math.hypot(tx - x, ty - y) < self.threshold:
            self.index = (self.index + 1) % len(self.waypoints)
            return True
        return False


# ──────────────────────────────────────────────────────────────────────
#  Wheeled-Legged Robot Controller
# ──────────────────────────────────────────────────────────────────────
class WheeledLeggedController:
    """
    qpos layout (nq=13):
        [0:3]   body position (x, y, z)
        [3:7]   body quaternion (w, x, y, z)
        [7]     left_hip
        [8]     left_knee
        [9]     left_wheel
        [10]    right_hip
        [11]    right_knee
        [12]    right_wheel

    qvel layout (nv=12):
        [0:3]   body linear vel
        [3:6]   body angular vel  (wx, wy, wz)
        [6]     left_hip_vel
        [7]     left_knee_vel
        [8]     left_wheel_vel
        [9]     right_hip_vel
        [10]    right_knee_vel
        [11]    right_wheel_vel

    ctrl layout (nu=6):
        [0] left_wheel_motor
        [1] right_wheel_motor
        [2] left_hip_motor
        [3] right_hip_motor
        [4] left_knee_motor
        [5] right_knee_motor
    """

    # Desired leg angles for the default standing posture
    # With longer shins (0.28m), lower stance for stability
    DEFAULT_HIP  =  0.35   # rad  (~20°)
    DEFAULT_KNEE = -0.70   # rad  (~-40°)

    def __init__(self, model, data):
        self.m = model
        self.d = data

        # ----- PIDs -----
        # Balance (pitch) - inner loop
        self.pitch_pid = PID(kp=80.0, ki=5.0, kd=12.0)
        # Position (forward travel) - outer loop
        self.pos_pid = PID(kp=0.6, ki=0.03, kd=1.2)
        # Yaw (steering)
        self.yaw_pid = PID(kp=6.0, ki=0.1, kd=1.5)

        # Leg posture PD (same gains for L/R)
        self.hip_kp,  self.hip_kd  = 40.0, 4.0
        self.knee_kp, self.knee_kd = 30.0, 3.0

        # Target waypoint
        self.target_xy = (0.0, 0.0)

        # Desired leg posture (can be modulated for crouching / jumping)
        self.des_hip  = self.DEFAULT_HIP
        self.des_knee = self.DEFAULT_KNEE

        # Jump controller
        self.jump = JumpController()

    # ── state readers ────────────────────────────────────────────────
    def pitch(self):
        w, x, y, z = self.d.qpos[3:7]
        return np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))

    def yaw(self):
        w, x, y, z = self.d.qpos[3:7]
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def xy(self):
        return self.d.qpos[0], self.d.qpos[1]

    def height(self):
        return self.d.qpos[2]

    # ── leg posture PD ───────────────────────────────────────────────
    def _leg_torques(self, gains=None):
        """PD control to hold desired hip/knee angles.
        gains: optional (hip_kp, hip_kd, knee_kp, knee_kd) override."""
        hkp = self.hip_kp
        hkd = self.hip_kd
        kkp = self.knee_kp
        kkd = self.knee_kd
        if gains is not None:
            hkp, hkd, kkp, kkd = gains

        lh, lk = self.d.qpos[7], self.d.qpos[8]
        rh, rk = self.d.qpos[10], self.d.qpos[11]
        lhv, lkv = self.d.qvel[6], self.d.qvel[7]
        rhv, rkv = self.d.qvel[9], self.d.qvel[10]

        tau_lh = hkp * (self.des_hip  - lh) - hkd * lhv
        tau_lk = kkp * (self.des_knee - lk) - kkd * lkv
        tau_rh = hkp * (self.des_hip  - rh) - hkd * rhv
        tau_rk = kkp * (self.des_knee - rk) - kkd * rkv

        lim_h, lim_k = 50, 50
        return (np.clip(tau_lh, -lim_h, lim_h), np.clip(tau_lk, -lim_k, lim_k),
                np.clip(tau_rh, -lim_h, lim_h), np.clip(tau_rk, -lim_k, lim_k))

    def vz(self):
        """Vertical velocity of body."""
        return self.d.qvel[2]

    # ── main control step ────────────────────────────────────────────
    def step(self, dt):
        px, py = self.xy()
        tx, ty = self.target_xy

        # ---- check jump state machine ----
        jump_gains = None
        jump_result = self.jump.update(self.d.time, self.height(), self.vz())
        if jump_result is not None:
            des_hip, des_knee, gains, _ = jump_result
            self.des_hip  = des_hip
            self.des_knee = des_knee
            jump_gains = gains

        # Reset balance PIDs when entering LAND for a clean start
        if self.jump.needs_pid_reset:
            self.pitch_pid.reset()
            self.pos_pid.reset()
            self.jump.needs_pid_reset = False

        # ---- yaw steering ----
        des_yaw = math.atan2(ty - py, tx - px)
        yaw_err = (des_yaw - self.yaw() + math.pi) % (2 * math.pi) - math.pi
        yaw_cmd = self.yaw_pid.compute(-yaw_err, dt)
        yaw_cmd = np.clip(yaw_cmd, -6.0, 6.0)

        # ---- forward position -> target pitch angle ----
        dist = math.hypot(tx - px, ty - py)
        self.pos_pid.setpoint = 0.0
        target_pitch = self.pos_pid.compute(-dist, dt)
        target_pitch = np.clip(target_pitch, -0.15, 0.15)

        # During thrust/flight/land, keep upright
        if self.jump.state in (JumpState.THRUST, JumpState.FLIGHT, JumpState.LAND):
            target_pitch = 0.0

        # ---- pitch balance loop -> wheel torque ----
        self.pitch_pid.setpoint = target_pitch
        bal = -self.pitch_pid.compute(self.pitch(), dt)
        # Use higher balance torque limit during all jump phases
        if self.jump.state in (JumpState.THRUST, JumpState.FLIGHT, JumpState.LAND):
            bal = bal * 1.8
            bal = np.clip(bal, -35.0, 35.0)
        else:
            bal = np.clip(bal, -20.0, 20.0)

        # ---- mix balance + yaw into wheel torques ----
        if self.jump.state in (JumpState.THRUST, JumpState.FLIGHT, JumpState.LAND):
            yaw_cmd = 0.0   # disable steering during jump
        wlim = 35.0 if self.jump.state in (JumpState.THRUST, JumpState.FLIGHT, JumpState.LAND) else 20.0
        left_wheel  = np.clip(bal - yaw_cmd, -wlim, wlim)
        right_wheel = np.clip(bal + yaw_cmd, -wlim, wlim)

        # ---- leg torques (always PD-based) ----
        tau_lh, tau_lk, tau_rh, tau_rk = self._leg_torques(gains=jump_gains)

        # ---- write actuators ----
        self.d.ctrl[0] = left_wheel
        self.d.ctrl[1] = right_wheel
        self.d.ctrl[2] = tau_lh
        self.d.ctrl[3] = tau_rh
        self.d.ctrl[4] = tau_lk
        self.d.ctrl[5] = tau_rk

    def set_stance(self, height_frac):
        """Set leg crouch: 0.0 = fully crouched, 1.0 = fully extended."""
        height_frac = np.clip(height_frac, 0.0, 1.0)
        self.des_hip  = 0.8 - 0.5 * height_frac    # crouched 0.8 → extended 0.3
        self.des_knee = -1.2 + 0.6 * height_frac    # crouched -1.2 → extended -0.6

    def reset_pids(self):
        self.pitch_pid.reset()
        self.pos_pid.reset()
        self.yaw_pid.reset()


# ──────────────────────────────────────────────────────────────────────
#  Visualisation – waypoint markers
# ──────────────────────────────────────────────────────────────────────
def draw_waypoints(viewer, waypoints, idx):
    for i, (wx, wy) in enumerate(waypoints):
        rgba = [0.2, 0.9, 0.2, 0.7] if i == idx else [0.4, 0.4, 0.4, 0.25]
        viewer.user_scn.ngeom += 1
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                [0.10, 0, 0], [wx, wy, 0.10],
                np.eye(3).flatten().astype(np.float64),
                np.array(rgba, dtype=np.float32))


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    model = mujoco.MjModel.from_xml_path("two_wheel_legged_robot.xml")
    data  = mujoco.MjData(model)

    # ---- set initial pose ----
    # Body position & orientation
    data.qpos[2] = 0.68       # standing height (wheels just touch ground)
    data.qpos[3:7] = [1, 0, 0, 0]   # upright
    # Leg joint angles (standing posture)
    data.qpos[7]  =  0.35     # left hip
    data.qpos[8]  = -0.70     # left knee
    data.qpos[10] =  0.35     # right hip
    data.qpos[11] = -0.70     # right knee

    mujoco.mj_forward(model, data)

    # ---- controller + planner ----
    ctrl = WheeledLeggedController(model, data)
    planner = WaypointPlanner(path="figure8", radius=3.0)
    ctrl.target_xy = planner.current

    dt = model.opt.timestep

    # ---- jump config ----
    JUMP_INTERVAL = 4.0  # seconds between automatic jumps
    next_jump_time = 3.0  # first jump after 3 s of stabilising

    print("=" * 64)
    print("  Two-Wheel Legged Robot – Jump + Navigation Demo")
    print("=" * 64)
    print(f"  Joints : 2 hips + 2 knees + 2 wheels  (nu={model.nu})")
    print(f"  Path   : figure-8  |  Waypoints : {len(planner.waypoints)}")
    print(f"  Jump every {JUMP_INTERVAL}s")
    print(f"  Pitch PID : Kp={ctrl.pitch_pid.kp}  Ki={ctrl.pitch_pid.ki}  Kd={ctrl.pitch_pid.kd}")
    print(f"  Pos   PID : Kp={ctrl.pos_pid.kp}  Ki={ctrl.pos_pid.ki}  Kd={ctrl.pos_pid.kd}")
    print(f"  Yaw   PID : Kp={ctrl.yaw_pid.kp}  Ki={ctrl.yaw_pid.ki}  Kd={ctrl.yaw_pid.kd}")
    print(f"  Hip PD    : Kp={ctrl.hip_kp}  Kd={ctrl.hip_kd}")
    print(f"  Knee PD   : Kp={ctrl.knee_kp}  Kd={ctrl.knee_kd}")
    print("-" * 64)
    print("  Close viewer to quit.  Space = pause.\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_n = 0
        log_interval = 2000
        prev_jump_state = "IDLE"
    
        while viewer.is_running():
            t0 = time.time()
            sim_t = data.time

            # advance waypoint
            px, py = ctrl.xy()
            planner.advance(px, py)
            ctrl.target_xy = planner.current

            # ---- trigger periodic jumps ----
            if sim_t >= next_jump_time and not ctrl.jump.is_jumping:
                ctrl.jump.request_jump(sim_t)
                next_jump_time = sim_t + JUMP_INTERVAL
                print(f"  🦘 JUMP #{ctrl.jump.jump_count + 1} triggered at t={sim_t:.2f}s")

            # When not jumping, hold default stance
            if not ctrl.jump.is_jumping:
                ctrl.des_hip  = ctrl.DEFAULT_HIP
                ctrl.des_knee = ctrl.DEFAULT_KNEE

            # Log jump state transitions
            cur_js = ctrl.jump.state_name
            if cur_js != prev_jump_state:
                if cur_js != "IDLE":
                    print(f"     jump phase: {cur_js}  h={ctrl.height():.3f}m  vz={ctrl.vz():+.2f}")
                prev_jump_state = cur_js

            ctrl.step(dt)
            mujoco.mj_step(model, data)

            # draw waypoints
            viewer.user_scn.ngeom = 0
            draw_waypoints(viewer, planner.waypoints, planner.index)
            viewer.sync()

            step_n += 1
            if step_n % log_interval == 0:
                p_deg = np.degrees(ctrl.pitch())
                y_deg = np.degrees(ctrl.yaw())
                js = ctrl.jump.state_name
                print(f"  step {step_n:7d} | pitch {p_deg:+6.2f}° | "
                      f"yaw {y_deg:+6.1f}° | h={ctrl.height():.3f}m | "
                      f"pos ({px:+.2f},{py:+.2f}) | wp#{planner.index} | {js}")

            # fall detection
            if abs(ctrl.pitch()) > math.radians(60):
                print(f"\n  ⚠  Robot fell! ({np.degrees(ctrl.pitch()):.1f}°) – resetting…")
                time.sleep(1.5)
                mujoco.mj_resetData(model, data)
                data.qpos[2] = 0.68
                data.qpos[3:7] = [1, 0, 0, 0]
                data.qpos[7]  =  0.35
                data.qpos[8]  = -0.70
                data.qpos[10] =  0.35
                data.qpos[11] = -0.70
                mujoco.mj_forward(model, data)
                ctrl.reset_pids()
                ctrl.jump.state = JumpState.IDLE
                ctrl.des_hip  = ctrl.DEFAULT_HIP
                ctrl.des_knee = ctrl.DEFAULT_KNEE
                planner.index = 0
                ctrl.target_xy = planner.current
                next_jump_time = data.time + 3.0
                step_n = 0
                print("  Reset done.\n")

            # real-time pacing
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    print(f"\nSimulation finished. Total jumps: {ctrl.jump.jump_count}")


if __name__ == "__main__":
    main()
