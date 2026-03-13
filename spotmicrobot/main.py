#!/usr/bin/env python3
"""SpotMicro quadruped forward locomotion — correctly calibrated IK.

IK approach: Direct sagittal-plane 2-link IK (thigh + calf) plus a simple
abduction solve. This is derived from the actual MJCF joint definitions and
verified against forward kinematics.

MJCF joint conventions (from spotmicro.xml axis definitions):
  hip_joint:   axis="1 0 0"  → rotation around X  (abduction, +outward)
  thigh_joint: axis="0 1 0"  → rotation around Y  (+forward/down)
  calf_joint:  axis="0 1 0"  → rotation around Y  (negative = elbow-down)

Home pose (FK verified):
  hip=0.0000, thigh=0.8393, calf=-1.4618
  → foot is directly below shoulder at 0.185 m depth ✓

Run with viewer:   python main.py
Headless test:     python main.py --headless --duration 5
"""

import argparse
import math
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

# ── Robot geometry (metres) ──────────────────────────────────────────────
L1 = 0.055    # abduction arm (shoulder lateral offset to hip-pitch pivot)
L3 = 0.1085   # thigh length
L4 = 0.1385   # calf  length

# Standing depth: foot is 0.185 m below shoulder pivot at home pose
STAND_DEPTH = 0.185


@dataclass(frozen=True)
class LegConfig:
    name: str
    side_sign: float      # +1 = left,  -1 = right
    phase_offset: float   # trot phase (0.0 or 0.5)


def ik_sagittal(px: float, pz: float) -> tuple[float, float]:
    """
    2-link planar IK in the sagittal (XZ) plane of the hip-pitch joint.

    Input:
        px  — forward offset of foot relative to hip-pitch pivot (metres)
        pz  — downward distance of foot from hip-pitch pivot (positive down)

    Returns (q_thigh, q_calf) in MJCF joint space:
        q_thigh > 0  → leg swings forward
        q_calf  < 0  → knee bends backward (elbow-down configuration)

    Derivation from MJCF joint axes (both rotate around Y):
        foot_x = -L3*sin(q_thigh) - L4*sin(q_thigh+q_calf)
        foot_z =  L3*cos(q_thigh) + L4*cos(q_thigh+q_calf)

    Wait — let's be careful. With thigh_joint axis=Y and the thigh link
    going from (0,0,0) to (0,0,-L3) in the thigh body frame:
        A rotation of q_thigh around Y transforms (0,0,-L3) to:
          x' =  L3*sin(q_thigh)   [forward]
          z' = -L3*cos(q_thigh)   [downward, negative]
    So:
        foot_x =  L3*sin(q_thigh) + L4*sin(q_thigh+q_calf)
        foot_z = -L3*cos(q_thigh) - L4*cos(q_thigh+q_calf)

    We want foot_x = px (forward), foot_z = -pz (below, so negative).
    Solving:
        D  = sqrt(px²+pz²)
        cos_q3 = (D²-L3²-L4²)/(2·L3·L4)   → q_calf = -acos(cos_q3)
        q_thigh = atan2(px, pz) - atan2(L4·sin(-q_calf), L3+L4·cos(q_calf))

    Verified: at px=0, pz=0.185 → q_thigh=0.8393, q_calf=-1.4618 ✓
    """
    D2 = px * px + pz * pz
    D2 = float(np.clip(D2, 1e-8, (L3 + L4 - 1e-4) ** 2))

    cos_q3 = (D2 - L3 * L3 - L4 * L4) / (2.0 * L3 * L4)
    cos_q3 = float(np.clip(cos_q3, -1.0, 1.0))
    q_calf = -math.acos(cos_q3)   # negative = elbow-down

    q_thigh = math.atan2(px, pz) - math.atan2(
        L4 * math.sin(q_calf),
        L3 + L4 * math.cos(q_calf)
    )
    return q_thigh, q_calf


def ik_abduction(y_target: float, z_target: float, side_sign: float) -> float:
    """
    Solve abduction angle so the hip-pitch pivot sits at lateral offset L1.

    The abduction joint (axis=X) rotates the shoulder arm, which moves the
    hip-pitch pivot in the YZ plane. With hip=0 and shoulder arm length L1:
        pivot_y = L1·cos(q_hip)
        pivot_z = L1·sin(q_hip)   (for left legs; negate for right)

    For normal flat-ground walking we keep q_hip ≈ 0.
    Only solve if a non-zero lateral foot target is requested.
    For our simple trot controller, q_hip = 0 always.
    """
    return 0.0   # flat ground: no abduction needed


class SpotMicroWalker:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data

        # ── Gait parameters ──────────────────────────────────────────────
        self.freq_hz     = 1.6    # stride frequency (Hz)
        self.duty_factor = 0.60   # fraction of cycle in stance (foot on ground)
        self.step_length = 0.06   # total fore-aft foot sweep per stride (m)
        self.step_height = 0.030  # peak lift height (m)

        # Trot: FL+RR in phase 0.0, FR+RL in phase 0.5
        self.legs = [
            LegConfig("FL", +1.0, 0.0),
            LegConfig("FR", -1.0, 0.5),
            LegConfig("RL", +1.0, 0.5),
            LegConfig("RR", -1.0, 0.0),
        ]

        # ── Home joint angles (FK verified) ──────────────────────────────
        self.HOME_HIP   =  0.0000
        self.HOME_THIGH =  0.8393
        self.HOME_CALF  = -1.4618

        # ── Resolve MuJoCo indices ────────────────────────────────────────
        self.joint_ids    = {}
        self.actuator_ids = {}
        self.qvel_ids     = {}
        for leg in self.legs:
            for part in ("hip", "thigh", "calf"):
                jname = f"{leg.name}_{part}_joint"
                aname = f"{leg.name}_{part}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    jname)
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                self.joint_ids   [(leg.name, part)] = jid
                self.actuator_ids[(leg.name, part)] = aid
                self.qvel_ids    [(leg.name, part)] = model.jnt_dofadr[jid]

        # ── PD gains ─────────────────────────────────────────────────────
        self.kp = {"hip": 20.0, "thigh": 32.0, "calf": 32.0}
        self.kd = {"hip":  0.6, "thigh":  1.0, "calf":  1.0}

        # ── Self-test ─────────────────────────────────────────────────────
        q2, q3 = ik_sagittal(0.0, STAND_DEPTH)
        print(f"[IK self-test] home: thigh={q2:.4f} (expect 0.8393), "
              f"calf={q3:.4f} (expect -1.4618)")
        ok = abs(q2 - 0.8393) < 0.001 and abs(q3 - (-1.4618)) < 0.001
        print(f"[IK self-test] {'PASS ✓' if ok else 'FAIL ✗ — check geometry constants'}")

    # ------------------------------------------------------------------ #
    #  Foot trajectory (body/shoulder frame)                               #
    #  px = forward offset from shoulder pivot                             #
    #  pz = downward distance (positive = below shoulder)                 #
    # ------------------------------------------------------------------ #

    def _foot_target(self, leg: LegConfig, t: float) -> tuple[float, float]:
        phase  = (t * self.freq_hz + leg.phase_offset) % 1.0
        sw     = 1.0 - self.duty_factor   # swing fraction = 0.40

        if phase < sw:
            # SWING: lift and carry foot forward (rear → front)
            u  = phase / sw                            # 0 → 1
            px = self.step_length * (u + 0.3)          # -L/2 → +L/2  (forward)
            pz = STAND_DEPTH - self.step_height * math.sin(math.pi * u)
            # pz is reduced (foot lifted) during swing — depth < STAND_DEPTH
        else:
            # STANCE: foot on ground, sweeps front → rear (pushes body forward)
            u  = (phase - sw) / max(1e-9, self.duty_factor)   # 0 → 1
            px = self.step_length * (-0.3 - u)          # +L/2 → -L/2
            pz = STAND_DEPTH

        return px, pz

    # ------------------------------------------------------------------ #
    #  Joint-level PD control                                             #
    # ------------------------------------------------------------------ #

    def _joint_state(self, leg_name: str, part: str) -> tuple[float, float]:
        jid = self.joint_ids[(leg_name, part)]
        return (float(self.data.qpos[self.model.jnt_qposadr[jid]]),
                float(self.data.qvel[self.qvel_ids[(leg_name, part)]]))

    def _apply_pd(self, leg_name: str, part: str, q_des: float, qd_des=0.0):
        q, qd  = self._joint_state(leg_name, part)
        aid    = self.actuator_ids[(leg_name, part)]
        lo, hi = self.model.actuator_ctrlrange[aid]
        tau    = self.kp[part] * (q_des - q) + self.kd[part] * (qd_des - qd)
        self.data.ctrl[aid] = float(np.clip(tau, lo, hi))

    # ------------------------------------------------------------------ #
    #  High-level control                                                 #
    # ------------------------------------------------------------------ #

    def control_step(self, t: float):
        for leg in self.legs:
            px, pz         = self._foot_target(leg, t)
            # Negate px: MuJoCo rotates around +Y which moves child bodies
            # to NEGATIVE X. Our trajectory uses +px = forward, so we flip
            # the sign here so the IK produces the correct thigh angle.
            q_thigh, q_calf = ik_sagittal(-px, pz)

            self._apply_pd(leg.name, "hip",   self.HOME_HIP)
            self._apply_pd(leg.name, "thigh", q_thigh)
            self._apply_pd(leg.name, "calf",  q_calf)

    def hold_home_pose(self):
        for leg in self.legs:
            self._apply_pd(leg.name, "hip",   self.HOME_HIP)
            self._apply_pd(leg.name, "thigh", self.HOME_THIGH)
            self._apply_pd(leg.name, "calf",  self.HOME_CALF)


# ──────────────────────────────────────────────────────────────────────── #

def run_simulation(xml_path: str, duration: float, headless: bool):
    model  = mujoco.MjModel.from_xml_path(xml_path)
    data   = mujoco.MjData(model)
    walker = SpotMicroWalker(model, data)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    warmup  = 1.0           # seconds to hold home before walking
    end_t   = max(duration, warmup + 1.0)

    if headless:
        while data.time < end_t:
            if data.time < warmup:
                walker.hold_home_pose()
            else:
                walker.control_step(data.time - warmup)
            mujoco.mj_step(model, data)
        print(f"Headless run complete. Final body x = "
              f"{data.qpos[0]:.3f} m at t={data.time:.2f}s")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < end_t:
            t0 = time.time()
            if data.time < warmup:
                walker.hold_home_pose()
            else:
                walker.control_step(data.time - warmup)
            mujoco.mj_step(model, data)
            viewer.sync()
            sl = model.opt.timestep - (time.time() - t0)
            if sl > 0:
                time.sleep(sl)


def parse_args():
    p = argparse.ArgumentParser(description="SpotMicro forward walking")
    p.add_argument("--xml",      default="scene.xml",
                   help="MJCF scene file (default: scene.xml)")
    p.add_argument("--duration", type=float, default=20.0,
                   help="Simulation duration seconds (default: 20)")
    p.add_argument("--headless", action="store_true",
                   help="Run without viewer")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(args.xml, args.duration, args.headless)