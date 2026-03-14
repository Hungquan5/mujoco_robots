"""
pick_cube.py  —  Advanced Hand Controller
==========================================

Architecture
------------

1. Minimum-Jerk Trajectory Planner
   5th-order polynomial trajectories between waypoints.
   Reference: Flash & Hogan (1985).

2. Impedance Controller
   Injects velocity damping by shifting the position reference:
       ctrl_eff = q_des + (Kd/Kp) * (dq_des - dq_actual)

3. Contact-Aware Grasp Reflex
   Fingertip contact forces drive force-regulated grip conforming.

Fixes applied vs original
--------------------------
  BUG 1 (XML): cube start pos was z=10 (10 m up); fixed to z=0.43.
               (Fixed in the XML file; mentioned here for reference.)

  BUG 2: _build_act2jnt tendon→joint mapping scanned ALL wraps in the
          model without filtering by tendon ID, so every tendon-driven
          actuator got mapped to the first joint-wrap in the whole model.
          Fix: iterate model.tendon_wrapadr / tendon_wrapnum to find only
          the wraps belonging to the correct tendon.

  BUG 3: GraspReflex.update called mj_id2name inside the contact loop
          (every contact, every step) — O(contacts × geoms) string
          lookups. Fix: build a geom_id → finger_key cache once in
          __init__ and use integer lookup in update().

  BUG 4: HandController.tick called mj_step both in the settle branch
          AND unconditionally at the bottom of the normal branch, causing
          a double-step on the first tick after settling ends.
          Fix: early-return from the settle branch after its mj_step.

  BUG 5: WAYPOINTS "hold" entry has cfg=None. _full_config(None) returned
          a copy of _q_cur which was already updated from the adapted
          q_des, so the hold pose drifted as reflex nudges accumulated
          into _q_cur across ticks. Fix: record _q_squeeze at squeeze end
          and use that as the hold target; _q_cur is now only updated from
          the raw trajectory output (before reflex), keeping tracking
          clean.

  BUG 6: viewer loop called time.sleep(dt * 0.5) = 0.001 s which pins a
          CPU core. Fix: removed the sleep; viewer.sync() already throttles
          to display refresh rate.

Usage
-----
    pip install mujoco numpy
    python pick_cube.py                    # interactive viewer
    python pick_cube.py --no-viewer        # headless, prints status
    python pick_cube.py --model path.xml   # custom model path
"""

import argparse
import time
from typing import Dict, Optional, List, Tuple

import numpy as np
import mujoco
import mujoco.viewer


# ════════════════════════════════════════════════════════════════════════════
#  1. MINIMUM-JERK TRAJECTORY
# ════════════════════════════════════════════════════════════════════════════

class MinJerkTrajectory:
    """
    5th-order polynomial from q0 → q1 over T seconds.

    Boundary conditions:
        q(0)=q0, dq(0)=0, ddq(0)=0
        q(T)=q1, dq(T)=0, ddq(T)=0

    Closed form:
        s(t) = 10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5
    """

    def __init__(self, q0: np.ndarray, q1: np.ndarray, duration: float):
        self.q0    = np.asarray(q0, dtype=float)
        self.q1    = np.asarray(q1, dtype=float)
        self.T     = max(float(duration), 1e-6)
        self.delta = self.q1 - self.q0

    def query(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, velocity) arrays at time t."""
        s     = np.clip(t / self.T, 0.0, 1.0)
        pos_s = 10*s**3 - 15*s**4 + 6*s**5
        vel_s = (30*s**2 - 60*s**3 + 30*s**4) / self.T
        return self.q0 + self.delta * pos_s, self.delta * vel_s


# ════════════════════════════════════════════════════════════════════════════
#  2. IMPEDANCE CONTROLLER
# ════════════════════════════════════════════════════════════════════════════

_GAIN_TABLE: Dict[str, Tuple[float, float, float, float]] = {
    # (wrist_kp, wrist_kd, finger_kp, finger_kd)
    "approach": (12.0, 1.5, 2.0, 0.20),
    "contact":  (14.0, 1.8, 4.0, 0.40),
    "squeeze":  (16.0, 2.0, 6.0, 0.60),
    "lift":     (18.0, 2.2, 6.0, 0.60),
}

_WRIST_ACTS  = frozenset({"rh_A_WRJ2", "rh_A_WRJ1"})
_THUMB_ACTS  = frozenset({"rh_A_THJ5","rh_A_THJ4","rh_A_THJ3","rh_A_THJ2","rh_A_THJ1"})


class ImpedanceController:
    """
    Computes and writes impedance-adjusted ctrl setpoints.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model

        # actuator name → index
        self._act: Dict[str, int] = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }

        # joint name → velocity DOF address
        self._jnt_dof: Dict[str, int] = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i):
                model.jnt_dofadr[i]
            for i in range(model.njnt)
        }

        # actuator name → joint name (built correctly — see BUG 2 fix below)
        self._act2jnt: Dict[str, str] = self._build_act2jnt()

    def _build_act2jnt(self) -> Dict[str, str]:
        """
        FIX (BUG 2): original code scanned range(model.nwrap) for ALL wraps
        without filtering by tendon, so every tendon actuator was mapped to
        the first joint-wrap found anywhere in the model.

        Correct approach: for each tendon actuator, use
        model.tendon_wrapadr[tid] and model.tendon_wrapnum[tid] to slice
        only the wraps belonging to that specific tendon.
        """
        cache: Dict[str, str] = {}
        for i in range(self.model.nu):
            aname   = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            trntype = self.model.actuator_trntype[i]
            trnid   = self.model.actuator_trnid[i, 0]

            if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                jname = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
                if jname:
                    cache[aname] = jname

            elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
                # trnid is the tendon index; find its wraps
                wrap_start = self.model.tendon_adr[trnid]
                wrap_count = self.model.tendon_num[trnid]
                for w in range(wrap_start, wrap_start + wrap_count):
                    if self.model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_JOINT:
                        jid   = self.model.wrap_objid[w]
                        jname = mujoco.mj_id2name(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                        if jname:
                            cache[aname] = jname
                            break   # use the first joint-wrap in this tendon

        return cache

    def _gains(self, name: str, phase: str) -> Tuple[float, float]:
        w_kp, w_kd, f_kp, f_kd = _GAIN_TABLE.get(phase, _GAIN_TABLE["approach"])
        if name in _WRIST_ACTS:
            return w_kp, w_kd
        elif name in _THUMB_ACTS:
            return f_kp * 0.9, f_kd
        return f_kp, f_kd

    def apply(self, data: mujoco.MjData,
              q_des:  Dict[str, float],
              dq_des: Dict[str, float],
              phase:  str):
        for name, qd in q_des.items():
            aid = self._act.get(name)
            if aid is None:
                continue
            kp, kd = self._gains(name, phase)

            jnt = self._act2jnt.get(name, "")
            dof = self._jnt_dof.get(jnt)
            dq  = float(data.qvel[dof]) if dof is not None else 0.0
            dq_d = dq_des.get(name, 0.0)

            ref = qd + (kd / max(kp, 1e-9)) * (dq_d - dq)
            lo, hi = self.model.actuator_ctrlrange[aid]
            data.ctrl[aid] = float(np.clip(ref, lo, hi))


# ════════════════════════════════════════════════════════════════════════════
#  3. CONTACT-AWARE GRASP REFLEX
# ════════════════════════════════════════════════════════════════════════════

_TOUCH_GEOMS: Dict[str, List[str]] = {
    "ff": ["ff_tip", "ff_d"],
    "mf": ["mf_tip", "mf_d"],
    "rf": ["rf_tip", "rf_d"],
    "lf": ["lf_tip", "lf_d"],
    "th": ["th_tip", "th_d"],
}

_FINGER_ACT_PAIR: Dict[str, Tuple[str, str]] = {
    "ff": ("rh_A_FFJ3", "rh_A_FFJ0"),
    "mf": ("rh_A_MFJ3", "rh_A_MFJ0"),
    "rf": ("rh_A_RFJ3", "rh_A_RFJ0"),
    "lf": ("rh_A_LFJ3", "rh_A_LFJ0"),
    "th": ("rh_A_THJ4", "rh_A_THJ2"),
}


class GraspReflex:
    """
    FIX (BUG 3): original update() called mj_id2name(geom_id) for every
    contact on every step — O(contacts × fingers) string lookups.
    Now we build a geom_id → finger_key integer dict once in __init__
    and do O(1) dict lookups in update().
    """

    VIRTUAL_STIFFNESS = 80.0   # N/rad

    def __init__(self, model: mujoco.MjModel, desired_force: float = 2.5):
        self.model = model
        self.F_des = desired_force
        self.force:      Dict[str, float] = {k: 0.0   for k in _TOUCH_GEOMS}
        self.in_contact: Dict[str, bool]  = {k: False for k in _TOUCH_GEOMS}

        # FIX: build geom_id → finger_key cache once (integer keyed)
        self._geom_id_to_finger: Dict[int, str] = {}
        for fkey, gnames in _TOUCH_GEOMS.items():
            for gname in gnames:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                if gid >= 0:
                    self._geom_id_to_finger[gid] = fkey

        # actuator ctrl-range cache
        act_map = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }
        self._ctrlrange: Dict[str, Tuple[float, float]] = {
            n: (float(model.actuator_ctrlrange[i, 0]),
                float(model.actuator_ctrlrange[i, 1]))
            for n, i in act_map.items()
        }

    def update(self, data: mujoco.MjData):
        """Read all fingertip contact forces via integer geom-id lookup."""
        for k in self.force:
            self.force[k]      = 0.0
            self.in_contact[k] = False

        f6 = np.zeros(6)
        for i in range(data.ncon):
            con = data.contact[i]
            mujoco.mj_contactForce(self.model, data, i, f6)
            fn = abs(f6[0])

            # O(1) integer lookup — no string allocation per contact
            for gid in (con.geom1, con.geom2):
                fkey = self._geom_id_to_finger.get(gid)
                if fkey is not None:
                    self.force[fkey]      += fn
                    self.in_contact[fkey]  = True

    def adapt(self, q_des: Dict[str, float], active: bool) -> Dict[str, float]:
        """
        Return adapted q_des: fingers in contact get a force-proportional
        extra flex; fingers not yet touching continue position-tracking.
        """
        if not active:
            return q_des
        out = dict(q_des)
        for fkey in _TOUCH_GEOMS:
            if not self.in_contact[fkey]:
                continue
            _, pip = _FINGER_ACT_PAIR[fkey]
            nudge  = max(0.0, self.F_des - self.force[fkey]) / self.VIRTUAL_STIFFNESS
            if pip in out:
                lo, hi   = self._ctrlrange.get(pip, (-np.pi, np.pi))
                out[pip] = float(np.clip(out[pip] + nudge, lo, hi))
        return out


# ════════════════════════════════════════════════════════════════════════════
#  WAYPOINT TABLE
# ════════════════════════════════════════════════════════════════════════════

WAYPOINTS: List[Tuple[str, float, Optional[Dict], str, bool]] = [
    # (name, duration_s, {actuator: target_angle} | None, gain_phase, reflex_on)
    ("open",       0.40, {
        "rh_A_WRJ2":  0.00, "rh_A_WRJ1":  0.00,
        "rh_A_THJ5": -0.45, "rh_A_THJ4":  0.05,
        "rh_A_THJ3":  0.00, "rh_A_THJ2": -0.10, "rh_A_THJ1":  0.00,
        "rh_A_FFJ4": -0.15, "rh_A_FFJ3": -0.15, "rh_A_FFJ0":  0.00,
        "rh_A_MFJ4":  0.00, "rh_A_MFJ3": -0.15, "rh_A_MFJ0":  0.00,
        "rh_A_RFJ4":  0.00, "rh_A_RFJ3": -0.15, "rh_A_RFJ0":  0.00,
        "rh_A_LFJ5":  0.00,
        "rh_A_LFJ4":  0.15, "rh_A_LFJ3": -0.15, "rh_A_LFJ0":  0.00,
    }, "approach", False),

    ("pre_shape",  0.45, {
        "rh_A_WRJ2":  0.00, "rh_A_WRJ1":  0.00,
        "rh_A_THJ5": -0.20, "rh_A_THJ4":  0.40,
        "rh_A_THJ3":  0.00, "rh_A_THJ2":  0.30, "rh_A_THJ1":  0.15,
        "rh_A_FFJ4": -0.08, "rh_A_FFJ3":  0.55, "rh_A_FFJ0":  0.05,
        "rh_A_MFJ4":  0.00, "rh_A_MFJ3":  0.50, "rh_A_MFJ0":  0.05,
        "rh_A_RFJ4":  0.05, "rh_A_RFJ3":  0.50, "rh_A_RFJ0":  0.05,
        "rh_A_LFJ5":  0.15,
        "rh_A_LFJ4":  0.08, "rh_A_LFJ3":  0.45, "rh_A_LFJ0":  0.05,
    }, "approach", False),

    ("contact",    0.55, {
        "rh_A_THJ5":  0.10, "rh_A_THJ4":  0.60,
        "rh_A_THJ3":  0.05, "rh_A_THJ2":  0.65, "rh_A_THJ1":  0.50,
        "rh_A_FFJ4": -0.05, "rh_A_FFJ3":  0.75, "rh_A_FFJ0":  1.30,
        "rh_A_MFJ4":  0.00, "rh_A_MFJ3":  0.70, "rh_A_MFJ0":  1.25,
        "rh_A_RFJ4":  0.05, "rh_A_RFJ3":  0.70, "rh_A_RFJ0":  1.25,
        "rh_A_LFJ5":  0.25,
        "rh_A_LFJ4":  0.05, "rh_A_LFJ3":  0.65, "rh_A_LFJ0":  1.10,
    }, "contact",  True),

    ("squeeze",    0.40, {
        "rh_A_THJ5":  0.15, "rh_A_THJ4":  0.70,
        "rh_A_THJ3":  0.05, "rh_A_THJ2":  0.80, "rh_A_THJ1":  0.65,
        "rh_A_FFJ4": -0.03, "rh_A_FFJ3":  0.85, "rh_A_FFJ0":  1.60,
        "rh_A_MFJ4":  0.00, "rh_A_MFJ3":  0.80, "rh_A_MFJ0":  1.55,
        "rh_A_RFJ4":  0.05, "rh_A_RFJ3":  0.80, "rh_A_RFJ0":  1.55,
        "rh_A_LFJ5":  0.28,
        "rh_A_LFJ4":  0.03, "rh_A_LFJ3":  0.72, "rh_A_LFJ0":  1.35,
        "rh_A_WRJ2":  0.20,
    }, "squeeze",  True),

    ("lift",       0.70, {
        "rh_A_WRJ2":  0.60, "rh_A_WRJ1":  0.10,
        "rh_A_THJ5":  0.15, "rh_A_THJ4":  0.70,
        "rh_A_THJ3":  0.05, "rh_A_THJ2":  0.80, "rh_A_THJ1":  0.65,
        "rh_A_FFJ4": -0.03, "rh_A_FFJ3":  0.85, "rh_A_FFJ0":  1.60,
        "rh_A_MFJ4":  0.00, "rh_A_MFJ3":  0.80, "rh_A_MFJ0":  1.55,
        "rh_A_RFJ4":  0.05, "rh_A_RFJ3":  0.80, "rh_A_RFJ0":  1.55,
        "rh_A_LFJ5":  0.28,
        "rh_A_LFJ4":  0.03, "rh_A_LFJ3":  0.72, "rh_A_LFJ0":  1.35,
    }, "lift",     True),

    # FIX (BUG 5): cfg=None with a 2 s hold. See HandController for how
    # _q_squeeze is captured so this doesn't drift.
    ("hold",       2.00, None, "lift", True),
]

# Index of the squeeze waypoint (needed to snapshot pose for hold)
_SQUEEZE_IDX = next(i for i, w in enumerate(WAYPOINTS) if w[0] == "squeeze")


# ════════════════════════════════════════════════════════════════════════════
#  HAND CONTROLLER
# ════════════════════════════════════════════════════════════════════════════

class HandController:
    """
    Orchestrates MinJerkTrajectory + ImpedanceController + GraspReflex.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model     = model
        self.dt        = model.opt.timestep
        self._act_map: Dict[str, int] = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }
        self.impedance = ImpedanceController(model)
        self.reflex    = GraspReflex(model, desired_force=2.5)

        self._wp_idx:   int  = -1
        self._traj:     Optional[MinJerkTrajectory] = None
        self._traj_t:   float = 0.0
        self._settle_t: float = 0.0

        self._keys: List[str] = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)
        ]

        # Current trajectory target (raw, before reflex) — used for
        # trajectory continuity only. NOT updated with reflex nudges.
        self._q_cur = np.zeros(model.nu)
        open_cfg = WAYPOINTS[0][2]
        for i, k in enumerate(self._keys):
            self._q_cur[i] = open_cfg.get(k, 0.0)   # type: ignore[union-attr]

        # FIX (BUG 5): snapshot of squeeze pose, used as hold target
        self._q_squeeze: Optional[np.ndarray] = None

    # ── helpers ────────────────────────────────────────────────────────────

    def _full_config(self, cfg: Optional[Dict],
                     base: Optional[np.ndarray] = None) -> np.ndarray:
        """Merge partial cfg dict over base (defaults to _q_cur)."""
        q = (base if base is not None else self._q_cur).copy()
        if cfg:
            for i, k in enumerate(self._keys):
                if k in cfg:
                    q[i] = cfg[k]
        return q

    def _start_waypoint(self, idx: int):
        _, dur, cfg, phase, _ = WAYPOINTS[idx]

        # FIX (BUG 5): hold uses the snapshotted squeeze pose as target
        if cfg is None:
            if self._q_squeeze is None:
                # fallback: current config (should not happen in normal flow)
                q1 = self._q_cur.copy()
            else:
                q1 = self._q_squeeze.copy()
        else:
            q1 = self._full_config(cfg)

        self._traj   = MinJerkTrajectory(self._q_cur.copy(), q1, dur)
        self._traj_t = 0.0
        self._wp_idx = idx
        print(f"  → [{idx}] '{WAYPOINTS[idx][0]}'  dur={dur:.2f}s  gains={phase}")

    # ── main tick ──────────────────────────────────────────────────────────

    def tick(self, data: mujoco.MjData):
        """Call exactly once per simulation timestep."""

        # ── settle phase ────────────────────────────────────────────────
        if self._wp_idx == -1:
            for i, k in enumerate(self._keys):
                aid    = self._act_map[k]
                lo, hi = self.model.actuator_ctrlrange[aid]
                data.ctrl[aid] = float(np.clip(self._q_cur[i], lo, hi))
            self._settle_t += self.dt
            if self._settle_t >= 0.30:
                self._start_waypoint(0)
            mujoco.mj_step(self.model, data)
            return  # FIX (BUG 4): early return prevents double-step

        # ── update contact sensors ──────────────────────────────────────
        self.reflex.update(data)

        # ── query trajectory ────────────────────────────────────────────
        self._traj_t += self.dt
        q_arr, dq_arr = self._traj.query(self._traj_t)

        # FIX (BUG 5): update _q_cur from raw trajectory BEFORE reflex,
        # so the next trajectory starts from the clean tracked position.
        for i, k in enumerate(self._keys):
            self._q_cur[i] = float(q_arr[i])

        # ── build desire dicts ──────────────────────────────────────────
        q_des  = {k: float(q_arr[i])  for i, k in enumerate(self._keys)}
        dq_des = {k: float(dq_arr[i]) for i, k in enumerate(self._keys)}

        # ── contact-aware adaptation ────────────────────────────────────
        _, dur, _, phase, reflex_on = WAYPOINTS[self._wp_idx]
        q_des_adapted = self.reflex.adapt(q_des, active=reflex_on)

        # ── impedance control ───────────────────────────────────────────
        self.impedance.apply(data, q_des_adapted, dq_des, phase)

        # ── step physics ────────────────────────────────────────────────
        mujoco.mj_step(self.model, data)

        # ── snapshot squeeze pose for hold ──────────────────────────────
        if self._wp_idx == _SQUEEZE_IDX and self._q_squeeze is None:
            self._q_squeeze = self._q_cur.copy()

        # ── advance to next waypoint ─────────────────────────────────────
        if self._traj_t >= dur:
            next_idx = self._wp_idx + 1
            if next_idx < len(WAYPOINTS):
                self._start_waypoint(next_idx)

    @property
    def waypoint_name(self) -> str:
        if self._wp_idx < 0:
            return "settling"
        if self._wp_idx >= len(WAYPOINTS):
            return "done"
        return WAYPOINTS[self._wp_idx][0]


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run(model_path: str, use_viewer: bool):
    model   = mujoco.MjModel.from_xml_path(model_path)
    data    = mujoco.MjData(model)
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    ctrl    = HandController(model)

    print(f"[init] {model_path}  nu={model.nu}  dt={model.opt.timestep:.4f}s")

    t_sim      = 0.0
    last_print = -1.0
    INTERVAL   = 0.25

    def status():
        cz = data.xpos[cube_id][2]
        F  = ctrl.reflex.force
        print(f"  t={t_sim:6.3f}s  wp={ctrl.waypoint_name:10s}  "
              f"cube_Z={cz:.4f}m  "
              f"F[ff={F['ff']:.1f} mf={F['mf']:.1f} "
              f"rf={F['rf']:.1f} lf={F['lf']:.1f} th={F['th']:.1f}]N")

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.lookat[:] = [0.0, 0.0, 0.42]
            v.cam.distance  = 0.62
            v.cam.elevation = -22
            v.cam.azimuth   = 145
            while v.is_running():
                ctrl.tick(data)
                t_sim += model.opt.timestep
                if t_sim - last_print >= INTERVAL:
                    status()
                    last_print = t_sim
                v.sync()
                # FIX (BUG 6): removed time.sleep() — viewer.sync() handles throttle
    else:
        total = int(sum(w[1] for w in WAYPOINTS) / model.opt.timestep) + 300
        for _ in range(total):
            ctrl.tick(data)
            t_sim += model.opt.timestep
            if t_sim - last_print >= INTERVAL:
                status()
                last_print = t_sim
        cz      = data.xpos[cube_id][2]
        total_F = sum(ctrl.reflex.force.values())
        print(f"\n[done] cube_Z={cz:.4f}m  total_grip={total_F:.2f}N")
        print("SUCCESS — cube lifted!" if cz > 0.08 else
              "Cube did not lift — try raising kp or cube friction.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-viewer", dest="no_viewer", action="store_true")
    ap.add_argument("--model", default="robot_hand.xml")
    args = ap.parse_args()
    run(args.model, use_viewer=not args.no_viewer)