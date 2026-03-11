#!/usr/bin/env python3
"""
6-DOF Industrial Cobot Arm – Jacobian-based IK Demo
=====================================================
Instead of hard-coding joint angles, we now give the robot a
TARGET POSITION in world-space (x, y, z) and let the Jacobian IK
solver figure out what joint angles are needed.

How it works (high-level):
  1. Compute the JACOBIAN  J  (3×6) – relates small joint-angle
     changes Δq to small end-effector displacements Δx:
           Δx  =  J · Δq
  2. Invert that relationship to get joint changes from a position error:
           Δq  =  J⁺ · Δx          (J⁺ = pseudo-inverse of J)
  3. Repeat until the end-effector reaches the target.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time


# ══════════════════════════════════════════════════════════════════════════════
#  ROBOT ARM CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class RobotArmController:
    def __init__(self, model_path):
        """Load the MuJoCo model and resolve all IDs we need."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)

        # 6 revolute arm joints + 1 gripper (coupled prismatic fingers)
        self.joint_names    = ['joint1', 'joint2', 'joint3',
                               'joint4', 'joint5', 'joint6',
                               'left_finger_joint']
        self.actuator_names = ['m_joint1', 'm_joint2', 'm_joint3',
                               'm_joint4', 'm_joint5', 'm_joint6',
                               'm_gripper']

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.joint_names
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.actuator_names
        ]

        # Tool Center Point site (the red dot on the gripper tip)
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')

        # Cache the arm DOF indices (computed once, reused every IK step)
        self._arm_dof_ids = self._get_arm_dof_ids()
        print(f"  Arm DOF indices in nv space: {self._arm_dof_ids}")

        # Target object – body ID for live position reads, geom for size
        self.target_obj_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_object')
        target_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, 'target_box')
        # model.geom_size is (ngeom, 3); for a box each entry is the half-size
        self.target_half_z = self.model.geom_size[target_geom_id, 2]

        # Touch sensors on the finger pads
        # model.sensor_adr[sid] gives the start index in sensordata for sensor sid
        l_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'left_touch')
        r_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'right_touch')
        self._left_touch_adr  = int(self.model.sensor_adr[l_sid])
        self._right_touch_adr = int(self.model.sensor_adr[r_sid])

        # Offset from ee_site to finger-pad centre along the approach axis.
        # From XML: pad local-z in gripper_body = 0.046(finger) + 0.032(pad) = 0.078
        #           ee_site local-z in gripper_body = 0.060
        # When the arm points down (+local-Z → -world-Z):
        #   pads are (0.078 - 0.060) = 0.018 m BELOW the ee_site in world frame.
        # So: ee_site_target_z = desired_pad_z + EE_TO_PAD_Z
        self.EE_TO_PAD_Z = 0.078 - 0.060   # 0.018 m

    # ─────────────────────────────────────────────────────────────────────────
    #  HELPER – get arm DOF indices
    # ─────────────────────────────────────────────────────────────────────────
    def _get_arm_dof_ids(self):
        """
        WHY THIS IS NEEDED
        ──────────────────
        MuJoCo stores the whole model state in a flat vector called qpos, and
        velocities in qvel (size = nv).  The Jacobian J from mj_jacSite has
        shape (3, nv) — one column PER DOF across the ENTIRE model, including
        free-joints of free-flying objects (like our target box), multi-DOF
        joints, etc.

        We only care about the 6 columns that belong to joint1…joint6.
        model.jnt_dofadr[joint_id] gives the index of the FIRST DOF for that
        joint inside nv.  For a 1-DOF hinge that is the only DOF, so this is
        directly the column index we want to slice from J.
        """
        arm_joint_ids = self.joint_ids[:6]   # exclude gripper joint
        return [self.model.jnt_dofadr[jid] for jid in arm_joint_ids]

    # ─────────────────────────────────────────────────────────────────────────
    #  BASIC HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def get_joint_positions(self):
        """Current ctrl targets [j1..j6, gripper]."""
        return self.data.ctrl[self.actuator_ids].copy()

    def set_joint_targets(self, targets):
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = targets[i]

    def get_ee_position(self):
        """TCP position in world frame (x, y, z) – live from sensor."""
        return self.data.sensordata[0:3].copy()

    def get_target_position(self):
        """Live (x, y, z) of the target object, read from data.xpos every call.
        Because the object has a freejoint it can be pushed around by physics,
        so we never cache this – always pull fresh from the sim state."""
        return self.data.xpos[self.target_obj_id].copy()

    def get_touch_forces(self):
        """Return (left_N, right_N) – normal contact force on each finger pad.
        A touch sensor in MuJoCo sums all normal contact forces whose contact
        point falls inside the sensor's *site* volume (left_pad_site /
        right_pad_site). Value is 0 when no contact, positive otherwise."""
        return (float(self.data.sensordata[self._left_touch_adr]),
                float(self.data.sensordata[self._right_touch_adr]))

    # ─────────────────────────────────────────────────────────────────────────
    #  PART 1 – COMPUTE THE JACOBIAN
    # ─────────────────────────────────────────────────────────────────────────
    def compute_jacobian(self):
        """
        Build the 3×6 positional Jacobian J for the end-effector site.

        WHAT IS THE JACOBIAN?
        ─────────────────────
        The Jacobian J maps a small change in joint angles Δq (6-vector)
        to the resulting small change in end-effector POSITION Δx (3-vector):

              Δx  =  J · Δq       (J is 3×6)

        Column j of J answers:
            "If I nudge joint j by a tiny amount, how does the EE move?"

        MuJoCo API — mj_jacSite fills two (3, nv) buffers:
          • jacp  – positional Jacobian  (what we need)
          • jacr  – rotational Jacobian  (we ignore for now)

        We slice out the 6 columns for our arm joints using _arm_dof_ids.
        """
        jacp = np.zeros((3, self.model.nv))   # positional, shape (3, nv)
        jacr = np.zeros((3, self.model.nv))   # rotational (unused here)

        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        # Keep only the 6 arm-joint columns  →  final shape (3, 6)
        return jacp[:, self._arm_dof_ids]

    # ─────────────────────────────────────────────────────────────────────────
    #  PART 2 – SINGLE IK STEP  (Moore-Penrose pseudo-inverse)
    # ─────────────────────────────────────────────────────────────────────────
    def ik_step(self, target_pos, alpha=0.5):
        """
        One IK iteration: compute Δq that nudges the EE toward target_pos.

        MATH
        ────
        We want:   J · Δq  =  Δx          (Δx = target – current EE pos)

        J is 3×6 — more unknowns than equations, so infinitely many Δq
        satisfy it.  The MINIMUM-NORM solution uses the Moore-Penrose
        pseudo-inverse J⁺:

              Δq  =  J⁺ · Δx

        np.linalg.pinv computes J⁺ via SVD:
          1. Decompose  J = U · Σ · Vᵀ
          2. Invert non-zero singular values:  Σ⁺ = diag(1/σᵢ)
          3. J⁺ = V · Σ⁺ · Uᵀ          shape (6, 3)

        This gives the joint correction with the SMALLEST possible ‖Δq‖
        that exactly achieves the requested Δx — as long as J has full
        row rank (i.e. the arm is not at a singularity).

        NOTE: near singularities the singular values σᵢ → 0, so 1/σᵢ → ∞
        and Δq can become very large.  That is the trade-off vs DLS.

        PARAMS
          alpha – step size / learning rate (0 < α ≤ 1).
        """
        # ── 2a: position error (3D) ───────────────────────────────────────────
        error = target_pos - self.get_ee_position()   # shape (3,)

        # ── 2b: Jacobian at current configuration ─────────────────────────────
        J = self.compute_jacobian()                   # shape (3, 6)

        # ── 2c: Moore-Penrose pseudo-inverse via SVD ──────────────────────────
        #   pinv handles rank-deficient J gracefully by zeroing out
        #   singular values below the numerical tolerance (rcond threshold).
        J_pinv = np.linalg.pinv(J)                   # shape (6, 3)

        # ── 2d: joint correction ──────────────────────────────────────────────
        dq = J_pinv @ error                           # (6,)

        # ── 2e: clamp dq to prevent singularity blowup ───────────────────────
        # Near singularities, 1/σᵢ → ∞ so dq can become enormous.
        # We cap the total correction to 0.2 rad per step; this is the
        # lightweight alternative to DLS damping.
        dq_norm = np.linalg.norm(dq)
        if dq_norm > 0.2:
            dq = dq * (0.2 / dq_norm)

        # ── 2f: update servo set-points, respecting joint limits ──────────────
        for i, aid in enumerate(self.actuator_ids[:6]):
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = float(
                np.clip(self.data.ctrl[aid] + alpha * dq[i], lo, hi)
            )

        return error, float(np.linalg.norm(error))

    # ─────────────────────────────────────────────────────────────────────────
    #  PART 3 – ITERATIVE IK LOOP
    # ─────────────────────────────────────────────────────────────────────────
    def move_to_cartesian(self, target_pos,
                          tolerance=0.005,
                          max_iters=1500,
                          alpha=0.1,
                          sim_steps=8,
                          viewer=None):
        """
        Move the end-effector to target_pos (x, y, z in world metres).

        HOW THE LOOP WORKS
        ──────────────────
        Each iteration:
          1. ik_step()        → compute new ctrl targets (IK math, once)
          2. mj_step() × N   → advance physics N steps so the arm
                               physically moves toward the new targets
          3. viewer sync      → we see it move in real-time
          4. Check ‖error‖ < tolerance → done

        WHY sim_steps > 1?
        The simulation timestep is 2 ms.  Running 1 IK update per 2 ms
        means we fire 500 new ctrl targets per second, but the arm joints
        (kp=400, inertia~4 kg) physically move at ~10 rad/s — they can
        never keep up.  The Jacobian ends up computed at the wrong config
        and the loop gets stuck.  Running sim_steps=8 per IK update gives
        the arm 16 ms to physically follow each correction before we
        recompute, which matches the servo bandwidth.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        print(f"  → IK target : {np.round(target_pos, 4)}")

        err_norm = float('inf')
        for i in range(max_iters):
            _, err_norm = self.ik_step(target_pos, alpha=alpha)
            for _ in range(sim_steps):
                mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

            if (i + 1) % 200 == 0:
                print(f"    iter {i+1:4d}  |error| = {err_norm*1000:.2f} mm")

            if err_norm < tolerance:
                print(f"  ✓ Converged at iter {i+1},  "
                      f"residual = {err_norm*1000:.2f} mm")
                break
        else:
            print(f"  ✗ Max iters reached,  "
                  f"residual = {err_norm*1000:.2f} mm")

        print(f"    EE pos : {np.round(self.get_ee_position(), 4)}")

    # ─────────────────────────────────────────────────────────────────────────
    #  GRIPPER HELPERS  (direct joint-space – not affected by IK)
    # ─────────────────────────────────────────────────────────────────────────
    def _set_gripper(self, value, iters=300, viewer=None):
        start = self.data.ctrl[self.actuator_ids[6]]
        gid   = self.actuator_ids[6]
        for i in range(iters):
            t = (1 - np.cos(i / iters * np.pi)) / 2
            self.data.ctrl[gid] = start + t * (value - start)
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

    def open_gripper(self, viewer=None):
        print("  Opening gripper …")
        self._set_gripper(0.04, viewer=viewer)

    def close_gripper(self, contact_threshold=0.3, viewer=None):
        """
        Close the gripper incrementally and STOP the moment both pads
        detect contact above contact_threshold Newtons.

        WHY NOT JUST COMMAND ctrl=0?
        ─────────────────────────────
        A pure position command drives fingers to 0 regardless of what's
        between them.  If the descent height is even a few mm off, the
        fingers slide past the box sides and close on air.

        With touch sensors we stop AS SOON AS both pads feel the object,
        locking the gripper at the exact depth where it has real contact.
        This also prevents crushing a fragile object.
        """
        print("  Closing gripper (force feedback) …")
        gid = self.actuator_ids[6]
        # Shrink ctrl by 0.2 mm per sim step → smooth, slow close
        step_size = 0.0002

        for _ in range(3000):   # safety cap: ~6 s max
            self.data.ctrl[gid] = max(0.0, self.data.ctrl[gid] - step_size)
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

            lf, rf = self.get_touch_forces()
            if lf > contact_threshold and rf > contact_threshold:
                print(f"    ✓ Contact! L={lf:.1f} N  R={rf:.1f} N  – gripper locked")
                return

        lf, rf = self.get_touch_forces()
        print(f"    Closed fully (no contact detected). L={lf:.1f} N  R={rf:.1f} N")

    def stabilize(self, n_steps=500, viewer=None):
        """
        Run N physics steps with NO IK updates.

        WHY THIS IS NEEDED
        ──────────────────
        When the gripper closes and makes contact with the object, MuJoCo's
        contact solver generates large corrective impulses to enforce the
        non-penetration constraint.  These forces spike and then decay over
        tens of milliseconds as the solver iterates to equilibrium.

        If we immediately fire IK ctrl updates on top of those unstable
        contact forces, the arm reacts to both the IK correction AND the
        contact impulse simultaneously — producing the wild up/down oscillation.

        Stabilize lets the contact forces settle to steady-state BEFORE we
        start commanding new joint targets.
        """
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

    # ─────────────────────────────────────────────────────────────────────────
    #  JOINT-SPACE MOVE  (kept for homing only)
    # ─────────────────────────────────────────────────────────────────────────
    def move_to_joint_positions(self, target_q, iters=500, viewer=None):
        """Cosine-interpolated joint-space move (used only for the home pose)."""
        start = self.get_joint_positions()
        for i in range(iters):
            t = (1 - np.cos(i / iters * np.pi)) / 2
            self.set_joint_targets(start + t * (target_q - start))
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()
        print(f"    EE pos : {np.round(self.get_ee_position(), 4)}")


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO – PICK & PLACE WITH CARTESIAN IK
# ══════════════════════════════════════════════════════════════════════════════

def demo_ik_pick_and_place(controller, viewer=None):
    """
    Pick & place using CARTESIAN waypoints (x, y, z in metres).

    The robot no longer needs hand-tuned joint angles for each pose.
    We just say "go HERE" and the Jacobian IK works out the joints.

    Scene layout (from the XML):
      • Robot base at world origin (0, 0, 0)
      • Target red box at  (0.45, 0.15, 0.05)   – on the floor
      • Place spot at     (-0.35, 0.30, 0.10)   – to the left
    """
    print("\n" + "=" * 60)
    print("  Jacobian IK – Pick & Place Demo")
    print("=" * 60)

    # ── Home in joint-space (safe starting pose) ─────────────────────────────
    #                     j1    j2     j3     j4   j5    j6   grip
    home_q = np.array([ 0.0, -0.8,  -0.4,  0.0, -0.6,  0.0, 0.04])
    print("\n[0] Moving to home joint pose …")
    controller.move_to_joint_positions(home_q, iters=600, viewer=viewer)

    # ── Cartesian waypoints ───────────────────────────────────────────────────
    # Place location is a user-defined drop-off spot (fine to keep fixed).
    place_xy  = np.array([-0.35, 0.30])
    place_z   = 0.18
    hover_clearance = 0.15   # how far above the box top to hover

    print("\nOpening gripper …")
    controller.open_gripper(viewer=viewer)

    # Step 1 – read object position LIVE, hover above it
    print("\n[1] Pre-grasp – hover above object")
    obj = controller.get_target_position()
    box_top = obj[2] + controller.target_half_z
    controller.move_to_cartesian(
        [obj[0], obj[1], box_top + hover_clearance],
        tolerance=0.008, viewer=viewer)

    # Step 2 – re-read position just before descending (box may have shifted)
    # TARGET HEIGHT: put finger PADS at the box midplane (obj[2] = box centre).
    # Pads are EE_TO_PAD_Z = 0.018 m below ee_site when arm points down,
    # so we command:  ee_site_z = obj[2] + EE_TO_PAD_Z
    # Previously we used box_top + 0.01 = 0.085 m, which put pads at 0.067 m
    # – ABOVE the box top (0.075 m). Fingers slid past the box completely.
    print("\n[2] Descend to grip height")
    obj = controller.get_target_position()
    controller.move_to_cartesian(
        [obj[0], obj[1], obj[2] + controller.EE_TO_PAD_Z],
        tolerance=0.006, alpha=0.08, viewer=viewer)

    # Step 3 – grip, then wait for contact forces to settle
    print("\n[3] Closing gripper")
    controller.close_gripper(viewer=viewer)
    print("    Stabilising contact forces …")
    controller.stabilize(n_steps=400, viewer=viewer)

    # Step 4 – lift (use last known obj XY; it's now attached to the gripper)
    # sim_steps=20 and alpha=0.05: much gentler so the carried mass doesn't
    # bounce against the IK corrections.
    print("\n[4] Lifting object")
    obj = controller.get_target_position()
    controller.move_to_cartesian(
        [obj[0], obj[1], box_top + hover_clearance + 0.08],
        tolerance=0.008, alpha=0.05, sim_steps=20, viewer=viewer)

    # Step 5 – sweep to place XY at safe height
    print("\n[5] Moving to place location")
    controller.move_to_cartesian(
        [place_xy[0], place_xy[1], place_z + hover_clearance],
        tolerance=0.008, alpha=0.05, sim_steps=20, viewer=viewer)

    # Step 6 – lower to place height
    print("\n[6] Lowering to place height")
    controller.move_to_cartesian(
        [place_xy[0], place_xy[1], place_z],
        tolerance=0.008, alpha=0.04, sim_steps=20, viewer=viewer)

    # Step 7 – release
    print("\n[7] Releasing object")
    controller.open_gripper(viewer=viewer)

    # Step 8 – retreat
    print("\n[8] Retreating up")
    controller.move_to_cartesian(
        [place_xy[0], place_xy[1], place_z + hover_clearance],
        tolerance=0.010, viewer=viewer)

    # Step 9 – back home (joint-space)
    print("\n[9] Returning home")
    controller.move_to_joint_positions(home_q, iters=600, viewer=viewer)

    print("\n" + "=" * 60)
    print("  IK Pick & Place Complete!")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    model_path = "/home/quannh49/workspace/mujoco_robots/robot_arm/robot_arm.xml"

    print("Loading model …")
    controller = RobotArmController(model_path)
    print(f"  nv (total DOFs in model) : {controller.model.nv}")
    print(f"  Joints   : {controller.joint_names}")
    print(f"  EE site  : {controller.ee_site_id}")

    obj_pos = controller.get_target_position()
    print(f"  Target object pos (initial): {np.round(obj_pos, 4)}")
    print(f"  Target box half-z: {controller.target_half_z} m")

    print("\nLaunching MuJoCo viewer …  (Ctrl+C to stop)\n")
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:

        # Let physics settle
        for _ in range(300):
            mujoco.mj_step(controller.model, controller.data)
            viewer.sync()

        demo_ik_pick_and_place(controller, viewer)

        print("Demo done. Viewer open – Ctrl+C to exit.")
        try:
            while viewer.is_running():
                mujoco.mj_step(controller.model, controller.data)
                viewer.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nShutting down …")


if __name__ == "__main__":
    main()
