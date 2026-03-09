#!/usr/bin/env python3
"""
6-DOF Industrial Cobot Arm – Pick & Place Demo
Joints: J1(base-Z)  J2(shoulder-Y)  J3(elbow-Y)
        J4(wrist-Z) J5(wrist-Y)     J6(tool-Z)
End-effector: industrial parallel-jaw gripper (coupled prismatic)
"""

import mujoco
import mujoco.viewer
import numpy as np
import time


class RobotArmController:
    def __init__(self, model_path):
        """Initialize the 6-DOF robot arm controller."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 6 revolute arm joints + 1 gripper actuator (coupled fingers)
        self.joint_names = [
            'joint1', 'joint2', 'joint3',
            'joint4', 'joint5', 'joint6',
            'left_finger_joint',
        ]
        self.actuator_names = [
            'm_joint1', 'm_joint2', 'm_joint3',
            'm_joint4', 'm_joint5', 'm_joint6',
            'm_gripper',
        ]

        # Resolve IDs
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.joint_names
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.actuator_names
        ]

        # Tool Center Point
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')

    # ── helpers ──────────────────────────────────────────────────
    def get_joint_positions(self):
        """Return current ctrl targets [j1..j6, gripper]."""
        return self.data.ctrl[self.actuator_ids].copy()

    def set_joint_targets(self, targets):
        """Write ctrl targets for all actuators."""
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = targets[i]

    def get_ee_position(self):
        """TCP position in world frame."""
        return self.data.site_xpos[self.ee_site_id].copy()

    # ── motion ──────────────────────────────────────────────────
    def move_to_joint_positions(self, target_positions, duration=2.0, viewer=None):
        """Smooth cosine-interpolated move to *target_positions*
        [j1, j2, j3, j4, j5, j6, gripper]."""
        start_positions = self.get_joint_positions()
        start_time = self.data.time

        print(f"  ctrl → {np.round(target_positions, 3)}")

        while self.data.time - start_time < duration:
            alpha = (self.data.time - start_time) / duration
            smooth = (1 - np.cos(alpha * np.pi)) / 2
            self.set_joint_targets(
                start_positions + smooth * (target_positions - start_positions))
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

        print(f"  EE pos: {np.round(self.get_ee_position(), 4)}")

    def open_gripper(self, duration=0.5, viewer=None):
        """Open parallel-jaw gripper (slide to 0)."""
        current = self.get_joint_positions()
        target = current.copy()
        target[6] = 0.04          # fully open (40 mm each side)
        print("Opening gripper …")
        self.move_to_joint_positions(target, duration, viewer)

    def close_gripper(self, duration=0.5, viewer=None):
        """Close parallel-jaw gripper (slide to 0)."""
        current = self.get_joint_positions()
        target = current.copy()
        target[6] = 0.0           # fully closed
        print("Closing gripper …")
        self.move_to_joint_positions(target, duration, viewer)


def add_target_object(model, data):
    """
    Add a target object to grip in the scene.
    Note: This modifies the model to include a small box to grip.
    """
    # Get object body ID if it exists
    try:
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target_object')
        print(f"Target object found at position: {data.xpos[obj_id]}")
        return data.xpos[obj_id].copy()
    except:
        print("Warning: 'target_object' not found in model. Using default position.")
        # Return a reasonable default position in front of the robot
        return np.array([0.3, 0.0, 0.3])


def demo_pick_and_place(controller, viewer=None):
    """
    Demonstrate a pick-and-place sequence with the 6-DOF cobot.

    Pose vector: [j1, j2, j3, j4, j5, j6, gripper]
      j1  = base rotation (Z)      j4  = wrist roll  (Z)
      j2  = shoulder lift (Y)      j5  = wrist pitch (Y)
      j3  = elbow bend   (Y)      j6  = tool rotation(Z)
      gripper: 0.0 = closed, 0.04 = fully open (40 mm)
    """
    print("\n" + "=" * 55)
    print("  6-DOF Industrial Cobot – Pick & Place Demo")
    print("=" * 55 + "\n")

    # ── Key poses ───────────────────────────────────────────────
    #                     j1    j2     j3     j4   j5     j6   grip
    home      = np.array([ 0.0, -0.8,  -0.4,  0.0, -0.6,  0.0, 0.04])
    pre_grasp = np.array([ 0.35, -1.2,  -1.0,  0.0, -0.8,  0.0, 0.04])
    grasp     = np.array([ 0.35, -1.4,  -1.2,  0.0, -0.6,  0.0, 0.04])
    grip      = np.array([ 0.35, -1.4,  -1.2,  0.0, -0.6,  0.0, 0.00])
    lift      = np.array([ 0.35, -0.9,  -0.7,  0.0, -0.6,  0.0, 0.00])
    place     = np.array([-0.5,  -0.9,  -0.7,  0.3, -0.6,  0.0, 0.00])
    release   = np.array([-0.5,  -0.9,  -0.7,  0.3, -0.6,  0.0, 0.04])

    time.sleep(0.5)

    steps = [
        ("1. Home position",        home,      2.0),
        ("2. Open gripper",         home,      0.8),
        ("3. Pre-grasp approach",   pre_grasp, 2.5),
        ("4. Descend to object",    grasp,     1.5),
        ("5. Close gripper",        grip,      1.0),
        ("6. Lift object",          lift,      2.0),
        ("7. Move to place",        place,     3.0),
        ("8. Release object",       release,   1.0),
        ("9. Return home",          home,      2.5),
    ]

    for label, pose, dur in steps:
        print(f"\n{label}")
        controller.move_to_joint_positions(pose, duration=dur, viewer=viewer)
        time.sleep(0.3)

    print("\n" + "=" * 55)
    print("  Pick & Place Demo Complete!")
    print("=" * 55 + "\n")


def main():
    """Main function to run the 6-DOF cobot arm demo."""
    model_path = "/home/quannh/working/intern_vin/robot_arm/my_robot.xml"

    print("Initializing 6-DOF industrial cobot controller …")
    controller = RobotArmController(model_path)

    print("Joints     :", controller.joint_names)
    print("Actuators  :", controller.actuator_names)
    print(f"EE site ID : {controller.ee_site_id}")

    # Check for target object
    target_pos = add_target_object(controller.model, controller.data)
    print(f"Target pos : {target_pos}")

    # Launch viewer and run demo
    print("\nLaunching MuJoCo viewer …")
    print("Press Ctrl+C to stop the simulation.\n")

    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        # Stabilise
        for _ in range(200):
            mujoco.mj_step(controller.model, controller.data)
            viewer.sync()

        # Run demo
        demo_pick_and_place(controller, viewer)

        # Keep viewer alive
        print("Demo complete. Viewer remains open – Ctrl+C to exit.")
        try:
            while viewer.is_running():
                mujoco.mj_step(controller.model, controller.data)
                viewer.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nShutting down …")


if __name__ == "__main__":
    main()
