import mujoco
import mujoco.viewer
import numpy as np
import time

class PID:
    def __init__(self, kp, ki, kd, setpoint=0.0, ilimit=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.ilimit = ilimit

        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        self.integral = max(-self.ilimit, min(self.ilimit, self.integral))
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0   
def main():
    model = mujoco.MjModel.from_xml_path("pendulum_balance_cart.xml")
    data = mujoco.MjData(model)

    # Use model.opt.timestep — fixed and accurate, not wall-clock time
    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # setpoint=0.0 for all angles: upright position is 0 rad in MuJoCo
        # (pole starts vertical in XML with euler="0 0 0")
        pid_cart  = PID(kp=100.0, ki=0.0, kd=20.0,  setpoint=0.0)
        pid_pole1 = PID(kp=200.0, ki=0.0, kd=30.0,  setpoint=0.0)
        pid_pole2 = PID(kp=100.0, ki=0.0, kd=20.0,  setpoint=0.0)
        pid_pole3 = PID(kp=50.0,  ki=0.0, kd=10.0,  setpoint=0.0)

        while viewer.is_running():
            # Read state from sensordata (order matches XML sensor block)
            cart_pos   = data.sensordata[0]  # jointpos: cart_slide
            # cart_vel = data.sensordata[1]  # jointvel: cart_slide
            pole1_angle = data.sensordata[2] # jointpos: pole_hinge
            # pole1_vel = data.sensordata[3] # jointvel: pole_hinge
            pole2_angle = data.sensordata[4] # jointpos: pole_pole_hinge
            # pole2_vel = data.sensordata[5] # jointvel: pole_pole_hinge
            pole3_angle = data.sensordata[6] # jointpos: pole_pole_pole_hinge
            # pole3_vel = data.sensordata[7] # jointvel: pole_pole_pole_hinge

            # Compute control: weighted sum of all PIDs
            # Higher poles get smaller weights (less leverage on the cart)
            force = (
                pid_pole1.compute(pole1_angle, dt) * 1.00 +
                pid_pole2.compute(pole2_angle, dt) * 0.50 +
                pid_pole3.compute(pole3_angle, dt) * 0.25 +
                pid_cart.compute(cart_pos, dt)     * 0.30
            )

            # Clip to motor ctrlrange [-100, 100]
            data.ctrl[0] = np.clip(force, -100.0, 100.0)

            mujoco.mj_step(model, data)
            viewer.sync()  # passive viewer uses sync(), not render()

if __name__ == "__main__":
    main()