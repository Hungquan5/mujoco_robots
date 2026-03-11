import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.linalg import solve_continuous_are
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

def f(model,data, input_x,input_u):
    """input: x = [
cart_pos,
pole1_angle,
pole2_angle,
pole3_angle,
cart_vel,
pole1_vel,
pole2_vel,
pole3_vel
]

output = [cart_vel,
 pole1_vel,
 pole2_vel,
 pole3_vel,
 cart_acc,
 pole1_acc,
 pole2_acc,
 pole3_acc]
"""
    data.qpos[:4] = input_x[:4]
    data.qvel[:4] = input_x[4:]
    data.ctrl[0] = input_u[0]
    mujoco.mj_forward(model, data)

    xdot = np.zeros(8)
    xdot[:4] = data.qvel[:4]
    xdot[4:] = data.qacc[:4]

    return xdot

def init_controller(model,data):    
# calculate A and B matrices for linearized dynamics at the upright equilibrium
# A= ∂f/∂x, B=∂f/∂u
    input_x = np.zeros(8)
    input_u = np.zeros(1)

    A = np.zeros((8,8))
    B = np.zeros((8,1))

    mujoco.mj_forward(model, data)
    expectation_state = np.concatenate([data.qpos[:4], data.qvel[:4]])

    eps = 0.000001
    for i in range(8):
        input_x = expectation_state.copy()
        input_x[i] += eps
        xdot_plus = f(model,data, input_x,input_u)

        input_x = expectation_state.copy()
        input_x[i] -= eps
        xdot_minus = f(model,data, input_x,input_u)

        A[:, i] = (xdot_plus - xdot_minus) / (2 * eps)
    for i in range(1):
        input_u = np.zeros(1)
        input_u[i] += eps
        xdot_plus = f(model,data, expectation_state,input_u)

        input_u = np.zeros(1)
        input_u[i] -= eps
        xdot_minus = f(model,data, expectation_state,input_u)

        B[:, i] = (xdot_plus - xdot_minus) / (2 * eps)
    
    Q = np.diag([
1,     # cart pos
200,   # pole1
400,   # pole2
600,   # pole3
1,
10,
10,
10
])
    R = np.diag([0.1])  # control cost
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K
def my_controller(model,data, K):
    x = np.concatenate([data.qpos[:4], data.qvel[:4]])
    u = -K @ x
    data.ctrl[0] = float(u)

    #add force noise
    noise = 0.1 * np.random.randn()
    
    data.xfrc_applied[5] += noise
    noise = 0.1 * np.random.randn()

    data.xfrc_applied[6] += noise
    noise = 0.1 * np.random.randn()

    data.xfrc_applied[7] += noise
def main():
    model = mujoco.MjModel.from_xml_path("pendulum_balance_cart.xml")
    data = mujoco.MjData(model)
    # Use model.opt.timestep — fixed and accurate, not wall-clock time
    dt = model.opt.timestep
    K = init_controller(model,data)
    flag=1
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            my_controller(model,data, K)
            mujoco.mj_step(model, data)
            viewer.sync()  # passive viewer uses sync(), not render()

if __name__ == "__main__":
    main()