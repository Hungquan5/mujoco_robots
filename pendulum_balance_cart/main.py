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

def compute_total_energy_swingup(model, data, theta, theta_dot):
    # Parameters
    m_pole1 = 0.1
    m_pole2 = 0.1
    m_pole3 = 0.1
    m_cart = 1.0
    lc = 0.2   # CoM distance from each hinge (= half of pole cylinder length)
    L  = 0.4   # Full pole length (distance between consecutive hinges)
    g = 9.81

    # MuJoCo joint angles are RELATIVE to the parent body frame.
    # Convert to absolute angles from vertical.
    th1 = data.qpos[1]
    th2 = data.qpos[1] + data.qpos[2]
    th3 = data.qpos[1] + data.qpos[2] + data.qpos[3]

    # Absolute angular velocities
    dth1 = data.qvel[1]
    dth2 = data.qvel[1] + data.qvel[2]
    dth3 = data.qvel[1] + data.qvel[2] + data.qvel[3]
    v_cart = data.qvel[0]

    # Potential energy (heights of CoMs relative to pole1 hinge level)
    PE = (m_pole1 * g * lc * np.cos(th1) +
          m_pole2 * g * (L * np.cos(th1) + lc * np.cos(th2)) +
          m_pole3 * g * (L * np.cos(th1) + L * np.cos(th2) + lc * np.cos(th3)))

    # Kinetic energy — Cartesian velocity of each CoM
    # (pole rotates around y-axis: x = pivot_x + r*sin(th), z = pivot_z + r*cos(th))
    vx1 = v_cart + lc * dth1 * np.cos(th1)
    vz1 =        - lc * dth1 * np.sin(th1)
    KE_pole1 = 0.5 * m_pole1 * (vx1**2 + vz1**2)

    vx2 = v_cart + L * dth1 * np.cos(th1) + lc * dth2 * np.cos(th2)
    vz2 =        - L * dth1 * np.sin(th1) - lc * dth2 * np.sin(th2)
    KE_pole2 = 0.5 * m_pole2 * (vx2**2 + vz2**2)

    vx3 = v_cart + L * dth1 * np.cos(th1) + L * dth2 * np.cos(th2) + lc * dth3 * np.cos(th3)
    vz3 =        - L * dth1 * np.sin(th1) - L * dth2 * np.sin(th2) - lc * dth3 * np.sin(th3)
    KE_pole3 = 0.5 * m_pole3 * (vx3**2 + vz3**2)

    KE_cart = 0.5 * m_cart * v_cart**2
    KE = KE_cart + KE_pole1 + KE_pole2 + KE_pole3

    return PE + KE

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

mode = "swingup"
step_count = 0

def my_controller(model,data, K):
    global mode, step_count
    step_count += 1
    # PE at upright equilibrium (all thetas=0, zero velocity):
    # m1*g*lc + m2*g*(L+lc) + m3*g*(L+L+lc)
    E_desired = 0.1 * 9.81 * (0.2 + 0.6 + 1.0)  # ≈ 1.765 J
    if (mode=="swingup"):
        E = compute_total_energy_swingup(model, data, data.qpos[1:4], data.qvel[1:4])
        energy_error = E - E_desired
        k_energy = 15.0
        # Correct Lyapunov-based energy pumping: u = k*(E-Eref)*dth1*cos(th1)
        # At theta1=pi (hanging down): cos(pi)=-1, so sign flips vs sign(qvel[1])
        pump_dir = np.sign(data.qvel[1] * np.cos(data.qpos[1]))
        if abs(data.qvel[1]) < 1e-4:  # zero velocity fallback
            pump_dir = -np.sign(np.cos(data.qpos[1]))  # direction that injects energy
        u = np.clip(k_energy * energy_error * pump_dir, -50.0, 50.0)
        data.ctrl[0] = float(u)
        if step_count % 1000 == 0:
            print(f"[swingup] t={data.time:.2f}s  E={E:.3f}  E_des={E_desired:.3f}  err={energy_error:.3f}  u={u:.2f}  th1={data.qpos[1]:.2f}  qvel1={data.qvel[1]:.3f}")
        if energy_error > -0.3:
            mode = "balance"
            print(">>> switching to BALANCE mode")
    else:
        x = np.concatenate([data.qpos[:4], data.qvel[:4]])
        u = -K @ x
        data.ctrl[0] = float(u)

        # apply random force noise — use = not += to avoid accumulation
        data.xfrc_applied[5][0] = 0.1 * np.random.randn()
        data.xfrc_applied[6][0] = 0.1 * np.random.randn()
        data.xfrc_applied[7][0] = 0.1 * np.random.randn()
        if step_count % 1000 == 0:
            print(f"[balance] t={data.time:.2f}s  qpos={data.qpos[:4]}  u={float(u):.2f}")
        if (abs(data.qpos[1]) < 0.1 and abs(data.qpos[2]) < 0.1 and abs(data.qpos[3]) < 0.1):
            mode = "swingup"  # switch back to swing-up mode if poles deviate too much

def main():
    global mode, step_count
    step_count = 0
    model = mujoco.MjModel.from_xml_path("pendulum_balance_cart.xml")
    data = mujoco.MjData(model)
    # Use model.opt.timestep — fixed and accurate, not wall-clock time
    dt = model.opt.timestep
    K = init_controller(model,data)
    # Start with all poles pointing down
    data.qpos[1] = 3.14
    data.qvel[1] = 0.1  # small initial perturbation to break zero-velocity deadlock
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            my_controller(model,data, K)
            mujoco.mj_step(model, data)
            viewer.sync()  # passive viewer uses sync(), not render()

if __name__ == "__main__":
    main()