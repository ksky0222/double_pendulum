import numpy as np

from double_pendulum.model.model_parameters import model_parameters

design = "design_C.1"
model = "model_1.0"
robot = "acrobot"

model_par_path = (
    "../../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit([0.0, 6.0])
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])


integrator = "runge_kutta"
dt = 0.002
t0 = 0.0
t_final = 10.0
# x0 = [0.0, 0.0, 0.0, 0.0]
noise = np.random.randn(4)
x0 = [-1.0, -1.0, 0.0, 0.0] + np.random.randn(4) * 0.01
goal = [np.pi, 0.0, 0.0, 0.0]