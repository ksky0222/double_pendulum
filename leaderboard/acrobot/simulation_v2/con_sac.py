import numpy as np
from stable_baselines3 import SAC

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.wrap_angles import wrap_angles_top

from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)
from double_pendulum.utils.wrap_angles import wrap_angles_diff

from sim_parameters import (
    mpar,
    dt,
    t_final,
    t0,
    x0,
    goal,
    integrator,
    design,
    model,
    robot,
)

name = "sac"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "SAC",
    "short_description": "Swing-up with an RL Policy learned with SAC.",
    "readme_path": f"readmes/{name}.md",
    "username": "pendulum",
}

# class SACController(AbstractController):
#     def __init__(self, model_path, dynamics_func, dt):
#         super().__init__()

#         self.model = SAC.load(model_path)
#         self.dynamics_func = dynamics_func
#         self.dt = dt

#     def get_control_output_(self, x, t=None):
#         obs = self.dynamics_func.normalize_state(x)
#         action = self.model.predict(obs)
#         u = self.dynamics_func.unscale_action(action)
#         return u

class SACController(AbstractController):
    def __init__(self, model_path, dynamics_func, dt,scaling = True):
        super().__init__()

        self.model = SAC.load(model_path)
        self.dynamics_func = dynamics_func
        self.dt = dt
        self.model.predict([0, 0, 0, 0])
        self.scaling = scaling

    def get_control_output_(self, x, t=None):
        if self.scaling:
            obs = self.dynamics_func.normalize_state(x)
            action = self.model.predict(obs)
            u = self.dynamics_func.unscale_action(action)
        else:
            action = self.model.predict(x)
            u = self.dynamics_func.unscale_action(action)
        return u


torque_limit = [0.0, 5.0]

######################################################

dynamics_func = double_pendulum_dynamics_func(
    simulator=None,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
    max_velocity=20.0,
    torque_limit=torque_limit,
)

# controller = SACController(
#     model_path="../../../data/policies/design_C.1/model_1.0/acrobot/SAC/sac_model.zip",
#     dynamics_func=dynamics_func,
#     dt=dt,
# )
model_path = "../../../data/policies/design_C.1/model_1.0/acrobot/SAC/sac_model.zip"
controller = SACController(
    model_path = model_path,
    dynamics_func=dynamics_func,
    dt=dt,
)

controller.init()
# controller2 = LQRController(model_pars=mpar)
# controller2.set_goal(goal)
# controller2.set_cost_matrices(Q=Q, R=R)
# controller2.set_parameters(failure_value=0.0, cost_to_go_cut=15)

# controller = CombinedController(
#     controller1=controller1,
#     controller2=controller2,
#     condition1=condition1,
#     condition2=condition2,
#     compute_both=False,
# )
