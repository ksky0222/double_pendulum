from stable_baselines3 import SAC

from typing import Any, ClassVar, Optional, TypeVar, Union

from functorch import jacrev, vmap

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy


from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)


def mlp(sizes, hid_nonliear, out_nonliear):
    # declare layers
    layers = []
    for j in range(len(sizes) - 1):
        nonliear = hid_nonliear if j < len(sizes) - 2 else out_nonliear
        layers += [nn.Linear(sizes[j], sizes[j + 1]), nonliear()]
    # init weight
    for i in range(len(layers) - 1):
        if isinstance(layers[i], nn.Linear):
            if isinstance(layers[i+1], nn.ReLU):
                nn.init.kaiming_normal_(layers[i].weight, nonlinearity='relu')
            elif isinstance(layers[i+1], nn.LeakyReLU):
                nn.init.kaiming_normal_(layers[i].weight, nonlinearity='leaky_relu')
            else:
                nn.init.xavier_normal_(layers[i].weight)
    return nn.Sequential(*layers)


class K_net(nn.Module):
    def __init__(self, global_lips, k_init, sizes, hid_nonliear, out_nonliear) -> None:
        super().__init__()
        self.global_lips = global_lips
        if global_lips:
            # declare global Lipschitz constant
            self.k = th.nn.Parameter(th.tensor(k_init, dtype=th.float), requires_grad=True)
        else:
            # declare network
            self.k = mlp(sizes, hid_nonliear, out_nonliear)
            # set K_init
            self.k[-2].bias.data += th.tensor(k_init, dtype=th.float).data

    def forward(self, x):
        if self.global_lips:
            return F.softplus(self.k).repeat(x.shape[0]).unsqueeze(1)
        else:
            return self.k(x)
        
class LipsNet(nn.Module):
    def __init__(self, f_sizes, f_hid_nonliear=nn.ReLU, f_out_nonliear=nn.Identity,
                 global_lips=True, k_init=100, k_sizes=None, k_hid_act=nn.Tanh, k_out_act=nn.Identity,
                 loss_lambda=0.1, eps=1e-4, squash_action=True) -> None:
        super().__init__()
        # declare network
        self.f_net = mlp(f_sizes, f_hid_nonliear, f_out_nonliear)
        self.k_net = K_net(global_lips, k_init, k_sizes, k_hid_act, k_out_act)
        # declare hyperparameters
        self.loss_lambda = loss_lambda
        self.eps = eps
        self.squash_action = squash_action
        # initialize as eval mode
        self.eval()

    def forward(self, x):
        # K(x) forward
        k_out = self.k_net(x)
        # L2 regularization backward
        if self.training and k_out.requires_grad:
            lips_loss = self.loss_lambda * (k_out ** 2).mean()
            lips_loss.backward(retain_graph=True)
        # f(x) forward
        f_out = self.f_net(x)
        # calcute jac matrix
        if k_out.requires_grad:
            jacobi = vmap(jacrev(self.f_net))(x)
        else:
            with th.no_grad():
                jacobi = vmap(jacrev(self.f_net))(x)
        # jacobi.dim: (x.shape[0], f_out.shape[1], x.shape[1])
        #             (batch     , f output dim  , x feature dim)
        # calcute jac norm
        jac_norm = th.norm(jacobi, 2, dim=(1,2)).unsqueeze(1)
        # multi-dimensional gradient normalization (MGN)
        action = k_out * f_out / (jac_norm + self.eps)
        # squash action
        if self.squash_action:
            action = th.tanh(action)
        return action
    

class SAC_LipsActor(Actor):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        lips_arch: list[int],
        lips_kwargs: dict,
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )


        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]
        self.mu = LipsNet(f_sizes=[last_layer_dim,*lips_kwargs["lips_f_size"],action_dim], f_hid_nonliear=nn.ReLU, f_out_nonliear=nn.Identity,
                  global_lips=False, k_init=lips_kwargs["lips_k_init"], k_sizes=[last_layer_dim,*lips_kwargs["lips_k_size"],1], k_hid_act=nn.Tanh, k_out_act=nn.Softplus,
                  loss_lambda=lips_kwargs["lips_lam"], eps=lips_kwargs["lips_eps"], squash_action=True)
        

class SAC_LipsPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        lips_kwargs : Optional[dict[str, Any]] = dict(
                    {
                        "lips_lam" : 0.001,
                        "lips_eps" : 1e-4,
                        "lips_k_init" : [256],
                        "lips_f_size" : [256,256],
                        "lips_k_size" : 1
                    }
                )
    ):
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            net_arch = dict(pi=[256], qf=[256, 256], lips=dict(f=lips_kwargs["lips_f_size"], k=lips_kwargs["lips_k_size"]))

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        lips_arch = net_arch["lips"]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "lips_arch" : lips_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "lips_kwargs" : lips_kwargs
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)

        critic_base_kwargs = {
            k: v for k, v in self.net_args.items() if k not in ["lips_arch", "lips_kwargs"]
        }
        
        self.critic_kwargs = critic_base_kwargs
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)


    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return SAC_LipsActor(**actor_kwargs).to(self.device)

class SAC_Lips(SAC):
    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        lips_lam = 0.001,
        lips_eps = 1e-4,
        lips_k_init = 1.0,
        lips_f_size = [256,256],
        lips_k_size = [256],
    ):
        if policy_kwargs is None:
            policy_kwargs = dict()
        policy_kwargs.update(
            {
                "lips_kwargs" : dict(
                    {
                        "lips_lam" : lips_lam,
                        "lips_eps" : lips_eps,
                        "lips_k_init" : lips_k_init,
                        "lips_f_size" : lips_f_size,
                        "lips_k_size" : lips_k_size
                    }
                )
                
            }
        )
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            ent_coef = ent_coef,
            target_update_interval = target_update_interval,
            target_entropy = target_entropy,
        )
