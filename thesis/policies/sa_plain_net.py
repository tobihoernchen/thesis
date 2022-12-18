import gym.spaces
from torch import nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog



def register_plain_model():
    ModelCatalog.register_custom_model("plain_model", SAPlainPolicy)



class SAPlainPolicy(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_args
    ):
        nn.Module.__init__(self)
        self.discrete_action_space = isinstance(action_space, gym.spaces.Discrete)
        action_space_max = (
            action_space.nvec.max()
            if not self.discrete_action_space
            else None
        )
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        if len(custom_model_args) > 0:
            custom_config = custom_model_args
        else:
            custom_config = model_config["custom_model_config"]
        for key, default in dict(
            embed_dim=64,
            with_action_mask=True,
            activation=nn.ReLU,
        ).items():
            setattr(
                self,
                key,
                custom_config[key] if key in custom_config.keys() else default,
            )

        self.name = name

        self.fe = nn.Sequential(
            nn.Linear(obs_space.shape[0], self.embed_dim*4),
            self.activation(),
            nn.Linear(self.embed_dim*4, self.embed_dim*4),
            self.activation(),
            nn.Linear(self.embed_dim*4, self.embed_dim*2),
            self.activation(),
            nn.Linear(self.embed_dim*2, self.embed_dim),
        )

        self.action_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            self.activation(),
            nn.Linear(self.embed_dim, num_outputs),
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            self.activation(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, obsdict, state, seq_lengths):
        self.obs = obsdict["obs_flat"]
        self.features = self.fe(self.obs)

        actions = self.action_net(self.features)
        if self.discrete_action_space:
            action_out = actions
        else:
            action_out = actions.reshape(
                obsdict["obs"]["action_mask"].shape
            )
        if self.with_action_mask:
            action_out = action_out + (obsdict["obs"]["action_mask"] - 1) * 1e8
        return action_out.flatten(1), []

    def value_function(self):
        return self.value_net(self.features).flatten()
