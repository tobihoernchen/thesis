import gym.spaces
from torch import nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from .custom_blocks import (
    TransformerEncoderBlock,
)


def register_sa_attn_model():
    ModelCatalog.register_custom_model("sa_attn_model", SAAttnPolicy)


class AttnFE(nn.Module):
    def __init__(
        self,
        obs_space,
        embed_dim=64,
        position_embedd_dim=2,
        n_heads=4,
        depth=4,
        activation=nn.ReLU,
    ) -> None:

        super().__init__()
        n_features = obs_space["agvs"].shape[1]
        n_agvs = obs_space["agvs"].shape[0]


        self.embedd_agvs = self.get_embedder(
            n_features, embed_dim, activation
        )
        self.embedd_station = self.get_embedder(
            n_features, embed_dim, activation
        )

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, n_heads, n_agvs, activation)
                for _ in range(depth)
            ]
        )


    def get_embedder(self, n_features, embed_dim, activation):
        return nn.Sequential(
            nn.Linear(n_features, embed_dim * 2),
            activation(),
            nn.Linear(embed_dim * 2, embed_dim),
            activation(),
        )

    def forward(self, obs):
        agvs_embedded = self.embedd_agvs(obs["agvs"])
        for block in self.encoder_blocks:
            agvs_embedded = block(agvs_embedded)
        return agvs_embedded


class SAAttnPolicy(TorchModelV2, nn.Module):
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
            else action_space.n
        )
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        if len(custom_model_args) > 0:
            custom_config = custom_model_args
        else:
            custom_config = model_config["custom_model_config"]
        for key, default in dict(
            single_mode=False,
            embed_dim=64,
            with_action_mask=True,
            with_stations=True,
            activation=nn.ReLU,
        ).items():
            setattr(
                self,
                key,
                custom_config[key] if key in custom_config.keys() else default,
            )

        n_agvs = obs_space.original_space["agvs"].shape[0]
        self.name = name

        self.fe = AttnFE(
            obs_space=obs_space.original_space,
            embed_dim=self.embed_dim,
        )

        self.action_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            self.activation(),
            nn.Linear(self.embed_dim, action_space_max),
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.embed_dim * n_agvs, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            self.activation(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, obsdict, state, seq_lengths):
        self.obs = obsdict["obs"]
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
        return self.value_net(self.features.flatten(1)).flatten()
