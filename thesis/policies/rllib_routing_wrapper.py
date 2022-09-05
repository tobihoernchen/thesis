from pyexpat import features
from turtle import forward
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import gym.spaces
import torch
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from torch import nn
from .ma_attention import MultiAgentFE


def register_attention_model():
    ModelCatalog.register_custom_model("attention_model", AttentionPolicy)


class CombinedFE(nn.Module):
    """
    Handle stations and agvs seperately for [depth] attention blocks and the combined for [depth] attention blocks.
    """

    def __init__(
        self, obs_space, n_agents, n_stations, embed_dim, n_heads, depth
    ) -> None:
        super().__init__()

        self.agvfe = MultiAgentFE(
            obs_space["agvs"],
            n_agents,
            embed_dim=embed_dim,
            n_heads=n_heads,
            depth=depth,
            use_attention_mask=True,
        )
        self.stationfe = MultiAgentFE(
            obs_space["stations"],
            n_stations,
            embed_dim=embed_dim,
            n_heads=n_heads,
            depth=depth,
            use_attention_mask=False,
        )
        combined_shape = (embed_dim * (n_agents + n_stations),)
        self.combinedfe = MultiAgentFE(
            gym.spaces.Box(0, 1, shape=combined_shape),
            n_agents + n_stations,
            embed_dim=embed_dim,
            n_heads=n_heads,
            depth=depth,
            use_attention_mask=False,
        )

    def forward(self, x):
        agvfeatures = self.agvfe(x["agvs"]).flatten(1)
        stationfeatures = self.stationfe(x["stations"]).flatten(1)
        return self.combinedfe(torch.concat([agvfeatures, stationfeatures], axis=1))


class AttentionPolicy(TorchModelV2, nn.Module):
    """
    Custom RLLIB model for multiagent self-attention with agvs (and stations)
    obs_space: Should be Dict if with_action_mask and Box otherwise
    action_space: action space
    fleetsize: number of AGVs
    n_stations: number of stations
    with_action_mask: If the environment provides action masks

    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        model_config: ModelConfigDict,
        name: str,
        fleetsize: int = 10,
        n_stations: int = 0,
        embed_dim=32,
        n_heads=8,
        depth=4,
        with_action_mask=True,
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, action_space.n, model_config, name
        )
        self.embed_dim = embed_dim
        self.with_action_mask = with_action_mask
        if isinstance(obs_space.original_space, gym.spaces.Dict):
            assert n_stations > 0
            self.fe = CombinedFE(
                obs_space.original_space,
                fleetsize,
                n_stations,
                embed_dim,
                n_heads,
                depth,
            )
        else:
            self.fe = MultiAgentFE(
                obs_space,
                fleetsize,
                embed_dim=self.embed_dim,
                n_heads=n_heads,
                depth=depth,
            )
        self.action_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, action_space.n),
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.embed_dim + 5, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, obsdict, state, seq_lens):
        if self.with_action_mask:
            self.mask = obsdict["obs"]["action_mask"]
        obs = obsdict["obs"]
        self.features = self.fe(obs)
        flat = self.features.max(dim=1)[0]
        actions = self.action_net(flat)
        if self.with_action_mask:
            actions = torch.masked_fill(actions, self.mask == 0, 1e-9)
        state = []

        return actions, state

    def value_function(self):
        features_flat = self.features.max(dim=1)[0]
        features_flat = torch.concat([features_flat, self.mask], axis=1)
        return self.value_net(features_flat).flatten()


class AgentFlatten(nn.Module):
    """
        Can be used instead of max-reduction, but parameters depend on the fleetsize and therefore this part can not be reused for bigger fleets
    """
    def __init__(self, n_agents):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(n_agents, n_agents * 4),
            nn.ReLU(),
            nn.Linear(n_agents * 4, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        x = self.lin(x)
        x = x.transpose(1, 2)
        return x.flatten(1)
