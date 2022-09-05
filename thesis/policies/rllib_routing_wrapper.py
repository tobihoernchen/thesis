from pyexpat import features
from turtle import forward
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import gym.spaces
import torch
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from torch import nn
from .routing_attention import RoutingFE


def register_attention_model():
    ModelCatalog.register_custom_model("attention_model", AttentionPolicy)


class CombinedFE(nn.Module):
    def __init__(self, obs_space, fleetsize, n_stations, embed_dim, n_heads, depth) -> None:
        super().__init__()
        
        self.agvfe = RoutingFE(obs_space["agvs"], fleetsize, embed_dim=embed_dim, n_heads=n_heads, depth=depth)
        self.stationfe = RoutingFE(obs_space["stations"], n_stations, embed_dim=embed_dim, n_heads=n_heads, depth=depth)
        combined_shape = (embed_dim * (fleetsize + n_stations),)
        self.combinedfe = RoutingFE(gym.spaces.Box(0, 1, shape = combined_shape), fleetsize + n_stations, embed_dim=embed_dim, n_heads=n_heads, depth=depth)


    def forward(self, x):
        agvfeatures = self.agvfe(x["agvs"]).flatten(1)
        stationfeatures = self.stationfe(x["stations"]).flatten(1)
        return self.combinedfe(torch.concat([agvfeatures, stationfeatures], axis = 1))


class AttentionPolicy(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        num_outputs: 5,
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
            self, obs_space, action_space, num_outputs, model_config, name
        )
        self.embed_dim = embed_dim
        self.with_action_mask = with_action_mask
        if with_action_mask:
            assert n_stations > 0
            self.fe = CombinedFE(obs_space.original_space, fleetsize, n_stations, embed_dim, n_heads, depth)
        else:
            self.fe = RoutingFE(
                obs_space, fleetsize, embed_dim=self.embed_dim, n_heads=n_heads, depth=depth
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
        # print(obsdict["obs"].shape) #(batch, obs_len)
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
        features_flat = torch.concat([features_flat, self.mask], axis = 1)
        return self.value_net(features_flat).flatten()


class AgentFlatten(nn.Module):
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
