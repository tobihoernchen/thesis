import gym.spaces
from torch import nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog


def register_attention_model():
    ModelCatalog.register_custom_model("attention_model", AttentionPolicy)


class Embedder(nn.Module):
    def __init__(self, embed_dim, original_dim, activation) -> None:
        super().__init__()
        self.original_dim = original_dim
        self.embedd = nn.Sequential(
            nn.Linear(original_dim, embed_dim * 4),
            activation(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
            activation(),
            nn.Linear(embed_dim * 4, embed_dim),
            activation(),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(x.shape[0], x.shape[1] // self.original_dim, self.original_dim)
        return self.embedd(x)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_agents, activation) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm((n_agents, embed_dim))
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            activation(),
            nn.Linear(4 * embed_dim, embed_dim),
            activation(),
        )
        self.norm2 = nn.LayerNorm((n_agents, embed_dim))

    def forward(self, q, k, v):
        attended = self.attention(q, k, v, need_weights=False)[0]
        x = v + attended
        normed1 = self.norm1(x)
        fedforward = self.feedforward(normed1)
        x = fedforward + x
        normed2 = self.norm2(x)
        return normed2


class AttentionPolicy(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Discrete,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        fleetsize: int = 10,
        n_stations: int = 0,
        embed_dim=64,
        n_heads=8,
        depth=4,
        with_action_mask=True,
        activation=nn.ReLU,
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, action_space.n, model_config, name
        )
        self.with_action_mask = with_action_mask
        self.n_features = self.obs_space.original_space["agvs"].shape[0] // fleetsize

        self.embedd_main = Embedder(embed_dim, self.n_features, activation)
        self.embedd_agvs = Embedder(embed_dim, self.n_features * 2, activation)
        self.embedd_station = Embedder(embed_dim, self.n_features, activation)

        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    embed_dim, n_heads, fleetsize + n_stations - 1, activation
                )
                for _ in range(depth)
            ]
        )

        self.action_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            activation(),
            nn.Linear(embed_dim * 4, embed_dim),
            activation(),
            nn.Linear(embed_dim, action_space.n),
        )

        self.value_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            activation(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, obsdict, state, seq_lengths):
        main_data = obsdict["obs"]["agvs"][:, : self.n_features]
        agvs_data = obsdict["obs"]["agvs"][:, self.n_features :]
        station_data = obsdict["obs"]["stations"]

        main_embedded = self.embedd_main(main_data)
        n_other_agvs = agvs_data.shape[1] // self.n_features
        main_repeated = main_data[:, None, :].repeat(1, n_other_agvs, 1)
        agvs_reshaped = agvs_data.view(
            agvs_data.shape[0], n_other_agvs, self.n_features
        )
        agvs_embedded = self.embedd_agvs(
            torch.concat([main_repeated, agvs_reshaped], 2).flatten(1)
        )
        stations_embedded = self.embedd_station(station_data)

        queries = torch.concat([agvs_embedded, stations_embedded], 1)
        values = main_embedded.repeat(1, queries.shape[1], 1)
        for block in self.attention_blocks:
            values = block(queries, values, values)

        self.features = values.max(dim=1)[0]

        actions = self.action_net(self.features)

        if self.with_action_mask:
            actions = actions + (obsdict["obs"]["action_mask"] - 1) * 1e8

        return actions, []

    def value_function(self):
        return self.value_net(self.features).flatten()
