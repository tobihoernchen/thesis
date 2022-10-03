from turtle import forward
import gym.spaces
from torch import nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog


def register_lin_model():
    ModelCatalog.register_custom_model("lin_model", LinPolicy)


class PositionEncoder(nn.Module):
    def __init__(
        self, pos_cols, original_dim, embed_size, resolution=100, mask_valid=None
    ) -> None:
        super().__init__()
        self.original_dim = original_dim
        self.embedding = nn.Embedding(resolution, embed_size)
        self.pos_cols = pos_cols
        self.resolution = resolution
        self.embed_size = embed_size

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 2:
            x = x.view(x.shape[0], x.shape[1] // self.original_dim, self.original_dim)
        blocks = []
        last_col = None
        for col in self.pos_cols:
            if last_col is None:
                blocks.append(x[:, :, :col])
            last_col = col
            to_encode = x[:, :, col : col + 2]
            to_encode = (to_encode * (self.resolution - 1)).int()
            encoded = self.embedding(to_encode).flatten(2)
            blocks.append(encoded)
        blocks.append(x[:, :, col + 2 :])
        out = torch.concat(blocks, 2)
        return out

    def out_features(self):
        return self.original_dim + len(self.pos_cols) * (self.embed_size - 1) * 2


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


class LinPolicy(TorchModelV2, nn.Module):
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
        with_agvs=True,
        with_stations=True,
        activation=nn.ReLU,
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, action_space.n, model_config, name
        )
        self.with_action_mask = with_action_mask
        self.n_features = self.obs_space.original_space["agvs"].shape[0] // fleetsize
        self.fleetsize = fleetsize
        self.n_stations = n_stations

        self.embedd_main = Embedder(embed_dim, self.n_features, activation)

        self.main_ff = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            activation(),
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            activation(),
            nn.Linear(2 * embed_dim, embed_dim),
            activation(),
        )

        self.with_agvs = with_agvs
        if with_agvs:
            self.embedd_agvs = Embedder(embed_dim *2, self.n_features * 2, activation)
            self.lintention_agvs = nn.Sequential(
                nn.Linear(2 * embed_dim, 4 * embed_dim),
                activation(),
                nn.Linear(4 * embed_dim, 2 * embed_dim),
                activation(),
                nn.Linear(2 * embed_dim, embed_dim),
                activation(),
            )

        self.with_stations = with_stations
        if with_stations:
            self.embedd_station = Embedder(embed_dim * 2, self.n_features * 2, activation)
            self.lintention_station = nn.Sequential(
                nn.Linear(2 * embed_dim, 4 * embed_dim),
                activation(),
                nn.Linear(4 * embed_dim, 2 * embed_dim),
                activation(),
                nn.Linear(2 * embed_dim, embed_dim),
                activation(),
            )

        n_concats = 1 + with_agvs + with_stations

        self.action_net = nn.Sequential(
            nn.Linear(embed_dim * n_concats, embed_dim * 4),
            activation(),
            nn.Linear(embed_dim * 4, embed_dim),
            activation(),
            nn.Linear(embed_dim, action_space.n),
        )

        self.value_net = nn.Sequential(
            nn.Linear(embed_dim * n_concats, embed_dim * 4),
            activation(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
            activation(),
            nn.Linear(embed_dim * 4, embed_dim),
            activation(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, obsdict, state, seq_lengths):
        features = []
        main_data = obsdict["obs"]["agvs"][:, : self.n_features]
        main_embedded = self.embedd_main(main_data)
        features_main = self.main_ff(main_embedded).squeeze(dim = 1)
        features.append(features_main)

        if self.with_agvs:
            agvs_data = obsdict["obs"]["agvs"][:, self.n_features :]
            n_agvs = agvs_data.shape[1] // self.n_features
            agvs_reshaped = agvs_data.reshape((agvs_data.shape[0], n_agvs, self.n_features))
            main_repeated = main_data[:, None, :].repeat(1, n_agvs, 1)
            agvs_embedded = self.embedd_agvs(
                torch.concat([main_repeated, agvs_reshaped], 2)
            )
            lintentioned_agvs = self.lintention_agvs(agvs_embedded)
            features_agvs = lintentioned_agvs.max(dim=1)[0]
            features.append(features_agvs)

        if self.with_stations:
            station_data = obsdict["obs"]["stations"]
            n_stations = station_data.shape[1] // self.n_features
            stations_reshaped = station_data.reshape((station_data.shape[0], n_stations, self.n_features))
            main_repeated = main_data[:, None, :].repeat(1, n_stations, 1)
            stations_embedded = self.embedd_station(
                torch.concat([main_repeated, stations_reshaped], 2)
            )
            lintentioned_station = self.lintention_station(stations_embedded)
            features_stations = lintentioned_station.max(dim=1)[0]
            features.append(features_stations)

        self.features = torch.concat(features, 1)

        actions = self.action_net(self.features)
        if self.with_action_mask:
            actions = actions + (obsdict["obs"]["action_mask"] - 1) * 1e8
        return actions, []

    def value_function(self):
        return self.value_net(self.features).flatten()
