import gym.spaces
from torch import nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from .custom_blocks import PositionEncoder, Embedder


def register_lin_model():
    ModelCatalog.register_custom_model("lin_model", LinPolicy)


class LinFE(nn.Module):
    def __init__(
        self,
        action_space_max: int,
        n_features,
        embed_dim=64,
        with_agvs=True,
        with_stations=True,
        activation=nn.ReLU,
    ) -> None:

        super().__init__()
        self.n_features = n_features
        self.encode_main = PositionEncoder(list(range(2, 15, 2)), self.n_features, 2)
        self.embedd_main = Embedder(
            embed_dim, self.encode_main.out_features(), activation
        )

        self.main_ff = nn.Sequential(
            nn.Linear(embed_dim + action_space_max, 2 * embed_dim),
            activation(),
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            activation(),
            nn.Linear(2 * embed_dim, embed_dim),
            activation(),
        )

        self.with_agvs = with_agvs
        if with_agvs:
            self.encode_agvs = PositionEncoder(
                list(range(2, 15, 2)), self.n_features, 2
            )
            self.embedd_agvs = Embedder(
                embed_dim * 2,
                self.encode_agvs.out_features() + self.encode_main.out_features(),
                activation,
            )
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
            self.encode_station = PositionEncoder(
                [
                    1,
                ],
                self.n_features,
                2,
            )
            self.embedd_station = Embedder(
                embed_dim * 2,
                self.encode_station.out_features() + self.encode_main.out_features(),
                activation,
            )
            self.lintention_station = nn.Sequential(
                nn.Linear(2 * embed_dim, 4 * embed_dim),
                activation(),
                nn.Linear(4 * embed_dim, 2 * embed_dim),
                activation(),
                nn.Linear(2 * embed_dim, embed_dim),
                activation(),
            )

    def forward(self, obs):
        features = []
        main_data = self.encode_main(obs["agvs"][:, None, : self.n_features])
        main_embedded = torch.concat(
            [self.embedd_main(main_data), obs["last_action"][:, None, :]], 2
        )
        features_main = self.main_ff(main_embedded).squeeze(dim=1)
        features.append(features_main)

        if self.with_agvs:
            agvs_data = obs["agvs"][:, self.n_features :]
            n_agvs = agvs_data.shape[1] // self.n_features
            agvs_reshaped = self.encode_agvs(
                agvs_data.reshape((agvs_data.shape[0], n_agvs, self.n_features))
            )
            main_repeated = main_data.repeat(1, n_agvs, 1)
            agvs_embedded = self.embedd_agvs(
                torch.concat([main_repeated, agvs_reshaped], 2)
            )
            lintentioned_agvs = self.lintention_agvs(agvs_embedded)
            features_agvs = lintentioned_agvs.max(dim=1)[0]
            features.append(features_agvs)

        if self.with_stations:
            station_data = obs["stations"]
            n_stations = station_data.shape[1] // self.n_features
            stations_reshaped = self.encode_station(
                station_data.reshape(
                    (station_data.shape[0], n_stations, self.n_features)
                )
            )
            main_repeated = main_data.repeat(1, n_stations, 1)
            stations_embedded = self.embedd_station(
                torch.concat([main_repeated, stations_reshaped], 2)
            )
            lintentioned_station = self.lintention_station(stations_embedded)
            features_stations = lintentioned_station.max(dim=1)[0]
            features.append(features_stations)

        return torch.concat(features, 1)


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
        action_space_max = action_space.nvec.max()
        TorchModelV2.__init__(
            self, obs_space, action_space, action_space_max, model_config, name
        )
        self.with_action_mask = with_action_mask
        self.n_features = self.obs_space.original_space["agvs"].shape[0] // fleetsize
        self.fleetsize = fleetsize
        self.n_stations = n_stations
        self.embed_dim = embed_dim
        self.name = name

        self.action_fe = LinFE(
            action_space_max=action_space_max,
            n_features=self.n_features,
            embed_dim=embed_dim,
            with_agvs=with_agvs,
            with_stations=with_stations,
        )
        self.value_fe = LinFE(
            action_space_max=action_space_max,
            n_features=self.n_features,
            embed_dim=embed_dim,
            with_agvs=with_agvs,
            with_stations=with_stations,
        )

        n_concats = 1 + with_agvs + with_stations

        self.action_net = nn.Sequential(
            nn.Linear(embed_dim * n_concats, embed_dim * 4),
            activation(),
            nn.Linear(embed_dim * 4, embed_dim),
            activation(),
            nn.Linear(embed_dim, action_space_max),
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
        self.obs = obsdict["obs"]
        features = self.action_fe(self.obs)

        actions = self.action_net(features)
        if self.with_action_mask:
            actions = actions + (obsdict["obs"]["action_mask"] - 1) * 1e8
        return actions, []

    def value_function(self):
        features = self.value_fe(self.obs)
        return self.value_net(features).flatten()
