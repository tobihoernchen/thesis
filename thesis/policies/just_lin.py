from turtle import forward
import gym.spaces
from torch import nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog


def register_lin_model():
    ModelCatalog.register_custom_model("lin_model", LinPolicy)


class PointMapping:
    def __init__(self, nodes: dict) -> None:
        self.nodes = torch.tensor(list(nodes.values()), dtype=float, device="cuda")
        self.indices = torch.arange(0, len(self.nodes) + 1, device="cuda")

    def __call__(self, points):
        points = points.to(dtype=float)
        nodes_rep = self.nodes.repeat(len(points), 1, 1)
        points_rep = points[:, None, :].repeat(1, len(self.nodes), 1)
        choice = torch.all(torch.isclose(points_rep, nodes_rep, atol=0.01), axis=2)
        anyone = torch.logical_not(torch.any(choice, axis=1))
        choice = torch.concat([anyone.unsqueeze(1), choice], axis=1)
        return self.indices.repeat(len(points), 1)[choice]


class PositionEncoder(nn.Module):
    def __init__(
        self, pos_cols, original_dim, embed_size, resolution=100, mask_valid=None
    ) -> None:
        super().__init__()
        self.original_dim = original_dim
        self.pointmapper = PointMapping(
            {
                0: (0.5454545454545454, 0.76),
                1: (0.6022727272727273, 0.76),
                2: (0.5454545454545454, 0.86),
                3: (0.6022727272727273, 0.86),
                4: (0.4772727272727273, 0.76),
                5: (0.42045454545454547, 0.76),
                6: (0.42045454545454547, 0.86),
                7: (0.4772727272727273, 0.86),
                8: (0.32954545454545453, 0.808),
                9: (0.42045454545454547, 0.48),
                10: (0.4772727272727273, 0.48),
                11: (0.4772727272727273, 0.38),
                12: (0.42045454545454547, 0.38),
                13: (0.32954545454545453, 0.428),
                14: (0.5727272727272728, 0.62),
                15: (0.7613636363636364, 0.76),
                16: (0.8181818181818182, 0.76),
                17: (0.8181818181818182, 0.86),
                18: (0.7613636363636364, 0.86),
                19: (0.7909090909090909, 0.62),
                20: (0.9431818181818182, 0.76),
                21: (1.0, 0.76),
                22: (1.0, 0.86),
                23: (0.9431818181818182, 0.86),
                24: (0.9727272727272728, 0.62),
                25: (0.9727272727272728, 1.0),
            }
        )
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
                blocks.append(x[:, :, : col + 2])
            else:
                blocks.append(x[:, :, last_col + 2 : col + 2])
            last_col = col
            to_encode = x[:, :, col : col + 2]
            ints = self.pointmapper(to_encode.flatten(end_dim=1))
            encoded = self.embedding(ints.to(dtype=int)).reshape(
                to_encode.shape[:-1] + (self.embed_size,)
            )
            blocks.append(encoded)
        blocks.append(x[:, :, col + 2 :])
        out = torch.concat(blocks, 2)
        return out

    def out_features(self):
        return self.original_dim + len(self.pos_cols) * (self.embed_size)


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


class LinFE(nn.Module):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
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
            nn.Linear(embed_dim + action_space.n, 2 * embed_dim),
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
        TorchModelV2.__init__(
            self, obs_space, action_space, action_space.n, model_config, name
        )
        self.with_action_mask = with_action_mask
        self.n_features = self.obs_space.original_space["agvs"].shape[0] // fleetsize
        self.fleetsize = fleetsize
        self.n_stations = n_stations
        self.embed_dim = embed_dim
        self.name = name

        self.action_fe = LinFE(
            action_space=action_space,
            n_features=self.n_features,
            embed_dim=embed_dim,
            with_agvs=with_agvs,
            with_stations=with_stations,
        )
        self.value_fe = LinFE(
            action_space=action_space,
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
        self.obs = obsdict["obs"]
        features = self.action_fe(self.obs)

        actions = self.action_net(features)
        if self.with_action_mask:
            actions = actions + (obsdict["obs"]["action_mask"] - 1) * 1e8
        return actions, []

    def value_function(self):
        features = self.value_fe(self.obs)
        return self.value_net(features).flatten()
