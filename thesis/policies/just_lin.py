import gym.spaces
from torch import nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from .custom_blocks import MatrixPositionEncoder


def register_lin_model():
    ModelCatalog.register_custom_model("lin_model", LinPolicy)


class LinFE(nn.Module):
    def __init__(
        self,
        n_features,
        embed_dim=64,
        with_agvs=True,
        with_stations=True,
        position_embedd_dim = 2,
        activation=nn.ReLU,
    ) -> None:

        super().__init__()
        self.encoder = MatrixPositionEncoder(position_embedd_dim)

        self.main_ff = nn.Sequential(
            nn.Linear(n_features + position_embedd_dim * 7, 2 * embed_dim),
            activation(),
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            activation(),
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            activation(),
            nn.Linear(2 * embed_dim, embed_dim),
            activation(),
        )

        self.with_agvs = with_agvs
        if with_agvs:
            self.lintention_agvs = nn.Sequential(
                nn.Linear(2 * n_features + position_embedd_dim * 14, 4 * embed_dim),
                activation(),
                nn.Linear(4 * embed_dim, 4 * embed_dim),
                activation(),
                nn.Linear(4 * embed_dim, 2 * embed_dim),
                activation(),
                nn.Linear(2 * embed_dim, embed_dim),
                activation(),
            )

        self.with_stations = with_stations
        if with_stations:
            self.lintention_station = nn.Sequential(
                nn.Linear(2 * n_features + position_embedd_dim * 8, 4 * embed_dim),
                activation(),
                nn.Linear(4 * embed_dim, 4 * embed_dim),
                activation(),
                nn.Linear(4 * embed_dim, 2 * embed_dim),
                activation(),
                nn.Linear(2 * embed_dim, embed_dim),
                activation(),
            )

    def forward(self, obs):
        features = []
        main_data = self.encoder(obs["agvs"][:, 1], range(1, 14, 2))[:, None, :]
        features_main = self.main_ff(main_data).squeeze(dim=1)
        features.append(features_main)

        if self.with_agvs:
            agvs_data = obs["agvs"][:, 1:]
            agvs_reshaped = self.encoder(agvs_data, range(1, 14, 2))
            main_repeated = main_data.repeat(1, obs["agvs"].shape[1] - 1, 1)
            lintentioned_agvs = self.lintention_agvs(
                torch.concat([main_repeated, agvs_reshaped], 2)
            )
            features_agvs = lintentioned_agvs.max(dim=1)[0]
            features.append(features_agvs)

        if self.with_stations:
            station_data = obs["stat"]
            stations_reshaped = self.encoder(
                station_data,
                [
                    0,
                ],
            )
            main_repeated = main_data.repeat(1, obs["stat"].shape[1], 1)
            lintentioned_station = self.lintention_station(
                torch.concat([main_repeated, stations_reshaped], 2)
            )
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
        embed_dim=64,
        with_action_mask=True,
        with_agvs=True,
        with_stations=True,
        activation=nn.ReLU,
    ):
        nn.Module.__init__(self)
        action_space_max = action_space.nvec.max()
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        self.with_action_mask = with_action_mask
        self.n_features = self.obs_space.original_space["agvs"].shape[1]
        self.embed_dim = embed_dim
        self.name = name

        self.action_fe = LinFE(
            n_features=self.n_features,
            embed_dim=embed_dim,
            with_agvs=with_agvs,
            with_stations=with_stations,
        )
        self.value_fe = LinFE(
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
        action_tmp = torch.zeros(
            obsdict["obs"]["action_mask"].shape,
            device=actions.device,
        )
        action_tmp[:, 0, :] = actions
        if self.with_action_mask:
            action_tmp = action_tmp + (obsdict["obs"]["action_mask"] - 1) * 1e8
        return action_tmp.flatten(1), []

    def value_function(self):
        features = self.value_fe(self.obs)
        return self.value_net(features).flatten()
