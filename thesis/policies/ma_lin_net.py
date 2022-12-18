import gym.spaces
from torch import nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from .custom_blocks import MatrixPositionEmbedder, FourierFeatureEmbedder


def register_lin_model():
    ModelCatalog.register_custom_model("lin_model", MALinPolicy)


class LinFE(nn.Module):
    def __init__(
        self,
        obs_space,
        embed_dim=64,
        with_agvs=True,
        with_stations=True,
        position_embedd_dim=2,
        ff_embedd_dim = 0,
        activation=nn.ReLU,
    ) -> None:

        super().__init__()
        self.with_pos_embedding =  position_embedd_dim > 0
        self.with_ff_embedding =  ff_embedd_dim > 0
        if self.with_pos_embedding:
            self.pos_encoder = MatrixPositionEmbedder(position_embedd_dim)
        if self.with_ff_embedding:
            self.ff_encoder = FourierFeatureEmbedder(ff_embedd_dim)
        n_features = obs_space["agvs"].shape[1]

        self.main_ff = nn.Sequential(
            nn.Linear(n_features + position_embedd_dim * 7 + ff_embedd_dim * 2 * 7, 2 * embed_dim),
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
                nn.Linear(2 * n_features + position_embedd_dim * 14 + ff_embedd_dim * 2 * 14, 4 * embed_dim),
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
                nn.Linear(2 * n_features + position_embedd_dim * 8  + ff_embedd_dim * 2 * 8, 4 * embed_dim),
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
        main_data = obs["agvs"][:, 0:1]
        if self.with_pos_embedding:
            main_data = self.pos_encoder(main_data, range(2, 15, 2))
        if self.with_ff_embedding:
            main_data = self.ff_encoder(main_data, range(2, 15, 2))
        features_main = self.main_ff(main_data).squeeze(dim=1)
        features.append(features_main)

        if self.with_agvs:
            agvs_data = obs["agvs"][:, 1:]
            if self.with_pos_embedding:
                agvs_data = self.pos_encoder(agvs_data, range(2, 15, 2))
            if self.with_ff_embedding:
                agvs_data = self.ff_encoder(agvs_data, range(2, 15, 2))
            main_repeated = main_data.repeat(1, obs["agvs"].shape[1] - 1, 1)
            lintentioned_agvs = self.lintention_agvs(
                torch.concat([main_repeated, agvs_data], 2)
            )
            features_agvs_max = lintentioned_agvs.max(dim=1)[0] 
            features_agvs_mean = lintentioned_agvs.mean(dim=1)
            features.append(features_agvs_mean)
            features.append(features_agvs_max)

        if self.with_stations:
            station_data = obs["stat"]
            if self.with_pos_embedding:
                station_data = self.pos_encoder(station_data, [0])
            if self.with_ff_embedding:
                station_data = self.ff_encoder(station_data, [0])
            main_repeated = main_data.repeat(1, obs["stat"].shape[1], 1)
            lintentioned_station = self.lintention_station(
                torch.concat([main_repeated, station_data], 2)
            )
            features_stations_max = lintentioned_station.max(dim=1)[0]
            features_stations_mean = lintentioned_station.mean(dim=1)
            features.append(features_stations_max)
            features.append(features_stations_mean)

        return torch.concat(features, 1)


class MALinPolicy(TorchModelV2, nn.Module):
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
            with_agvs=True,
            with_stations=True,
            position_embedd_dim = 2,
            ff_embedd_dim = 0,
            activation=nn.ReLU,
        ).items():
            setattr(
                self,
                key,
                custom_config[key] if key in custom_config.keys() else default,
            )

        self.name = name

        self.fe = LinFE(
            obs_space=obs_space.original_space,
            embed_dim=self.embed_dim,
            with_agvs=self.with_agvs,
            with_stations=self.with_stations,
            position_embedd_dim = self.position_embedd_dim,
            ff_embedd_dim = self.ff_embedd_dim
        )

        n_concats = 1 + 2* self.with_agvs +  2 * self.with_stations

        self.action_net = nn.Sequential(
            nn.Linear(self.embed_dim * n_concats, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            self.activation(),
            nn.Linear(
                self.embed_dim,
                action_space_max if action_space_max is not None else num_outputs,
            ),
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.embed_dim * n_concats, self.embed_dim * 4),
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
            action_out = torch.zeros(
                obsdict["obs"]["action_mask"].shape,
                device=actions.device,
            )
            action_out[:, 0, :] = actions
        if self.with_action_mask:
            action_out = action_out + (obsdict["obs"]["action_mask"] - 1) * 1e8
        return action_out.flatten(1), []

    def value_function(self):
        return self.value_net(self.features).flatten()
