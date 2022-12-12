import gym.spaces
from torch import nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from .custom_blocks import MatrixPositionEncoder, TransformerDecoderBlock, TransformerEncoderBlock


def register_attn_model():
    ModelCatalog.register_custom_model("attn_model", MAAttnPolicy)


class AttnFE(nn.Module):
    def __init__(
        self,
        obs_space,
        embed_dim=64,
        with_agvs=True,
        with_stations=True,
        position_embedd_dim=2,
        n_heads=4,
        depth=4,
        activation=nn.ReLU,
    ) -> None:

        super().__init__()
        n_features = obs_space["agvs"].shape[1]
        self.with_agvs = with_agvs
        self.with_stations = with_stations
        n_agents = 0
        if with_agvs:
            n_agents += obs_space["agvs"].shape[0] - 1
        if with_stations:
            n_agents += obs_space["stat"].shape[0]
        assert (
            with_stations or with_agvs
        ), "Either with_stations or with_agvs has to bet true"
        self.encoder = MatrixPositionEncoder(position_embedd_dim)

        self.embedd_main = self.get_embedder(
            n_features + position_embedd_dim * 7, embed_dim, activation
        )
        if self.with_agvs:
            self.embedd_agvs = self.get_embedder(
                n_features + position_embedd_dim * 7, embed_dim, activation
            )
        if self.with_stations:
            self.embedd_station = self.get_embedder(
                n_features + position_embedd_dim, embed_dim, activation
            )

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim, n_heads, n_agents, activation
                )
                for _ in range(depth)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_dim, n_heads, 1, activation
                )
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
        main_data = self.encoder(obs["agvs"][:, 0], range(2, 15, 2))[:, None, :]
        features_main = self.embedd_main(main_data)

        objects = []
        if self.with_agvs:
            agvs_data = self.encoder(obs["agvs"][:, 1:], range(2, 15, 2))
            features_agvs = self.embedd_agvs(agvs_data)
            objects.append(features_agvs)

        if self.with_stations:
            station_data = self.encoder(obs["agvs"][:, 1:], [0])
            features_stations = self.embedd_station(station_data)
            objects.append(features_stations)

        all_objects = torch.concat(objects, 1)
        for block in self.encoder_blocks:
            all_objects = block(all_objects)
        for block in self.decoder_blocks:
            features_main = block(all_objects, features_main)
        return features_main.flatten(1)


class MAAttnPolicy(TorchModelV2, nn.Module):
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
            activation=nn.ReLU,
        ).items():
            setattr(
                self,
                key,
                custom_config[key] if key in custom_config.keys() else default,
            )

        self.name = name

        self.fe = AttnFE(
            obs_space=obs_space.original_space,
            embed_dim=self.embed_dim,
            with_agvs=self.with_agvs,
            with_stations=self.with_stations,
        )

        self.action_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            self.activation(),
            nn.Linear(
                self.embed_dim,
                action_space_max if action_space_max is not None else num_outputs,
            ),
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
