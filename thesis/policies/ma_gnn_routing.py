import gym.spaces
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import torch
from .custom_blocks import MatrixGraph, MiniMatrixGraph, MatrixPositionEmbedder, FourierFeatureEmbedder, TransformerDecoderBlock
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential, GATv2Conv, GraphNorm


def register_gnn_model():
    ModelCatalog.register_custom_model("gnn_model", GNNRoutingNet)


class GNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        type,
        obs_space,
        n_features,
        embed_dim,
        with_stations,
        position_embedd_dim = 2,
        ff_embedd_dim = 0,
        activation=nn.ReLU,
        n_convolutions=2,
    ) -> None:
        super().__init__()
        self.with_stations = with_stations
        if type == "matrix":
            self.basegraph = MatrixGraph()
        elif type == "minimatrix":
            self.basegraph = MiniMatrixGraph()

        self.with_pos_embedding =  position_embedd_dim > 0
        self.with_ff_embedding =  ff_embedd_dim > 0
        if self.with_pos_embedding:
            self.pos_encoder = MatrixPositionEmbedder(position_embedd_dim)
        if self.with_ff_embedding:
            self.ff_encoder = FourierFeatureEmbedder(ff_embedd_dim)

        self.pathencoder = MatrixPositionEmbedder(4)

        self.embedd_main = self.get_embedder(
            n_features + 2 + 7 * position_embedd_dim + 2* 7 * ff_embedd_dim,
            embed_dim,
            len(self.basegraph.nodes),
            activation,
        )
        self.embedd_agv = self.get_embedder(
            n_features + 2 + 7 * position_embedd_dim + 2* 7 * ff_embedd_dim,
            embed_dim,
            len(self.basegraph.nodes),
            activation,
        )
        self.embedd_station = self.get_embedder(
            n_features + 2 + position_embedd_dim + 2 * ff_embedd_dim,
            embed_dim,
            len(self.basegraph.nodes),
            activation,
        )

        node_convs = []
        for i in range(n_convolutions):
            node_convs.extend(
                [
                    (
                        GATv2Conv(embed_dim, embed_dim),
                        "x, edge_index -> x"
                    ),
                    GraphNorm(embed_dim),
                ]
            )
        self.node_convolutions = Sequential("x, edge_index", node_convs)



    def get_embedder(self, n_features, embed_dim, n_objects, activation, d2=True):
        return nn.Sequential(
            nn.Linear(n_features, embed_dim * 2),
            nn.BatchNorm2d(n_objects) if d2 else nn.BatchNorm1d(n_objects),
            activation(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm2d(n_objects) if d2 else nn.BatchNorm1d(n_objects),
            activation(),
        )

    def obs_node_expansion(self, obs, nodes):
        return torch.concat(
            [
                obs.repeat((1, nodes.shape[1], 1, 1)),
                nodes.repeat((obs.shape[0], 1, obs.shape[2], 1)),
            ],
            dim=-1,
        )

    def forward(self, x):
        obs_main = x["agvs"][:, :1]
        if self.with_pos_embedding:
            obs_main=self.pos_encoder(obs_main, range(2, 15, 2))
        if self.with_ff_embedding:
            obs_main=self.ff_encoder(obs_main, range(2, 15, 2))
        obs_agvs = x["agvs"][:, None, 1:]
        if self.with_pos_embedding:
            obs_agvs=self.pos_encoder(obs_agvs, range(2, 15, 2))
        if self.with_ff_embedding:
            obs_agvs=self.ff_encoder(obs_agvs, range(2, 15, 2))
        if self.with_stations:
            obs_stations = x["stat"][:, None]
            if self.with_pos_embedding:
                obs_stations=self.pos_encoder(obs_stations, [0])
            if self.with_ff_embedding:
                obs_stations=self.ff_encoder(obs_stations, [0])
        # in_reach_indices = self.basegraph.get_node_indices(
        #     obs_main[:, None, :, 8:16].reshape(-1, 4, 2)
        # )
        nodes = self.basegraph.nodes[None, :, None, :]
        main_expanded = self.obs_node_expansion(obs_main[:, None], nodes)
        agvs_expanded = self.obs_node_expansion(obs_agvs, nodes)
        if self.with_stations:
            stations_expanded = self.obs_node_expansion(obs_stations, nodes)

        main_embedded = self.embedd_main(main_expanded)
        agvs_embedded = self.embedd_agv(agvs_expanded)
        if self.with_stations:
            stations_embedded = self.embedd_station(stations_expanded)

        node_info_raw = torch.concat(
            [main_embedded, agvs_embedded]
            + ([stations_embedded] if self.with_stations else []),
            dim=2,
        )

        node_info = node_info_raw.mean(dim=2)

        in_geo_format = Batch.from_data_list(
            [
                Data(x=x, edge_index=self.basegraph.paths)
                for x in node_info
            ]
        )
        convoluted = self.node_convolutions(
            in_geo_format.x, in_geo_format.edge_index
        )
        convoluted_by_batch = convoluted.reshape(node_info.shape)
        # filtered_in_reach = torch.gather(
        #     convoluted_by_batch,
        #     1,
        #     in_reach_indices.unsqueeze(-1).repeat(1, 1, convoluted_by_batch.shape[-1]),
        # )
        return convoluted_by_batch#.flatten(1)


class GNNRoutingNet(TorchModelV2, nn.Module):
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
            with_stations=True,
            position_embedd_dim = 2,
            ff_embedd_dim = 0,
            activation=nn.ReLU,
            n_convolutions=2,
            env_type="matrix",
        ).items():
            setattr(
                self,
                key,
                custom_config[key] if key in custom_config.keys() else default,
            )

        self.n_features = self.obs_space.original_space["agvs"].shape[1]
        self.name = name

        self.fe = GNNFeatureExtractor(
            self.env_type,
            obs_space=obs_space.original_space,
            n_features=self.n_features,
            embed_dim=self.embed_dim,
            position_embedd_dim = self.position_embedd_dim,
            ff_embedd_dim = self.ff_embedd_dim,
            with_stations=self.with_stations,
            n_convolutions=self.n_convolutions,
        )


        self.embedd_main_action = self.get_embedder(
            self.n_features,
            self.embed_dim,
            1,
            d2=False,
            activation=nn.ReLU
        )
        self.action_attention = TransformerDecoderBlock(self.embed_dim, 4, 1)
        self.embedd_main_value = self.get_embedder(
            self.n_features,
            self.embed_dim,
            1,
            d2=False,
            activation=nn.ReLU
        )
        self.value_attention = TransformerDecoderBlock(self.embed_dim, 4, 1)

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

    def get_embedder(self, n_features, embed_dim, n_objects, activation, d2=True):
        return nn.Sequential(
            nn.Linear(n_features, embed_dim * 2),
            nn.BatchNorm2d(n_objects) if d2 else nn.BatchNorm1d(n_objects),
            activation(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm2d(n_objects) if d2 else nn.BatchNorm1d(n_objects),
            activation(),
        )
    def forward(self, obsdict, state, seq_lengths):
        self.obs = obsdict["obs"]
        self.features = self.fe(self.obs)
        main_key = self.embedd_main_action(self.obs["agvs"][:, :1])
        features = self.action_attention(self.features, main_key).flatten(1)
        actions = self.action_net(features)
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
        main_key = self.embedd_main_value(self.obs["agvs"][:, :1])
        features = self.value_attention(self.features, main_key).flatten(1)
        return self.value_net(features).flatten()
