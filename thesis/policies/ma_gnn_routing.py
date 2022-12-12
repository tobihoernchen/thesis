import gym.spaces
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import torch
from .custom_blocks import MatrixGraph, MiniMatrixGraph, MatrixPositionEncoder
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential, GATConv, BatchNorm


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
        activation=nn.ReLU,
        n_convolutions=6,
    ) -> None:
        super().__init__()
        self.with_stations = with_stations
        if type == "matrix":
            self.basegraph = MatrixGraph()
        elif type == "minimatrix":
            self.basegraph = MiniMatrixGraph()

        self.paths = nn.Parameter(self.basegraph.paths.T[None, :], requires_grad=False)
        self.register_parameter("paths", self.paths)
        self.encoder = MatrixPositionEncoder(2)
        self.pathencoder = MatrixPositionEncoder(4)

        self.embedd_main = self.get_embedder(
            n_features + 2 + 7 * 2,
            int(embed_dim / 2),
            len(self.basegraph.nodes),
            activation,
        )
        self.embedd_agv = self.get_embedder(
            n_features + 2 + 7 * 2,
            int(embed_dim / 2),
            len(self.basegraph.nodes),
            activation,
        )
        self.embedd_station = self.get_embedder(
            n_features + 2 + 2,
            int(embed_dim / 2),
            len(self.basegraph.nodes),
            activation,
        )

        self.embedd_paths = self.get_embedder(
            2 * embed_dim + 4, embed_dim, self.paths.shape[1], activation, d2=False
        )

        node_convs = []
        for i in range(n_convolutions):
            node_convs.extend(
                [
                    (
                        GATConv(embed_dim, embed_dim, edge_dim=embed_dim),
                        "x, edge_index, edge_attr -> x",
                    ),
                    BatchNorm(embed_dim),
                ]
            )
        self.node_convolutions = Sequential("x, edge_index, edge_attr", node_convs)

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
        obs_main = self.encoder(x["agvs"][:, None, :1], range(2, 15, 2))
        obs_agvs = self.encoder(x["agvs"][:, None, 1:], range(2, 15, 2))
        if self.with_stations:
            obs_stations = self.encoder(x["stat"][:, None], [0])
        in_reach_indices = self.basegraph.get_node_indices(
            obs_main[:, :, :, 8:16].reshape(-1, 4, 2)
        )
        nodes = self.basegraph.nodes[None, :, None, :]
        main_expanded = self.obs_node_expansion(obs_main, nodes)
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

        node_info = torch.concat(
            [node_info_raw.mean(dim=2), node_info_raw.max(dim=2)[0]], dim=2
        )
        path_info_raw = torch.gather(
            node_info.repeat(1, 1, 2),
            1,
            self.paths.repeat(node_info.shape[0], 1, 1).repeat_interleave(
                node_info.shape[2], dim=2
            ),
        )
        path_info = self.embedd_paths(
            torch.concat(
                [
                    path_info_raw,
                    self.pathencoder(self.paths, [0])[:,:,2:].repeat(node_info.shape[0], 1, 1),
                ],
                dim=2,
            )
        )

        in_geo_format = Batch.from_data_list(
            [
                Data(x=x, edge_index=self.basegraph.paths, edge_attr=attr)
                for x, attr in zip(node_info, path_info)
            ]
        )
        convoluted = self.node_convolutions(
            in_geo_format.x, in_geo_format.edge_index, in_geo_format.edge_attr
        )
        convoluted_by_batch = convoluted.reshape(node_info.shape)
        filtered_in_reach = torch.gather(
            convoluted_by_batch,
            1,
            in_reach_indices.unsqueeze(-1).repeat(1, 1, convoluted_by_batch.shape[-1]),
        )
        return filtered_in_reach.flatten(1)


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
            activation=nn.ReLU,
            n_convolutions=6,
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
            with_stations=self.with_stations,
            n_convolutions=self.n_convolutions,
        )

        self.action_net = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            self.activation(),
            nn.Linear(
                self.embed_dim,
                action_space_max if action_space_max is not None else num_outputs,
            ),
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
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
