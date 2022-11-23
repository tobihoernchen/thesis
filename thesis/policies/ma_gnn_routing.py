import gym.spaces
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import torch
from .custom_blocks import MatrixGraph, MiniMatrixGraph
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential, GCNConv


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
        self.bits_to_add = nn.Parameter(
            torch.Tensor(
                [(1, 0)]
                + [(0, 0) for i in range(obs_space["agvs"].shape[0] - 1)]
                + (
                    [(0, 1) for i in range(obs_space["stat"].shape[0])]
                    if with_stations
                    else []
                )
            )
        )
        self.register_parameter("bits", self.bits_to_add)
        if type == "matrix":
            self.basegraph = MatrixGraph()
        elif type == "minimatrix":
            self.basegraph = MiniMatrixGraph()

        self.embedd = nn.Sequential(
            nn.Linear(n_features + 2, embed_dim * 2),
            activation(),
            nn.Linear(embed_dim * 2, embed_dim),
            activation(),
        )

        node_convs = []
        for i in range(n_convolutions):
            node_convs.extend(
                [
                    (GCNConv(embed_dim, embed_dim), "x, edge_index -> x"),
                    nn.ReLU(inplace=True),
                ]
            )
        self.node_convolutions = Sequential("x, edge_index", node_convs)

    def forward(self, x):
        obs = (
            x["agvs"]
            if not self.with_stations
            else torch.concat([x["agvs"], x["stat"]], 1)
        )
        obs_with_bits = torch.concat(
            [obs, self.bits_to_add.repeat(obs.shape[0], 1, 1)], 2
        )
        obs_embedded = self.embedd(obs_with_bits)
        in_reach_indices = self.basegraph.get_node_indices(
            x["agvs"][:, 0, 8:16].reshape(-1, 4, 2)
        )
        target_indices = self.basegraph.get_node_indices(
            x["agvs"][:, 0, 6:8].reshape(-1, 2)
        )
        node_coordinates = (
            torch.concat([x["agvs"][..., 4:6], x["stat"][..., :2]])
            if self.with_stations
            else x["agvs"][..., 4:6]
        )
        node_indices = (
            self.basegraph.get_node_indices(node_coordinates)
            .repeat_interleave(obs_embedded.shape[-1], -1)
            .reshape(obs_embedded.shape[:1] + (1,) + obs_embedded.shape[1:])
        ).to(dtype=torch.int64)
        nodewise = (
            torch.zeros(
                obs_embedded.shape[0],
                len(self.basegraph.nodes) + 1,
                obs_embedded.shape[1],
                obs_embedded.shape[2],
                device=node_indices.device,
            )
            .scatter_(1, node_indices, obs_embedded.unsqueeze(1))
            .sum(2)
        )[:, :-1]
        for batch, target in enumerate(target_indices):
            if target < nodewise.shape[1] - 1:
                nodewise[batch, target] = 1
        in_geo_format = Batch.from_data_list(
            [Data(x=x, edge_index=self.basegraph.paths) for x in nodewise]
        )
        convoluted = self.node_convolutions(in_geo_format.x, in_geo_format.edge_index)
        convoluted_by_batch = convoluted.reshape(nodewise.shape)
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
        action_space_max = (
            action_space.nvec.max()
            if isinstance(action_space, gym.spaces.MultiDiscrete)
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
            discrete_action_space=False,
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

        self.action_fe = GNNFeatureExtractor(
            self.env_type,
            obs_space=obs_space.original_space,
            n_features=self.n_features,
            embed_dim=self.embed_dim,
            with_stations=self.with_stations,
            n_convolutions=self.n_convolutions,
        )
        self.value_fe = GNNFeatureExtractor(
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
        features = self.action_fe(self.obs)

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
        features = self.value_fe(self.obs)
        return self.value_net(features).flatten()
