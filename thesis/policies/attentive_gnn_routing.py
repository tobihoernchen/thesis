import gym.spaces
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import torch
from .custom_blocks import (
    MatrixGraph,
    MiniMatrixGraph,
    TransformerDecoderBlock,
    MatrixPositionEmbedder,
    FourierFeatureEmbedder
)
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential, GATv2Conv, GraphNorm


def register_attn_gnn_model():
    ModelCatalog.register_custom_model("attn_gnn_model", AGNNRoutingNet)


class AGNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        type,
        obs_space,
        embed_dim,
        with_stations,
        position_embedd_dim = 2,
        ff_embedd_dim = 0,
        activation=nn.ReLU,
        n_convolutions=2,
    ) -> None:
        super().__init__()
        self.with_stations = with_stations
        n_features = obs_space["agvs"].shape[1]

        if type == "matrix":
            self.basegraph = MatrixGraph()
        elif type == "minimatrix":
            self.basegraph = MiniMatrixGraph()
        position_embedd_dim = 2
        
        self.with_pos_embedding =  position_embedd_dim > 0
        self.with_ff_embedding =  ff_embedd_dim > 0
        if self.with_pos_embedding:
            self.pos_encoder = MatrixPositionEmbedder(position_embedd_dim)
        if self.with_ff_embedding:
            self.ff_encoder = FourierFeatureEmbedder(ff_embedd_dim)

        self.embed_dim = embed_dim
        self.embedd_main = self.get_embedder(n_features+ 7*position_embedd_dim + 7*2 * ff_embedd_dim, embed_dim, activation)
        self.embedd_agv = self.get_embedder(n_features+ 7*position_embedd_dim + 7*2 * ff_embedd_dim, embed_dim, activation)
        self.embedd_station = self.get_embedder(n_features+ position_embedd_dim + 2 * ff_embedd_dim, embed_dim, activation)
        self.embedd_node = self.get_embedder(2 + position_embedd_dim + 2 * ff_embedd_dim, embed_dim, activation)
        #self.embedd_edge = self.get_embedder(4 + 2*position_embedd_dim + 2*2*ff_embedd_dim, embed_dim, activation)
        self.edge_coords = nn.Parameter(
            torch.gather(
                self.basegraph.nodes.repeat(1, 2),
                0,
                self.basegraph.paths.T.repeat_interleave(2, dim=1),
            ), requires_grad=False
        )
        self.register_parameter("edge_coords", self.edge_coords)

        self.node_attention = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, 4, len(self.basegraph.nodes))
                for _ in range(2)
            ]
        )
        # self.edge_attention = nn.ModuleList(
        #     [
        #         TransformerDecoderBlock(embed_dim, 4, len(self.edge_coords))
        #         for _ in range(2)
        #     ]
        # )

        node_convs = []
        for i in range(n_convolutions):
            node_convs.extend(
                [
                    (
                        GATv2Conv(embed_dim, embed_dim),#, edge_dim=embed_dim),
                        "x, edge_index -> x"#, edge_attr -> x",
                    ),
                    GraphNorm(embed_dim),
                ]
            )
        self.node_convolutions = Sequential("x, edge_index", node_convs)#, edge_attr

    def get_embedder(self, n_features, embed_dim, activation):
        return nn.Sequential(
            nn.Linear(n_features, embed_dim * 2),
            activation(),
            nn.Linear(embed_dim * 2, embed_dim),
            activation(),
        )

    def encode(self, obs, pos_cols):
        if self.with_pos_embedding:
            obs=self.pos_encoder(obs, pos_cols)
        if self.with_ff_embedding:
            obs=self.ff_encoder(obs, pos_cols)
        return obs

    def forward(self, x):
        obs_main = x["agvs"][:, :1]
        obs_main = self.encode(obs_main, range(2, 15, 2))
        obs_agvs = x["agvs"][:, 1:]
        obs_agvs = self.encode(obs_agvs, range(2, 15, 2))
        if self.with_stations:
            obs_stations = x["stat"]
            obs_stations = self.encode(obs_stations, [0])

        in_reach_indices = self.basegraph.get_node_indices(
            obs_main[:, :, 8:16].reshape(-1, 4, 2)
        )

        main_embedded = self.embedd_main(obs_main)
        agvs_embedded = self.embedd_agv(obs_agvs)
        if self.with_stations:
            stations_embedded = self.embedd_station(obs_stations)
        objects_embedded = torch.concat(
            [main_embedded, agvs_embedded]
            + ([stations_embedded] if self.with_stations else []),
            dim=1,
        )
        nodes_encoded = self.encode(self.basegraph.nodes, [0])
        nodes_embedded = self.embedd_node(nodes_encoded).repeat(obs_main.shape[0], 1, 1)


        object_coords =  torch.concat([x["agvs"][:, :, 4:6], x["stat"][:, :, :2]], dim=1)
        object_coords_enlarged = object_coords[:, None, :, :].repeat(1, len(nodes_encoded), 1, 1)
        node_coords_enlarged = self.basegraph.nodes[None, :, None, :].repeat(object_coords_enlarged.shape[0], 1, object_coords_enlarged.shape[2], 1)
        mask = torch.logical_not(torch.all(torch.isclose(object_coords_enlarged, node_coords_enlarged, 0.05), 3))
        mask[:, :, 0] = False
        # edges_encoded = self.encode(self.edge_coords, [0, 2])
        # edges_embedded = self.embedd_edge(edges_encoded).repeat(obs_main.shape[0], 1, 1)
        for block in self.node_attention:
            nodes_embedded = block(objects_embedded, nodes_embedded, attn_mask = mask.repeat_interleave(4, 0))
        # for block in self.edge_attention:
        #     edges_embedded = block(objects_embedded, edges_embedded)

        in_geo_format = Batch.from_data_list(
            [
                Data(x=x, edge_index=self.basegraph.paths)#, edge_attr=attr)
                for x in nodes_embedded#, attr in zip(nodes_embedded, edges_embedded)
            ]
        )
        convoluted = self.node_convolutions(
            in_geo_format.x, in_geo_format.edge_index#, in_geo_format.edge_attr
        )
        convoluted_by_batch = convoluted.reshape(nodes_embedded.shape)
        filtered_in_reach = torch.gather(
            convoluted_by_batch,
            1,
            in_reach_indices.unsqueeze(-1).repeat(1, 1, convoluted_by_batch.shape[-1]),
        )
        features = filtered_in_reach
        return features.flatten(1)

    def out_features(self):
        return 4 * self.embed_dim


class AGNNRoutingNet(TorchModelV2, nn.Module):
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
            n_convolutions=6,
            env_type="matrix",
        ).items():
            setattr(
                self,
                key,
                custom_config[key] if key in custom_config.keys() else default,
            )

        self.name = name

        self.fe = AGNNFeatureExtractor(
            self.env_type,
            obs_space=obs_space.original_space,
            embed_dim=self.embed_dim,
            with_stations=self.with_stations,
            position_embedd_dim = self.position_embedd_dim,
            ff_embedd_dim = self.ff_embedd_dim,
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
        features = self.fe(self.obs)

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
        features = self.fe(self.obs)
        return self.value_net(features).flatten()
