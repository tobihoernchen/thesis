import gym.spaces
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import torch
from .custom_blocks import MatrixGraph, MiniMatrixGraph, MatrixPositionEmbedder, FourierFeatureEmbedder, TransformerDecoderBlock
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential, GATv2Conv, GraphNorm,  GCNConv
from torch_geometric.utils import scatter, k_hop_subgraph


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
        self.n_convolutions = n_convolutions
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

        self.embedd_node = self.get_embedder(
            15,
            embed_dim,
            len(self.basegraph.nodes),
            activation,
            d2 = False
        )

        node_convs = []
        for i in range(n_convolutions):
            node_convs.extend(
                [
                    (
                         GCNConv(embed_dim, embed_dim),
                        "x, edge_index -> x"
                    ),
                    #GraphNorm(embed_dim),
                ]
            )
        self.node_convolutions = Sequential("x, edge_index", node_convs)



    def get_embedder(self, n_features, embed_dim, n_objects, activation, d2=True):
        return nn.Sequential(
            nn.Linear(n_features, embed_dim * 2),
            activation(),
            nn.BatchNorm2d(n_objects) if d2 else nn.BatchNorm1d(n_objects),
            nn.Linear(embed_dim * 2, embed_dim),
            activation(),
            nn.BatchNorm2d(n_objects) if d2 else nn.BatchNorm1d(n_objects),
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
        main_coords = obs_main[:, :, 2:16].reshape(obs_main.shape[0], 1, 7, 2)
        main_indices = self.basegraph.get_node_indices(main_coords)
        agv_coords = obs_agvs[:, :, :, 2:16].reshape(obs_agvs.shape[0], -1, 7, 2)  
        agv_indices = self.basegraph.get_node_indices(agv_coords)
        src_main = torch.ones(obs_main.shape[0], main_indices.shape[1], 7, device=obs_main.device)
        src_agvs = torch.ones(obs_main.shape[0], agv_indices.shape[1], 7, device=obs_main.device)
        node_info_main = torch.zeros(obs_main.shape[0], self.basegraph.nodes.shape[0], 7, device=obs_main.device)
        node_info_agvs = torch.zeros(obs_main.shape[0], self.basegraph.nodes.shape[0], 7, device=obs_main.device)
        node_info_main.scatter_(1, main_indices, src_main, reduce = "add")
        node_info_agvs.scatter_(1, agv_indices, src_agvs, reduce = "add")
        distances = torch.norm(self.basegraph.nodes[None, :].repeat(obs_main.shape[0], 1, 1) - obs_main[:, :, 6:8], dim=2)
        node_info = self.embedd_node(torch.concat([node_info_main, node_info_agvs, distances.unsqueeze(-1)], dim = 2).detach())

        graphs = []
        for i in range(obs_main.shape[0]):
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(int(self.basegraph.get_node_indices(obs_main[i, 0, 4:6])), self.n_convolutions, self.basegraph.paths, relabel_nodes=True)
            graphs.append(Data(node_info[i, subset], edge_index))
        batch:Batch = Batch.from_data_list(graphs)

        convoluted = self.node_convolutions(
            batch.x, batch.edge_index
        )
        out = scatter(convoluted, batch.batch, dim=0, reduce='mean')

        return out


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
