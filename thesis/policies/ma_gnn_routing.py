import gym.spaces
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import torch
from .custom_blocks import MatrixGraph, MiniMatrixGraph, MatrixPositionEmbedder, FourierFeatureEmbedder, TransformerDecoderBlock
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential, GATv2Conv, GraphNorm
from torch_geometric.utils import scatter, k_hop_subgraph
from torch_scatter import scatter

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
        activation=nn.LeakyReLU,
        n_convolutions=4,
        with_node_info=False
    ) -> None:
        super().__init__()
        self.n_convolutions = n_convolutions
        if type == "matrix":
            self.basegraph = MatrixGraph()
        elif type == "minimatrix":
            self.basegraph = MiniMatrixGraph()
        self.with_node_info = with_node_info
        self.with_pos_embedding =  position_embedd_dim > 0
        self.with_ff_embedding =  ff_embedd_dim > 0
        if self.with_pos_embedding:
            self.pos_encoder = MatrixPositionEmbedder(position_embedd_dim)
        if self.with_ff_embedding:
            self.ff_encoder = FourierFeatureEmbedder(ff_embedd_dim)

        self.embedd_goal = self.get_embedder(4 + position_embedd_dim * 2 + ff_embedd_dim * 4+1, embed_dim, 1, activation, False)
        self.embedd_node = self.get_embedder(22 if not self.with_node_info else 24, embed_dim, len(self.basegraph.nodes), activation, False)

        node_convs = []
        for i in range(n_convolutions):
            node_convs.extend(
                [
                    (
                         GATv2Conv(embed_dim, embed_dim, 4, False),
                        "x, edge_index -> x"
                    ),
                    (
                        GraphNorm(embed_dim),
                        "x, batch -> x"
                    )
                ]
            )
        self.node_convolutions = Sequential("x, edge_index, batch", node_convs)



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
        coords:torch.Tensor = x["agvs"][:, :, 2:16].reshape(x["agvs"].shape[0], -1, 7, 2) 
        distance_percentage = x["agvs"][:, :, 16:18]
        moving = x["agvs"][:, :, 1]
        stat_coords = x["stat"][:, :, :2].reshape(x["stat"].shape[0], -1, 1, 2) 
        obs_main:torch.Tensor = x["agvs"][:, :1]
        goal_features = torch.cat([obs_main[:, :1, 4:8], obs_main[:, :1, 18:19]], -1)
        if self.with_pos_embedding:
            goal_features = self.pos_encoder(goal_features, [0, 2])
        if self.with_ff_embedding:
            goal_features = self.ff_encoder(goal_features, [0, 2])
        goal_features = self.embedd_goal(goal_features).squeeze(1)

        indices = self.basegraph.get_node_indices(coords)
        indices = torch.concat([indices, indices[:,:,0:1].repeat(1,1,2), indices[:,:,1:2]], dim=-1).to(dtype=torch.int64)
        main_indices = indices[:, :1]
        agv_indices = indices[:, 1:]
        stat_indices = self.basegraph.get_node_indices(stat_coords)

        src_main = torch.ones_like(main_indices, dtype = obs_main.dtype, device=obs_main.device)
        src_main[:, :, 7] = moving[:, :1]  #moving
        src_main[:, :, 8] = distance_percentage[:, :1, 0]  #distance_last
        src_main[:, :, 9] = distance_percentage[:, :1, 1]  #distance_next
        src_agvs = torch.ones_like(agv_indices, dtype = obs_main.dtype, device=obs_main.device)
        src_agvs[:, :, 7] = moving[:, 1:]  #moving
        src_agvs[:, :, 8] = distance_percentage[:, 1:, 0]  #distance_last
        src_agvs[:, :, 9] = distance_percentage[:, 1:, 1]  #distance_next
        src_stat = torch.ones_like(stat_indices, dtype = obs_main.dtype, device=obs_main.device)

        node_info_main= scatter(src_main, main_indices, dim=1,dim_size = len(self.basegraph.nodes), reduce = "add")
        node_info_agvs = scatter(src_agvs, agv_indices, dim=1,dim_size = len(self.basegraph.nodes),   reduce = "add")
        node_info_stat = scatter(src_stat, stat_indices, dim=1,dim_size = len(self.basegraph.nodes),   reduce = "add")

        distances = torch.norm(self.basegraph.nodes[None, :].repeat(obs_main.shape[0], 1, 1) - obs_main[:, :, 6:8], dim=2)
        node_info = torch.concat([node_info_main, node_info_agvs, node_info_stat, distances.unsqueeze(-1)], dim = 2).detach()
        if self.with_node_info:
            node_info = torch.concat([node_info, self.basegraph.nodes[None, :].repeat(obs_main.shape[0], 1, 1)], dim = 2).detach()
        node_info = self.embedd_node(node_info)
        in_reach_indices = self.basegraph.get_node_indices(
            obs_main[:, :, 8:16].reshape(-1, 4, 2)
        )

        graphs = []
        for i in range(obs_main.shape[0]):
            graphs.append(Data(node_info[i], self.basegraph.paths))
        batch:Batch = Batch.from_data_list(graphs)

        convoluted = self.node_convolutions(
            batch.x, batch.edge_index, batch.batch
        ) #+ batch.x

        convoluted_by_batch = convoluted.reshape(node_info.shape)
        convoluted_by_batch[:, -1] = 0
        filtered_in_reach = torch.gather(
            convoluted_by_batch,
            1,
            in_reach_indices.unsqueeze(-1).repeat(1, 1, convoluted_by_batch.shape[-1]),
        )
        return torch.concat([filtered_in_reach.flatten(1), goal_features], dim = 1)



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
            activation=nn.LeakyReLU,
            n_convolutions=4,
            env_type="matrix",
            with_node_info=False
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
            with_node_info = self.with_node_info,
        )



        self.action_net = nn.Sequential(
            nn.Linear(self.embed_dim * 5, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            self.activation(),
            nn.Linear(
                self.embed_dim,
                6#action_space_max if action_space_max is not None else num_outputs,
            ),
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.embed_dim * 5, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
            self.activation(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            self.activation(),
            nn.Linear(self.embed_dim, 1),
        )


    def forward(self, obsdict, state, seq_lengths):
        self.obs = obsdict["obs"]
        self.features:torch.Tensor = self.fe(self.obs)
        actions = self.action_net(self.features)
        # if self.discrete_action_space:
        #     action_out = actions
        # else:
        #     action_out = torch.zeros(
        #         obsdict["obs"]["action_mask"].shape,
        #         device=actions.device,
        #     )
        #     action_out[:, 0, :] = actions
        action_out = torch.zeros(
            obsdict["obs"]["action_mask"].shape,
            device=actions.device,
        )
        if len(action_out.shape) == 3:
            action_out[:, 0, :6] = actions
        else:
            action_out[:, :6] = actions
        if self.with_action_mask:
            action_out[obsdict["obs"]["action_mask"] == 0] = -1e8
        return action_out.flatten(1), []

    def value_function(self):
        return self.value_net(self.features).flatten()
