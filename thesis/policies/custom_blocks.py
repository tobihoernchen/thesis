from torch import nn
import torch
from collections import OrderedDict


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_agents, activation=nn.ReLU) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm((n_agents, embed_dim))
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            activation(),
            nn.Linear(4 * embed_dim, embed_dim),
            activation(),
        )
        self.norm2 = nn.LayerNorm((n_agents, embed_dim))

    def forward(self, x):
        attended = self.attention(x, x, x, need_weights=False)[0]
        x = x + attended
        normed1 = self.norm1(x)
        fedforward = self.feedforward(normed1)
        x = fedforward + x
        normed2 = self.norm2(x)
        return normed2


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_agents, activation=nn.ReLU) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm((n_agents, embed_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm((n_agents, embed_dim))
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            activation(),
            nn.Linear(4 * embed_dim, embed_dim),
            activation(),
        )
        self.norm3 = nn.LayerNorm((n_agents, embed_dim))

    def forward(self, inputs, outputs, attn_mask  = None):
        self_attended = self.self_attention(
            outputs, outputs, outputs, need_weights=False
        )[0]
        x = outputs + self_attended
        normed1 = self.norm1(x)
        attended = self.attention(normed1, inputs, inputs, need_weights=False, attn_mask  = attn_mask  )[0]
        x = attended + x
        normed2 = self.norm2(x)
        fedforward = self.feedforward(normed2)
        x = fedforward + x
        normed3 = self.norm3(x)
        return normed3


class Graph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "nodes",
            torch.tensor(
                [coords for i, coords in self.init_nodes().items()] + [(0, 0)]
            ),
            persistent=False,
        )
        self.register_buffer(
            "paths",
            torch.tensor(self.init_paths() + [(len(self.nodes)-1,len(self.nodes)-1)]).t().contiguous().to(dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "indices",
            torch.arange(0, self.nodes.shape[0]), persistent=False
        )

    def init_nodes(self):
        return OrderedDict([])

    def init_paths(self):
        return []

    @torch.no_grad()
    def get_node_indices(self, nodes: torch.Tensor):
        original_shape = nodes.shape
        points = nodes.reshape(-1, 2)
        points_rep = points[:, None, :].repeat(1, len(self.nodes), 1)
        nodes_rep = self.nodes.repeat(len(points), 1, 1)
        choice = torch.all(points_rep == nodes_rep, axis=2)
        indices = self.indices.repeat(len(points), 1)[choice].reshape(
            original_shape[:-1]
        )
        return indices

    def subgraph(self, center, distance):
        nodes = self.get_node_indices(center).unsqueeze(0)
        for d in range(distance):
            new = self.paths[:,torch.isin(self.paths, nodes, assume_unique=True).any(dim=0)].flatten()
            nodes = torch.concat([nodes, new]).unique()
        paths = self.paths[:,torch.isin(self.paths, nodes, assume_unique=True).all(dim=0)]
        return nodes,paths 


class PointEmbedding(nn.Module):
    """Input:  (*, 2)
    Output: (*, embed_size)
    creates an embedding based on 2D-points
    """

    def __init__(self, max_len=200, embed_size=2) -> None:
        super().__init__()
        self.hwm = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.nodes = nn.Parameter(-torch.ones((max_len, 2)), requires_grad=False)
        self.indices = nn.Parameter(torch.arange(0, max_len), requires_grad=False)
        self.register_parameter("nodes", self.nodes)
        self.register_parameter("HWM", self.hwm)
        self.register_parameter("indices", self.indices)
        self.embedding = nn.Embedding(max_len, embed_size)

    def _add_nodes(self, new_nodes):
        assert int(self.hwm) + len(new_nodes) <= len(self.nodes)
        self.nodes[int(self.hwm) : int(self.hwm[0]) + len(new_nodes)] = new_nodes
        self.hwm += len(new_nodes)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        orig_dim = points.shape
        points = (
            points.reshape(orig_dim[:-1].numel(), 2)
            .to(dtype=torch.float)
            .round(decimals=4)
        )
        points_rep = points[:, None, :].repeat(1, len(self.nodes), 1)
        while True:
            nodes_rep = self.nodes.repeat(len(points), 1, 1)
            choice = torch.all(points_rep == nodes_rep, axis=2)
            not_anyone = torch.logical_not(torch.any(choice, axis=1))
            if not_anyone.any():
                self._add_nodes(points[not_anyone].unique(dim=0))
            else:
                break
        indices = (
            self.indices.repeat(len(points), 1)[choice].reshape(orig_dim[:-1]).detach()
        )
        return self.embedding(indices).detach()


class MatrixPositionEmbedder(nn.Module):
    """Inputs: (*, x)
    Output: (*, x + len(pos_cols) * embed_size)
    """

    def __init__(self, embed_size, resolution=200) -> None:
        super().__init__()
        self.pointembedder = PointEmbedding(max_len=resolution, embed_size=embed_size)

    def forward(self, x: torch.Tensor, pos_cols):
        pos_cols = torch.Tensor([[p, p + 1] for p in pos_cols]).to(dtype=torch.long)
        orig_dim = x.shape
        to_embed = x[..., pos_cols]
        to_embed = to_embed.reshape(-1, 2).detach()
        embedded = self.pointembedder(to_embed)
        back_in_shape = embedded.reshape(orig_dim[:-1] + (-1,))
        return torch.concat([x, back_in_shape], dim=-1)


class FourierFeatureEmbedder(nn.Module):

    def __init__(self, size = 4, sigma = 1) -> None:
        super().__init__()
        self.b = nn.Parameter(torch.randn(size, 2) * sigma, requires_grad=False)
        self.register_parameter("b", self.b)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, pos_cols):
        pos_cols = torch.Tensor([[p, p + 1] for p in pos_cols]).to(dtype=torch.long)
        orig_dim = x.shape
        to_embed = x[..., pos_cols]
        to_embed = to_embed.reshape((-1, 2)).detach()
        features = []
        for b in self.b:
            features.append(torch.sin(2*torch.pi*torch.tensordot(b, to_embed, dims=([0], [1]))).unsqueeze(1))
            features.append(torch.cos(2*torch.pi*torch.tensordot(b, to_embed, dims=([0], [1]))).unsqueeze(1))
        features_in_shape = torch.concat(features, dim=1).reshape(orig_dim[:-1] + (-1,))
        return torch.concat([x, features_in_shape], dim = -1)



class MatrixGraph(Graph):
    def __init__(self) -> None:
        super().__init__()

    def init_nodes(self):
        return OrderedDict(
            [
               (0, (0.2125984251968504, 0.37349397590361444)),
             (1, (0.1732283464566929, 0.37349397590361444)),
             (2, (0.1732283464566929, 0.4819277108433735)),
             (3, (0.2125984251968504, 0.4819277108433735)),
             (4, (0.14173228346456693, 0.43373493975903615)),
             (5, (0.1732283464566929, 0.3132530120481928)),
             (6, (0.2125984251968504, 0.3132530120481928)),
             (7, (0.2125984251968504, 0.20481927710843373)),
             (8, (0.1732283464566929, 0.20481927710843373)),
             (9, (0.14173228346456693, 0.25301204819277107)),
             (10, (0.25196850393700787, 0.4819277108433735)),
             (11, (0.3228346456692913, 0.4819277108433735)),
             (12, (0.3228346456692913, 0.5542168674698795)),
             (13, (0.25196850393700787, 0.5542168674698795)),
             (14, (0.28346456692913385, 0.6024096385542169)),
             (15, (0.3700787401574803, 0.5542168674698795)),
             (16, (0.4409448818897638, 0.5542168674698795)),
             (17, (0.4409448818897638, 0.4819277108433735)),
             (18, (0.3700787401574803, 0.4819277108433735)),
             (19, (0.4015748031496063, 0.42168674698795183)),
             (20, (0.48031496062992124, 0.4819277108433735)),
             (21, (0.5590551181102362, 0.4819277108433735)),
             (22, (0.5590551181102362, 0.5542168674698795)),
             (23, (0.48031496062992124, 0.5542168674698795)),
             (24, (0.5196850393700787, 0.6024096385542169)),
             (25, (0.5984251968503937, 0.5542168674698795)),
             (26, (0.6692913385826772, 0.5542168674698795)),
             (27, (0.6692913385826772, 0.4819277108433735)),
             (28, (0.5984251968503937, 0.4819277108433735)),
             (29, (0.6377952755905512, 0.42168674698795183)),
             (30, (0.7086614173228346, 0.4819277108433735)),
             (31, (0.7795275590551181, 0.4819277108433735)),
             (32, (0.7795275590551181, 0.5542168674698795)),
             (33, (0.7086614173228346, 0.5542168674698795)),
             (34, (0.7401574803149606, 0.6024096385542169)),
             (35, (0.8188976377952756, 0.5542168674698795)),
             (36, (0.8976377952755905, 0.5542168674698795)),
             (37, (0.8976377952755905, 0.4819277108433735)),
             (38, (0.8188976377952756, 0.4819277108433735)),
             (39, (0.8582677165354331, 0.6024096385542169)),
             (40, (0.7480314960629921, 0.42168674698795183)),
             (41, (0.6299212598425197, 0.6024096385542169)),
             (42, (0.5196850393700787, 0.42168674698795183)),
             (43, (0.4015748031496063, 0.6024096385542169)),
             (44, (0.28346456692913385, 0.42168674698795183)),
             (45, (0.2125984251968504, 0.5542168674698795)),
             (46, (0.2125984251968504, 0.6626506024096386)),
             (47, (0.1732283464566929, 0.6626506024096386)),
             (48, (0.1732283464566929, 0.5542168674698795)),
             (49, (0.14173228346456693, 0.6024096385542169)),
             (50, (0.2125984251968504, 0.7228915662650602)),
             (51, (0.2125984251968504, 0.8313253012048193)),
             (52, (0.1732283464566929, 0.8313253012048193)),
             (53, (0.1732283464566929, 0.7228915662650602)),
             (54, (0.14173228346456693, 0.7831325301204819)),
             (55, (0.1732283464566929, 0.14457831325301204)),
             (56, (0.2125984251968504, 0.14457831325301204)),
             (57, (0.2125984251968504, 0.024096385542168676)),
             (58, (0.1732283464566929, 0.024096385542168676)),
             (59, (0.14173228346456693, 0.08433734939759036)),
             (60, (0.937007874015748, 0.4819277108433735)),
             (61, (0.9763779527559056, 0.4819277108433735)),
             (62, (0.9763779527559056, 0.37349397590361444)),
             (63, (0.937007874015748, 0.37349397590361444)),
             (64, (0.937007874015748, 0.3132530120481928)),
             (65, (0.9763779527559056, 0.3132530120481928)),
             (66, (0.9763779527559056, 0.1927710843373494)),
             (67, (0.937007874015748, 0.1927710843373494)),
             (68, (0.937007874015748, 0.13253012048192772)),
             (69, (0.9763779527559056, 0.13253012048192772)),
             (70, (0.9763779527559056, 0.024096385542168676)),
             (71, (0.937007874015748, 0.024096385542168676)),
             (72, (1.0, 0.07228915662650602)),
             (73, (1.0, 0.25301204819277107)),
             (74, (1.0, 0.43373493975903615)),
             (75, (0.937007874015748, 0.5542168674698795)),
             (76, (0.9763779527559056, 0.5542168674698795)),
             (77, (0.9763779527559056, 0.6506024096385542)),
             (78, (0.937007874015748, 0.6506024096385542)),
             (79, (1.0, 0.6024096385542169)),
             (80, (0.9763779527559056, 0.7108433734939759)),
             (81, (0.9763779527559056, 0.8313253012048193)),
             (82, (0.937007874015748, 0.8313253012048193)),
             (83, (0.937007874015748, 0.7108433734939759)),
             (84, (0.937007874015748, 0.891566265060241)),
             (85, (0.937007874015748, 1.0)),
             (86, (0.9763779527559056, 1.0)),
             (87, (0.9763779527559056, 0.891566265060241)),
             (88, (1.0, 0.7710843373493976)),
             (89, (1.0, 0.9518072289156626)),
             (90, (0.8188976377952756, 0.42168674698795183)),
             (91, (0.8976377952755905, 0.42168674698795183)),
             (92, (0.8976377952755905, 0.3614457831325301)),
             (93, (0.8976377952755905, 0.30120481927710846)),
             (94, (0.8976377952755905, 0.24096385542168675)),
             (95, (0.8976377952755905, 0.18072289156626506)),
             (96, (0.8188976377952756, 0.3614457831325301)),
             (97, (0.8188976377952756, 0.30120481927710846)),
             (98, (0.8188976377952756, 0.24096385542168675)),
             (99, (0.8188976377952756, 0.18072289156626506)),
             (100, (0.8582677165354331, 0.18072289156626506)),
             (101, (0.8582677165354331, 0.24096385542168675)),
             (102, (0.8582677165354331, 0.30120481927710846)),
             (103, (0.8582677165354331, 0.3614457831325301)),
             (104, (0.8582677165354331, 0.42168674698795183))
            ]
        )

    def init_paths(self):
        return [
            (1, 0),
 (0, 1),
 (1, 2),
 (2, 1),
 (2, 3),
 (3, 2),
 (0, 3),
 (3, 0),
 (4, 2),
 (2, 4),
 (4, 1),
 (1, 4),
 (5, 1),
 (1, 5),
 (5, 6),
 (6, 5),
 (6, 0),
 (0, 6),
 (7, 6),
 (6, 7),
 (8, 7),
 (7, 8),
 (8, 5),
 (5, 8),
 (9, 8),
 (8, 9),
 (5, 9),
 (9, 5),
 (10, 3),
 (3, 10),
 (10, 11),
 (11, 10),
 (11, 12),
 (12, 11),
 (13, 12),
 (12, 13),
 (10, 13),
 (13, 10),
 (13, 14),
 (14, 13),
 (12, 14),
 (14, 12),
 (12, 15),
 (15, 12),
 (15, 16),
 (16, 15),
 (17, 16),
 (16, 17),
 (18, 17),
 (17, 18),
 (18, 11),
 (11, 18),
 (18, 15),
 (15, 18),
 (18, 19),
 (19, 18),
 (19, 17),
 (17, 19),
 (17, 20),
 (20, 17),
 (20, 21),
 (21, 20),
 (21, 22),
 (22, 21),
 (23, 22),
 (22, 23),
 (16, 23),
 (23, 16),
 (20, 23),
 (23, 20),
 (23, 24),
 (24, 23),
 (22, 24),
 (24, 22),
 (22, 25),
 (25, 22),
 (25, 26),
 (26, 25),
 (27, 26),
 (26, 27),
 (28, 27),
 (27, 28),
 (21, 28),
 (28, 21),
 (28, 25),
 (25, 28),
 (28, 29),
 (29, 28),
 (29, 27),
 (27, 29),
 (27, 30),
 (30, 27),
 (30, 31),
 (31, 30),
 (31, 32),
 (32, 31),
 (33, 32),
 (32, 33),
 (26, 33),
 (33, 26),
 (30, 33),
 (33, 30),
 (33, 34),
 (34, 33),
 (32, 34),
 (34, 32),
 (32, 35),
 (35, 32),
 (35, 36),
 (36, 35),
 (37, 36),
 (36, 37),
 (38, 37),
 (37, 38),
 (31, 38),
 (38, 31),
 (38, 35),
 (35, 38),
 (36, 39),
 (39, 36),
 (35, 39),
 (39, 35),
 (40, 31),
 (31, 40),
 (30, 40),
 (40, 30),
 (26, 41),
 (41, 26),
 (25, 41),
 (41, 25),
 (42, 21),
 (21, 42),
 (20, 42),
 (42, 20),
 (16, 43),
 (43, 16),
 (15, 43),
 (43, 15),
 (44, 11),
 (11, 44),
 (10, 44),
 (44, 10),
 (45, 13),
 (13, 45),
 (3, 45),
 (45, 3),
 (45, 46),
 (46, 45),
 (47, 46),
 (46, 47),
 (48, 47),
 (47, 48),
 (48, 45),
 (45, 48),
 (49, 48),
 (48, 49),
 (47, 49),
 (49, 47),
 (2, 48),
 (48, 2),
 (46, 50),
 (50, 46),
 (50, 51),
 (51, 50),
 (52, 51),
 (51, 52),
 (53, 52),
 (52, 53),
 (53, 50),
 (50, 53),
 (54, 53),
 (53, 54),
 (52, 54),
 (54, 52),
 (47, 53),
 (53, 47),
 (8, 55),
 (55, 8),
 (55, 56),
 (56, 55),
 (57, 56),
 (56, 57),
 (58, 57),
 (57, 58),
 (58, 55),
 (55, 58),
 (59, 58),
 (58, 59),
 (55, 59),
 (59, 55),
 (7, 56),
 (56, 7),
 (36, 75),
 (75, 36),
 (60, 75),
 (75, 60),
 (60, 61),
 (61, 60),
 (62, 61),
 (61, 62),
 (63, 62),
 (62, 63),
 (64, 63),
 (63, 64),
 (64, 65),
 (65, 64),
 (66, 65),
 (65, 66),
 (67, 66),
 (66, 67),
 (68, 67),
 (67, 68),
 (68, 69),
 (69, 68),
 (70, 69),
 (69, 70),
 (71, 70),
 (70, 71),
 (71, 68),
 (68, 71),
 (70, 72),
 (72, 70),
 (69, 72),
 (72, 69),
 (69, 66),
 (66, 69),
 (67, 64),
 (64, 67),
 (66, 73),
 (73, 66),
 (65, 73),
 (73, 65),
 (65, 62),
 (62, 65),
 (63, 60),
 (60, 63),
 (62, 74),
 (74, 62),
 (61, 74),
 (74, 61),
 (61, 76),
 (76, 61),
 (37, 60),
 (60, 37),
 (75, 76),
 (76, 75),
 (76, 77),
 (77, 76),
 (78, 77),
 (77, 78),
 (75, 78),
 (78, 75),
 (76, 79),
 (79, 76),
 (77, 79),
 (79, 77),
 (77, 80),
 (80, 77),
 (80, 81),
 (81, 80),
 (82, 81),
 (81, 82),
 (83, 82),
 (82, 83),
 (83, 80),
 (80, 83),
 (83, 78),
 (78, 83),
 (82, 84),
 (84, 82),
 (84, 85),
 (85, 84),
 (85, 86),
 (86, 85),
 (87, 86),
 (86, 87),
 (84, 87),
 (87, 84),
 (81, 87),
 (87, 81),
 (80, 88),
 (88, 80),
 (81, 88),
 (88, 81),
 (87, 89),
 (89, 87),
 (86, 89),
 (89, 86),
 (38, 90),
 (90, 38),
 (37, 91),
 (91, 37),
 (92, 91),
 (91, 92),
 (93, 92),
 (92, 93),
 (94, 93),
 (93, 94),
 (95, 94),
 (94, 95),
 (96, 90),
 (90, 96),
 (97, 96),
 (96, 97),
 (98, 97),
 (97, 98),
 (99, 98),
 (98, 99),
 (99, 100),
 (100, 99),
 (100, 95),
 (95, 100),
 (98, 101),
 (101, 98),
 (101, 94),
 (94, 101),
 (97, 102),
 (102, 97),
 (96, 103),
 (103, 96),
 (90, 104),
 (104, 90),
 (102, 93),
 (93, 102),
 (103, 92),
 (92, 103),
 (104, 91),
 (91, 104)
        ]


class MiniMatrixGraph(Graph):
    def __init__(self) -> None:
        super().__init__()

    def init_nodes(self):
        return OrderedDict(
            [
                (0, (0.5454545454545454, 0.76)),
                (1, (0.6022727272727273, 0.76)),
                (2, (0.5454545454545454, 0.86)),
                (3, (0.6022727272727273, 0.86)),
                (4, (0.4772727272727273, 0.76)),
                (5, (0.42045454545454547, 0.76)),
                (6, (0.42045454545454547, 0.86)),
                (7, (0.4772727272727273, 0.86)),
                (8, (0.32954545454545453, 0.808)),
                (9, (0.42045454545454547, 0.48)),
                (10, (0.4772727272727273, 0.48)),
                (11, (0.4772727272727273, 0.38)),
                (12, (0.42045454545454547, 0.38)),
                (13, (0.32954545454545453, 0.428)),
                (14, (0.5727272727272728, 0.62)),
                (15, (0.7613636363636364, 0.76)),
                (16, (0.8181818181818182, 0.76)),
                (17, (0.8181818181818182, 0.86)),
                (18, (0.7613636363636364, 0.86)),
                (19, (0.7909090909090909, 0.62)),
                (20, (0.9431818181818182, 0.76)),
                (21, (1.0, 0.76)),
                (22, (1.0, 0.86)),
                (23, (0.9431818181818182, 0.86)),
                (24, (0.9727272727272728, 0.62)),
                (25, (0.9727272727272728, 1.0)),
            ]
        )

    def init_paths(self):
        return [
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (2, 3),
            (3, 2),
            (1, 3),
            (3, 1),
            (4, 0),
            (0, 4),
            (5, 4),
            (4, 5),
            (5, 6),
            (6, 5),
            (6, 7),
            (7, 6),
            (4, 7),
            (7, 4),
            (8, 6),
            (6, 8),
            (8, 5),
            (5, 8),
            (5, 9),
            (9, 5),
            (9, 10),
            (10, 9),
            (10, 4),
            (4, 10),
            (11, 10),
            (10, 11),
            (12, 11),
            (11, 12),
            (12, 9),
            (9, 12),
            (13, 12),
            (12, 13),
            (13, 9),
            (9, 13),
            (2, 7),
            (7, 2),
            (0, 14),
            (14, 0),
            (14, 1),
            (1, 14),
            (1, 15),
            (15, 1),
            (15, 16),
            (16, 15),
            (16, 17),
            (17, 16),
            (18, 17),
            (17, 18),
            (15, 18),
            (18, 15),
            (18, 3),
            (3, 18),
            (15, 19),
            (19, 15),
            (19, 16),
            (16, 19),
            (16, 20),
            (20, 16),
            (20, 21),
            (21, 20),
            (21, 22),
            (22, 21),
            (23, 22),
            (22, 23),
            (20, 23),
            (23, 20),
            (23, 17),
            (17, 23),
            (24, 20),
            (20, 24),
            (24, 21),
            (21, 24),
            (25, 23),
            (23, 25),
            (22, 25),
            (25, 22),
        ]
