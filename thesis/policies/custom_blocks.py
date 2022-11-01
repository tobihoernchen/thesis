from torch import nn
import torch

class PointMapping:
    def __init__(self, nodes: dict) -> None:
        self.nodes = torch.tensor(list(nodes.values()), dtype=float, device="cuda")
        self.indices = torch.arange(0, len(self.nodes) + 1, device="cuda")

    def __call__(self, points):
        points = points.to(dtype=float)
        nodes_rep = self.nodes.repeat(len(points), 1, 1)
        points_rep = points[:, None, :].repeat(1, len(self.nodes), 1)
        choice = torch.all(torch.isclose(points_rep, nodes_rep, atol=0.01), axis=2)
        anyone = torch.logical_not(torch.any(choice, axis=1))
        choice = torch.concat([anyone.unsqueeze(1), choice], axis=1)
        return self.indices.repeat(len(points), 1)[choice]


class Embedder(nn.Module):
    def __init__(self, embed_dim, original_dim, activation) -> None:
        super().__init__()
        self.original_dim = original_dim
        self.embedd = nn.Sequential(
            nn.Linear(original_dim, embed_dim * 4),
            activation(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
            activation(),
            nn.Linear(embed_dim * 4, embed_dim),
            activation(),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(x.shape[0], x.shape[1] // self.original_dim, self.original_dim)
        return self.embedd(x)

class PositionEncoder(nn.Module):
    def __init__(
        self, pos_cols, original_dim, embed_size, resolution=100, mask_valid=None
    ) -> None:
        super().__init__()
        self.original_dim = original_dim
        self.pointmapper = PointMapping(
            {
                0: (0.5454545454545454, 0.76),
                1: (0.6022727272727273, 0.76),
                2: (0.5454545454545454, 0.86),
                3: (0.6022727272727273, 0.86),
                4: (0.4772727272727273, 0.76),
                5: (0.42045454545454547, 0.76),
                6: (0.42045454545454547, 0.86),
                7: (0.4772727272727273, 0.86),
                8: (0.32954545454545453, 0.808),
                9: (0.42045454545454547, 0.48),
                10: (0.4772727272727273, 0.48),
                11: (0.4772727272727273, 0.38),
                12: (0.42045454545454547, 0.38),
                13: (0.32954545454545453, 0.428),
                14: (0.5727272727272728, 0.62),
                15: (0.7613636363636364, 0.76),
                16: (0.8181818181818182, 0.76),
                17: (0.8181818181818182, 0.86),
                18: (0.7613636363636364, 0.86),
                19: (0.7909090909090909, 0.62),
                20: (0.9431818181818182, 0.76),
                21: (1.0, 0.76),
                22: (1.0, 0.86),
                23: (0.9431818181818182, 0.86),
                24: (0.9727272727272728, 0.62),
                25: (0.9727272727272728, 1.0),
            }
        )
        self.embedding = nn.Embedding(resolution, embed_size)
        self.pos_cols = pos_cols
        self.resolution = resolution
        self.embed_size = embed_size

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 2:
            x = x.view(x.shape[0], x.shape[1] // self.original_dim, self.original_dim)
        blocks = []
        last_col = None
        for col in self.pos_cols:
            if last_col is None:
                blocks.append(x[:, :, : col + 2])
            else:
                blocks.append(x[:, :, last_col + 2 : col + 2])
            last_col = col
            to_encode = x[:, :, col : col + 2]
            ints = self.pointmapper(to_encode.flatten(end_dim=1))
            encoded = self.embedding(ints.to(dtype=int)).reshape(
                to_encode.shape[:-1] + (self.embed_size,)
            )
            blocks.append(encoded)
        blocks.append(x[:, :, col + 2 :])
        out = torch.concat(blocks, 2)
        return out

    def out_features(self):
        return self.original_dim + len(self.pos_cols) * (self.embed_size)
