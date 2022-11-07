from torch import nn
import torch


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
        points = points.reshape(orig_dim[:-1].numel(), 2).to(dtype=torch.float).detach()
        points_rep = points[:, None, :].repeat(1, len(self.nodes), 1)
        while True:
            nodes_rep = self.nodes.repeat(len(points), 1, 1)
            choice = torch.all(points_rep == nodes_rep, axis=2)
            not_anyone = torch.logical_not(torch.any(choice, axis=1))
            if not_anyone.any():
                self._add_nodes(points[not_anyone].unique(dim=0))
            else:
                break
        indices = self.indices.repeat(len(points), 1)[choice].reshape(orig_dim[:-1])
        return self.embedding(indices).detach()


class MatrixPositionEncoder(nn.Module):
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
        to_embed = to_embed.reshape(int(to_embed.numel() / 2), 2)
        embedded = self.pointembedder(to_embed)
        back_in_shape = embedded.reshape(
            orig_dim[:-1] + (int(embedded.shape.numel() / orig_dim[:-1].numel()),)
        )
        return torch.concat([x, back_in_shape], dim=-1)
