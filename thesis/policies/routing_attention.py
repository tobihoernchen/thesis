from torch import nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from torch import nn
from ray.rllib.models.torch.modules.gru_gate import GRUGate


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, n_heads=8, fn_mask=None, fleetsize = 10):
        super().__init__()
        self.fn_mask = fn_mask
        self.n_heads = n_heads

        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm((fleetsize, embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.norm2 = nn.LayerNorm((fleetsize, embed_dim))
        self.fn_mask = fn_mask
        # self.gruIn = GRUGate(embed_dim)
        # self.gruOut = GRUGate(embed_dim)

    def forward(self, x):
        if self.fn_mask is not None:
            mask = self.fn_mask()
            mask = mask.repeat_interleave(self.n_heads, dim=0)
        else:
            mask = None
        attended = self.attention(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=mask,
        )[0]

        # x = self.gruIn([x, attended])
        x = attended + x
        toAttend = self.norm1(x)
        fedforward = self.ff(toAttend)
        # x = self.gruOut([fedforward, x])
        x = fedforward + x
        x = self.norm2(x)
        return x


class RoutingFE(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        max_fleetsize,
        n_features=None,
        embed_dim=64,
        n_heads=8,
        depth=8,
    ):
        n_features = embed_dim * max_fleetsize if n_features is None else n_features
        super().__init__(observation_space, n_features)
        self.embed_dim = embed_dim
        if isinstance(observation_space, spaces.Box):
            self.n_obs_per_agv = observation_space.shape[0] // max_fleetsize
        elif isinstance(observation_space, spaces.Dict):
            self.n_obs_per_agv = (
                observation_space.spaces["observation"].shape[0] // max_fleetsize
            )
        self.max_fleetsize = max_fleetsize
        self.mask = None
        self.aye = None

        self.embedd = nn.Sequential(
            nn.Linear(self.n_obs_per_agv, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
        )

        ablocks = []
        for i in range(depth):
            ablocks.append(
                AttentionBlock(
                    embed_dim=embed_dim, n_heads=n_heads, fleetsize=max_fleetsize #, fn_mask=self.get_mask
                )
            )
        self.ablocks = nn.Sequential(*ablocks)

    def get_mask(self):
        m = self.mask
        # return torch.bmm(m, m.transpose(1, 2)).detach()
        return m.repeat(1, 1, m.shape[1]) + self.aye == 0

    def forward(self, x: torch.Tensor):
        if isinstance(x, dict):
            x = x["observation"]
        reshaped = x.view(x.shape[:-1] + (self.max_fleetsize, self.n_obs_per_agv))
        self.mask = reshaped[:, :, 1, None]
        if self.aye is None:
            self.aye = torch.eye(self.mask.shape[1], device=self.mask.device)
        # self.actionmask = reshaped[:, :, 1, None]
        # reshaped = reshaped.masked_fill(self.mask.repeat(1, 1, self.n_obs_per_agv) == 0, 0)
        x = self.embedd(reshaped)
        x = self.ablocks(x)
        # x = x.masked_fill(self.actionmask.repeat(1, 1, self.embed_dim) == 0, 0)
        return x


class RoutingFE_offPolicy(RoutingFE):
    def __init__(
        self,
        observation_space: spaces.Box,
        max_fleetsize,
        n_features=None,
        embed_dim=64,
        n_heads=8,
        depth=6,
    ):
        super().__init__(
            observation_space, max_fleetsize, embed_dim, embed_dim, n_heads, depth
        )

    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        x = x.max(dim=1)[0]
        return x
