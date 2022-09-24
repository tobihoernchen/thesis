from torch import nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from torch import nn
from ray.rllib.models.torch.modules.gru_gate import GRUGate


class AttentionBlock(nn.Module):
    """
    Applies regular self attention and a feedforward-layer to an input shaped [batch_size, n_agents, embed_dim]
    """

    def __init__(self, embed_dim=64, n_heads=8, fn_mask=None, n_agents=10):
        super().__init__()
        self.fn_mask = fn_mask
        self.n_heads = n_heads

        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm((n_agents, embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.norm2 = nn.LayerNorm((n_agents, embed_dim))
        self.fn_mask = fn_mask

    def forward(self, x):
        if self.fn_mask is not None:
            mask = self.fn_mask()
        else:
            mask = None
        attended = self.attention(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=mask,
        )[0]

        x = attended + x
        toAttend = self.norm1(x)
        fedforward = self.ff(toAttend)
        x = fedforward + x
        x = self.norm2(x)
        return x


class MultiAgentFE(nn.Module):
    """
    Self-attention-based feature extractor for Stabel Baselines 3 (Will also be used within RLLIB custom models).
    For each agent, the feature vektor is first embedded (resulting shape [batch_size, n_agents, embed_dim])

    Inputs:
    observation_space: gym.spaces.Space. Should be either Box or Dict with "observation"-key, which os the expected to be a Box space.
    n_agents: number of agents. It is assumed that each agent is provided len(observation_space) // n_agents features.
    n_features: optional. Can be overwritten if this class will not provide n_agents * embed_dim features.
    embed_dim: size of the desired feature vector per agent for self attention.
    n_heads: number of heads for self attention
    depth: number of self attention blocks
    use_attention_mask: Agents can be "turned off" by switching the first entry of the feature vector to 0.
        The rest of the feature vector will not be attended to by other agents.

    """

    def __init__(
        self,
        observation_space: spaces.Space,
        n_agents,
        n_features=None,
        embed_dim=64,
        n_heads=8,
        depth=8,
        use_attention_mask=False,
    ):
        n_features = embed_dim * n_agents if n_features is None else n_features
        super().__init__()  # observation_space, n_features)
        self.embed_dim = embed_dim
        if isinstance(observation_space, spaces.Box):
            self.n_obs_per_agv = observation_space.shape[0] // n_agents
        elif isinstance(observation_space, spaces.Dict):
            self.n_obs_per_agv = (
                observation_space.spaces["observation"].shape[0] // n_agents
            )
        self.n_agents = n_agents
        self.use_attention_mask = use_attention_mask
        self.mask = None
        self.n_heads = n_heads
        self.aye = None

        self.embedd = nn.Sequential(
            nn.Linear(self.n_obs_per_agv, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
        )

        self.cross_agent_embedd = nn.Sequential(
            nn.Linear(n_agents, n_agents * 4),
            nn.ReLU(),
            nn.Linear(n_agents * 4, n_agents * 4),
            nn.ReLU(),
            nn.Linear(n_agents * 4, n_agents),
            nn.ReLU(),
        )

        if isinstance(self.use_attention_mask, torch.Tensor):
            fn_mask = self.get_mask_constant
        elif self.use_attention_mask == False:
            fn_mask = None
        elif self.use_attention_mask == True:
            fn_mask = self.get_mask

        ablocks = []
        for i in range(depth):
            ablocks.append(
                AttentionBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    n_agents=n_agents,
                    fn_mask=fn_mask,
                )
            )
        self.ablocks = nn.Sequential(*ablocks)

    def get_mask(self):
        m = self.mask
        return (m.repeat(1, 1, m.shape[1]) + self.aye == 0).repeat_interleave(
            self.n_heads, dim=0
        )

    def get_mask_constant(self):
        if self.use_attention_mask.device != self.mask.device:
            self.use_attention_mask = self.use_attention_mask.to(
                device=self.mask.device
            )
        return self.use_attention_mask

    def forward(self, x: torch.Tensor):
        """
        Resulting shape is [batch_size, n_agents, embed_dim]
        """

        if isinstance(x, dict):
            x = x["observation"]
        reshaped = x.view(x.shape[:-1] + (self.n_agents, self.n_obs_per_agv))
        self.mask = reshaped[:, :, 1, None]
        if self.aye is None:
            self.aye = torch.eye(self.mask.shape[1], device=self.mask.device)

        reshaped = self.cross_agent_embedd(reshaped.transpose(1, 2)).transpose(1, 2)
        embeddings = self.embedd(reshaped)
        x = self.ablocks(embeddings)
        return x


class MultiAgentFE_offPolicy(MultiAgentFE):
    def __init__(
        self,
        observation_space: spaces.Box,
        n_agents,
        n_features=None,
        embed_dim=64,
        n_heads=8,
        depth=6,
        use_attention_mask=False,
    ):
        super().__init__(
            observation_space,
            n_agents,
            n_features,
            embed_dim,
            n_heads,
            depth,
            use_attention_mask=use_attention_mask,
        )

    def forward(self, x: torch.Tensor):
        """
        Resulting shape is [batch_size, embed_dim]
        """
        x = super().forward(x)
        x = x.max(dim=1)[0]
        return x
