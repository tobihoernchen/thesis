from torch import nn
import torch
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn
from gym import spaces


class CustomValueNet(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Lin = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = x.max(dim=1)[0]
        return self.Lin(x)


class CustomActionNet(nn.Module):
    def __init__(self, embed_dim, multi=True):
        super().__init__()
        self.Lin = nn.Sequential(
            nn.Linear(embed_dim, 5),
            nn.Softmax(-1),
            nn.Flatten(1),
        )
        self.multi = multi

    def forward(self, x: torch.Tensor):
        if not self.multi:
            x = x.max(dim=1)[0]
        return self.Lin(x)


class AttentionACPolicy(ActorCriticPolicy):
    """
    Change AttentionACPolicy Behavior from [batch_size, n_features]->[batch_size, n_actions] to [batch_size, n_agents, n_features]->[batch_size, n_agents, action]
    (Each agent's action does not depend on the other agents, but only on its own feature vector)
    """

    def _build(self, lr_schedule: Schedule) -> None:
        ret = super()._build(lr_schedule)
        self.action_net = CustomActionNet(
            self.features_extractor.embed_dim,
            isinstance(self.action_space, spaces.MultiDiscrete),
        )
        self.value_net = CustomValueNet(self.features_extractor.embed_dim)
        return ret


class MaskableAttentionACPolicy(MaskableActorCriticPolicy):
    def _build(self, lr_schedule: Schedule) -> None:
        ret = super()._build(lr_schedule)
        self.action_net = CustomActionNet(
            self.features_extractor.embed_dim,
            isinstance(self.action_space, spaces.MultiDiscrete),
        )
        self.value_net = CustomValueNet(self.features_extractor.embed_dim)
        return ret
