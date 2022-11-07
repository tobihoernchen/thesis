from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical
import torch
import numpy as np
import gym.spaces
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog


def register_ma_action_dist():
    ModelCatalog.register_custom_action_dist(
        "MAActionDistribution", MAActionDistribution
    )


class MAActionDistribution(TorchMultiCategorical):
    def __init__(self, inputs, model, action_space=None):
        super().__init__(inputs, model, [22] * 30, action_space)

    @override(TorchMultiCategorical)
    def logp(self, actions: torch.Tensor) -> torch.Tensor:
        # # If tensor is provided, unstack it into list.
        if isinstance(actions, torch.Tensor):
            if isinstance(self.action_space, gym.spaces.Box):
                actions = torch.reshape(
                    actions, [-1, int(np.prod(self.action_space.shape))]
                )
            actions = torch.unbind(actions, dim=1)
        logps = torch.stack([cat.log_prob(act) for cat, act in zip(self.cats, actions)])
        return logps[0]

    @override(TorchMultiCategorical)
    def entropy(self) -> torch.Tensor:
        return self.multi_entropy()[:, 0]

    @override(TorchMultiCategorical)
    def kl(self, other) -> torch.Tensor:
        return self.multi_kl(other)[:, 0]
