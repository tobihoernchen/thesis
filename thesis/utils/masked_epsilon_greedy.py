from typing import Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.exploration.epsilon_greedy import EpsilonGreedy
from ray.rllib.utils.exploration.exploration import TensorType
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
import torch
import tree
import random


class MaskedEpsilonGreedy(EpsilonGreedy):
    def get_exploration_action(self, *, action_distribution: ActionDistribution, timestep: Union[int, TensorType], explore: Optional[Union[bool, TensorType]] = True):
        return super().get_exploration_action(action_distribution=action_distribution, timestep=timestep, explore=explore)
    
    def _get_torch_exploration_action(
        self,
        action_distribution: ActionDistribution,
        explore: bool,
        timestep: Union[int, TensorType],
    ) -> "torch.Tensor":
        """Torch method to produce an epsilon exploration action.

        Args:
            action_distribution: The instantiated
                ActionDistribution object to work with when creating
                exploration actions.

        Returns:
            The exploration-action.
        """
        q_values = action_distribution.inputs
        self.last_timestep = timestep
        exploit_action = action_distribution.deterministic_sample()
        batch_size = q_values.size()[0]
        action_logp = torch.zeros(batch_size, dtype=torch.float)

        # Explore.
        if explore:
            # Get the current epsilon.
            epsilon = self.epsilon_schedule(self.last_timestep)
            if isinstance(action_distribution, TorchMultiActionDistribution):
                exploit_action = tree.flatten(exploit_action)
                for i in range(batch_size):
                    if random.random() < epsilon:
                        # TODO: (bcahlit) Mask out actions
                        random_action = tree.flatten(self.action_space.sample())
                        for j in range(len(exploit_action)):
                            exploit_action[j][i] = torch.tensor(random_action[j])
                exploit_action = tree.unflatten_as(
                    action_distribution.action_space_struct, exploit_action
                )

                return exploit_action, action_logp

            else:
                # Mask out actions, whose Q-values are -inf, so that we don't
                # even consider them for exploration.
                random_valid_action_logits = torch.where(
                    q_values <= -1e7,
                    torch.ones_like(q_values) * 0.0,
                    torch.ones_like(q_values),
                )
                # A random action.
                random_actions = torch.squeeze(
                    torch.multinomial(random_valid_action_logits, 1), axis=1
                )

                # Pick either random or greedy.
                action = torch.where(
                    torch.empty((batch_size,)).uniform_().to(self.device) < epsilon,
                    random_actions,
                    exploit_action,
                )

                return action, action_logp
        # Return the deterministic "sample" (argmax) over the logits.
        else:
            return exploit_action, action_logp