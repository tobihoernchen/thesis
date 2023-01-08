from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from typing import Dict
from ray.rllib.policy.policy import Policy

from ray.rllib.evaluation.episode import Episode
from ray.rllib.utils.typing import PolicyID


class CustomCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs
    ) -> None:
        episode.custom_metrics.update(base_env.get_sub_environments()[episode.env_id].env.statistics)
        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs
        )
