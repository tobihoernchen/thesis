from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from typing import Dict, Optional, TYPE_CHECKING
from ray.rllib.policy.policy import Policy

from ray.rllib.evaluation.episode import Episode
from ray.rllib.utils.typing import PolicyID
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.typing import  EnvType

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker

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
    

    def on_sub_environment_created(
        self,
        *,
        worker: "RolloutWorker",
        sub_environment: EnvType,
        env_context: EnvContext,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        if not "client" in env_context.keys():
            env_context["client"] = sub_environment.env.client
        return super().on_sub_environment_created(
            worker=worker,
            sub_environment = sub_environment,
            env_context = env_context,
            env_index = env_index,
            **kwargs,
        )
