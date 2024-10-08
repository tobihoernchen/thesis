"""Example of using a custom training workflow.
Here we create a number of CartPole agents, some of which are trained with
DQN, and some of which are trained with PPO. Both are executed concurrently
via a custom training workflow.
"""

import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
    MultiAgentReplayBuffer,
)
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_TARGET_UPDATES,
    LAST_TARGET_UPDATE_TS,
)
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import register_env

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--mixed-torch-tf", action="store_true")
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=600, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=200000, help="Number of timesteps to train."
)
# 600.0 = 4 (num_agents) x 150.0
parser.add_argument(
    "--stop-reward", type=float, default=600.0, help="Reward at which we stop training."
)


# Define new Algorithm with custom `training_step()` method (training workflow).
class DoubleTrainer(Algorithm):
    @override(Algorithm)
    def setup(self, config):
        # Call super's `setup` to create rollout workers.
        super().setup(config)
        # Create local replay buffer.
        self.local_replay_buffer = MultiAgentReplayBuffer(num_shards=1, capacity=50000)

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        # Generate common experiences, collect batch for PPO, store every (DQN) batch
        # into replay buffer.
        num_env_steps = 0

        # PPO batch size fixed at 200.
        # TODO: Use `max_env_steps=200` option of synchronous_parallel_sample instead.
        if self._by_agent_steps:
            ma_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.get_policy("dispatcher").config["train_batch_size"] * 4
            )
        else:
            ma_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.get_policy("dispatcher").config["train_batch_size"] * 4
            )
        # while num_env_steps < 200:
        #     ma_batches = synchronous_parallel_sample(
        #         worker_set=self.workers, concat=False
        #     )
            # Loop through ma-batches (which were collected in parallel).
            #for ma_batch in ma_batches:
                # Update sampled counters.
        self._counters[NUM_ENV_STEPS_SAMPLED] += ma_batch.count
        self._counters[NUM_AGENT_STEPS_SAMPLED] += ma_batch.agent_steps()
        if "dispatcher" in ma_batch.policy_batches.keys():
            ppo_batch = ma_batch.policy_batches.pop("dispatcher")  
            num_env_steps += ppo_batch.count
        else:
            ppo_batch = None
        # Add collected batches (only for DQN policy) to replay buffer.
        self.local_replay_buffer.add(ma_batch)

        # DQN sub-flow.
        dqn_train_results = {}
        # Start updating DQN policy once we have some samples in the buffer.
        cur_ts = self._counters[
            NUM_AGENT_STEPS_SAMPLED if self._by_agent_steps else NUM_ENV_STEPS_SAMPLED
        ]
        if cur_ts > self.get_policy("agv").config["num_steps_sampled_before_learning_starts"]:
            # Update DQN policy n times while updating PPO policy once.
            for _ in range(10):
                dqn_train_batch = self.local_replay_buffer.sample(self.get_policy("agv").config["train_batch_size"])
                dqn_train_results = train_one_step(
                    self, dqn_train_batch, ["agv"]
                )
                self._counters[
                    "agent_steps_trained_DQN"
                ] += dqn_train_batch.agent_steps()
                # print(
                #     "DQN policy learning on samples from",
                #     "agent steps trained",
                #     dqn_train_batch.agent_steps(),
                # )
        # Update DQN's target net every n train steps (determined by the DQN config).
        if (
            self._counters["agent_steps_trained_DQN"]
            - self._counters[LAST_TARGET_UPDATE_TS]
            >= self.get_policy("agv").config["target_network_update_freq"]
        ):
            self.workers.local_worker().get_policy("agv").update_target()
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = self._counters[
                "agent_steps_trained_DQN"
            ]

        # PPO sub-flow.
        if ppo_batch:
            ppo_train_batch = ppo_batch
            self._counters["agent_steps_trained_PPO"] += ppo_train_batch.agent_steps()
            # Standardize advantages.
            ppo_train_batch[Postprocessing.ADVANTAGES] = standardized(
                ppo_train_batch[Postprocessing.ADVANTAGES]
            )
            # print(
            #     "PPO policy learning on samples from",
            #     "agent steps trained",
            #     ppo_train_batch.agent_steps(),
            # )
            ppo_train_batch = MultiAgentBatch(
                {"dispatcher": ppo_train_batch}, ppo_train_batch.count
            )
            ppo_train_results = train_one_step(self, ppo_train_batch, ["dispatcher"])
        else:
            ppo_train_results = {}
        # Combine results for PPO and DQN into one results dict.
        results = dict(ppo_train_results, **dqn_train_results)
        return results



# Define new Algorithm with custom `training_step()` method (training workflow).
class TripleTrainer(Algorithm):
    @override(Algorithm)
    def setup(self, config):
        # Call super's `setup` to create rollout workers.
        super().setup(config)
        # Create local replay buffer.
        self.local_replay_buffer = MultiAgentReplayBuffer(num_shards=1, capacity=50000)

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        # Generate common experiences, collect batch for PPO, store every (DQN) batch
        # into replay buffer.
        num_env_steps = 0

        # PPO batch size fixed at 200.
        # TODO: Use `max_env_steps=200` option of synchronous_parallel_sample instead.
        #if self._by_agent_steps:
        ma_batch = synchronous_parallel_sample(
            worker_set=self.workers, max_agent_steps=self.get_policy("dispatcher1").config["train_batch_size"] * 16
        )
        # else:
        #     ma_batch = synchronous_parallel_sample(
        #         worker_set=self.workers, max_env_steps=self.get_policy("dispatcher1").config["train_batch_size"] * 4
        #     )
        # while num_env_steps < 200:
        #     ma_batches = synchronous_parallel_sample(
        #         worker_set=self.workers, concat=False
        #     )
            # Loop through ma-batches (which were collected in parallel).
            #for ma_batch in ma_batches:
                # Update sampled counters.
        self._counters[NUM_ENV_STEPS_SAMPLED] += ma_batch.count
        self._counters[NUM_AGENT_STEPS_SAMPLED] += ma_batch.agent_steps()
        if "dispatcher1" in ma_batch.policy_batches.keys():
            ppo_batch1 = ma_batch.policy_batches.pop("dispatcher1")  
            num_env_steps += ppo_batch1.count
        else:
            ppo_batch1 = None
        if "dispatcher2" in ma_batch.policy_batches.keys():
            ppo_batch2 = ma_batch.policy_batches.pop("dispatcher2")  
            num_env_steps += ppo_batch2.count
        else:
            ppo_batch2 = None
        # Add collected batches (only for DQN policy) to replay buffer.
        self.local_replay_buffer.add(ma_batch)

        # DQN sub-flow.
        dqn_train_results = {}
        # Start updating DQN policy once we have some samples in the buffer.
        cur_ts = self._counters[
            NUM_AGENT_STEPS_SAMPLED #if self._by_agent_steps else NUM_ENV_STEPS_SAMPLED
        ]
        if cur_ts > self.get_policy("agv").config["num_steps_sampled_before_learning_starts"]:
            # Update DQN policy n times while updating PPO policy once.
            for _ in range(10):
                dqn_train_batch = self.local_replay_buffer.sample(self.get_policy("agv").config["train_batch_size"])
                dqn_train_results = train_one_step(
                    self, dqn_train_batch, ["agv"]
                )
                self._counters[
                    "agent_steps_trained_DQN"
                ] += dqn_train_batch.agent_steps()
                # print(
                #     "DQN policy learning on samples from",
                #     "agent steps trained",
                #     dqn_train_batch.agent_steps(),
                # )
        # Update DQN's target net every n train steps (determined by the DQN config).
        if (
            self._counters["agent_steps_trained_DQN"]
            - self._counters[LAST_TARGET_UPDATE_TS]
            >= self.get_policy("agv").config["target_network_update_freq"]
        ):
            self.workers.local_worker().get_policy("agv").update_target()
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = self._counters[
                "agent_steps_trained_DQN"
            ]

        # PPO sub-flow.
        ppo_train_results1 = self.ppo_sub_flow(ppo_batch1, "dispatcher1")
        ppo_train_results2 = self.ppo_sub_flow(ppo_batch2, "dispatcher2")

        # Combine results for PPO and DQN into one results dict.
        results = dict(ppo_train_results1, **ppo_train_results2, **dqn_train_results)
        return results

    def ppo_sub_flow(self, ppo_train_batch, policy_name):
        if ppo_train_batch:
            self._counters["agent_steps_trained_PPO"] += ppo_train_batch.agent_steps()
            ppo_train_batch[Postprocessing.ADVANTAGES] = standardized(
                ppo_train_batch[Postprocessing.ADVANTAGES]
            )
            ppo_train_batch = MultiAgentBatch(
                {policy_name: ppo_train_batch}, ppo_train_batch.count
            )
            ppo_train_results = train_one_step(self, ppo_train_batch, [policy_name])
        else:
            ppo_train_results = {}
        return ppo_train_results
