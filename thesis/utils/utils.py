import time
import json
import numpy as np
import time
import random


def save_hparams(
    fleetsize=None,
    max_fleetsize=None,
    env_args={},
    algo_args={},
    fe_args={},
    net_arch=[],
    dir="Default",
    run_name="",
    **kwargs,
):
    hparams = dict(
        fleetsize=fleetsize,
        max_fleetsize=max_fleetsize,
        env_args=env_args,
        algo_args=algo_args,
        fe_args=fe_args,
        net_arch=net_arch,
    )
    models_dir = f"../../models/{dir}"
    run_name = (
        run_name + f"{fleetsize}-{max_fleetsize}-{time.strftime('%d_%m-%H_%M_%S')}"
    )
    with open(f"{models_dir}/{run_name}.json", "w") as outfile:
        json.dump(hparams, outfile, indent=3)
        return models_dir, run_name


def manual_routing(next, target, possibles=None):
    if possibles is None:
        possibles = [
            True,
        ] * 4
    dx = abs(next[0] - target[0])
    dy = abs(next[1] - target[1])
    action = None
    if dx > dy:
        if next[0] > target[0]:
            action = 1
        if next[0] < target[0]:
            action = 3
    if dx < dy:
        if next[1] > target[1]:
            action = 4
        if next[1] < target[1]:
            action = 2
    if action is not None and possibles[action - 1]:
        return action
    elif action is not None:
        return random.choice([i + 1 for i, b in enumerate(possibles) if b])
    else:
        return 0


class RewardCheck:
    """WIP"""

    def __init__(self, env, obs2action=None) -> None:
        self.env = env
        self.hist = dict()
        if "agent_selection" in env.__dict__.keys():
            self.zoo = True
            self.gym = False
        else:
            self.zoo = False
            self.gym = True
        if obs2action is not None:
            self.obs2action = obs2action
        else:
            self.obs2action = lambda obs: self.get_action(obs)

    def run(
        self,
        n_episodes=None,
        agents=None,
        reward_above=-100,
        reward_below=100,
        seed=None,
    ):
        self.times = []
        counter = 0
        self.reward_above = reward_above
        self.reward_below = reward_below
        self.check_for = (
            agents
            if isinstance(agents, list)
            else [
                agents,
            ]
        )
        while counter < n_episodes or n_episodes is None:
            self.env.reset(seed=seed)
            self.action = 0
            self.total_rew = 0
            if self.gym:
                self.state = [0] * 20, 0, 0, 0
            if self.zoo:
                self.state = {
                    agent: [[0] * 20, 0, 0, 0] for agent in self.env.possible_agents
                }
            self.steps = 0
            if self.zoo:
                for agent in self.env.agent_iter():
                    last = self.env.last()
                    self.action = self.obs2action(last[0])
                    before = time.time()
                    self.env.step(self.action)
                    self.times.append(time.time() - before)
                    self.state[agent] = self.check(
                        self.state[agent], self.action, agent, last
                    )
                    self.total_rew += self.state[agent][1]
                    self.steps += 1
            if self.gym:
                done = False
                obs = reward = done = info = None
                agent = "default"
                while not done:
                    self.action = self.obs2action(obs, reward, done, info, agent)
                    obs, reward, done, info = self.check(
                        self.state, self.action, agent, self.env.step(self.action)
                    )
                    self.steps += 1
            counter += 1
            print("+" * 20)
            print(f"Total Reward: {self.total_rew}")
            print(
                f"Mean Step Time: {np.mean(self.times)}s ; Max: {max(self.times)}s ; Min: {min(self.times)}s"
            )

    def check(self, old_state, action, agent, new_state):

        obs_o, reward_o, done_o, info_o = old_state
        obs_n, reward_n, done_n, info_n = new_state

        if self.check_for[0] is None or agent in self.check_for:
            if reward_n > self.reward_above or reward_n < self.reward_below:
                print(f"{self.steps}---{agent}" + "-" * 20)
                print(f"OLD: {obs_o[:3]} \t done:{done_o}")
                print(f"OLD: curr: {obs_o[3:5]}\t next:{obs_o[5:7]}\t tar:{obs_o[7:9]}")
                print(f"ACTION: {action}")
                print(f"NEW: {obs_n[:3]} \t done:{done_n}")
                print(f"NEW: curr: {obs_n[3:5]}\t next:{obs_n[5:7]}\t tar:{obs_n[7:9]}")
                print(f"REWARD: {reward_n}")

        if np.all(np.isclose(obs_n[5:7], obs_n[7:9], 0.01)):
            pass
            # assert reward_n > 0.7
        if reward_n > 0.7:
            pass
            # assert np.all(np.isclose(obs_n[5:7], obs_n[7:9], 0.01))
        return obs_n, reward_n, done_n, info_n

    def get_action(self, observation):
        next = observation[5:7]
        target = observation[7:9]
        action = manual_routing(next, target)
        return action
