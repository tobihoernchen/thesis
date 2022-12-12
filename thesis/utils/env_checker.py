import time
import numpy as np
import random
from ..envs.matrix import Matrix
from IPython.display import clear_output, display
import plotly.express as px
import pandas as pd
import keyboard



class RewardCheck:
    def __init__(self, env: Matrix) -> None:
        self.env = env
        self.history = {}
        self.include_nodes_in_reach = env.config.obs_include_nodes_in_reach

    def run(
        self,
        n_episodes=1,
        seed=42,
        reset=True,
        manual_agents=[],
    ):
        random.seed(seed)
        self.step_times = []
        episode_counter = 0
        while episode_counter < n_episodes:
            if reset:
                self.env.reset(seed=seed)
            self.total_reward = 0
            step_counter = 0
            for agent in self.env.agent_iter():
                if agent not in self.history.keys():
                    self.history[agent] = dict(reward=[], render=[], action=[], dict=[])
                last = self.env.last()
                self.total_reward += last[1]
                self.history[agent]["reward"].append(last[1])
                render = self.env.render()
                dict_render = self.env.render(dict_mode=True)
                self.history[agent]["render"].append(render)
                self.history[agent]["dict"].append(dict_render)
                if "get_action" in self.env.agent_behaviors[agent].__class__.__dict__:
                    action = self.env.agent_behaviors[agent].get_action(
                        self.env.sim.get_observation()
                    )
                else:
                    if not agent in manual_agents:
                        action = [random.randint(0, 5) for i in range(10)]
                    else:
                        clear_output()
                        print(
                            f"AGENT {agent} --- STEP {step_counter} --- REWARD {last[1]}"
                        )
                        display(render)
                        action = [
                            0,
                        ]
                        while True:
                            if keyboard.is_pressed("left"):
                                action = [
                                    1,
                                ]
                                break
                            elif keyboard.is_pressed("up"):
                                action = [
                                    2,
                                ]
                                break
                            elif keyboard.is_pressed("right"):
                                action = [
                                    3,
                                ]
                                break
                            elif keyboard.is_pressed("down"):
                                action = [
                                    4,
                                ]
                                break
                            elif keyboard.is_pressed("enter"):
                                break
                        time.sleep(0.5)
                self.history[agent]["action"].append(action)
                before = time.time()
                self.env.step(action)
                self.step_times.append(time.time() - before)
                step_counter += 1
            episode_counter += 1
            print("+" * 20)
            print(f"Total Reward: {self.total_reward}; {step_counter} steps taken")
            print(
                f"Mean Step Time: {np.mean(self.step_times)}s ; Max: {max(self.step_times)}s ; Min: {min(self.step_times)}s"
            )
            print(self.env.statistics)

    def get_reward_situations(
        self,
        agents=None,
        reward_above=0,
        reward_below=0,
        play_random=False,
        frames_before=10,
        frames_after=5,
    ):

        situations = []
        if agents is None:
            agents = list(self.history.keys())
        for agent in agents:
            for i, reward in enumerate(self.history[agent]["reward"]):
                if reward > reward_above and reward < reward_below:
                    situations.append((agent, i))
        if play_random and len(situations) > 0:
            situation = random.choice(situations)
            self.replay(
                situation[0],
                0.5,
                max(0, situation[1] - frames_before),
                situation[1] + frames_after,
            )
        return situations

    def replay(self, agent, wait_s=0.5, start=0, end=10000):
        assert agent in self.history.keys()
        for i, (render, reward, action) in enumerate(
            zip(
                self.history[agent]["render"][start:end],
                self.history[agent]["reward"][start:end],
                self.history[agent]["action"][start:end],
            )
        ):
            time.sleep(wait_s)
            clear_output()
            print(
                f"AGENT {agent} --- STEP {start + i} --- ACTION {action} --- REWARD {reward}"
            )
            display(render)

    def plot(self, agent):
        data = []
        for i, (dict_render, reward, action) in enumerate(
            zip(
                self.history[agent]["dict"],
                self.history[agent]["reward"],
                self.history[agent]["action"],
            )
        ):
            for agv in dict_render["agvs"].keys():
                data.append(
                    [
                        i,
                        agv,
                        agv == "0",
                        "last",
                        dict_render["agvs"][agv]["last_node"][0],
                        dict_render["agvs"][agv]["last_node"][1],
                        reward,
                        action,
                        f"Last node of {agv} <br> reward: {reward} <br> action: {action} <br> in system: {dict_render['agvs'][agv]['in_system']}",
                    ]
                )
                data.append(
                    [
                        i,
                        agv,
                        agv == "0",
                        "next",
                        dict_render["agvs"][agv]["next_node"][0],
                        dict_render["agvs"][agv]["next_node"][1],
                        reward,
                        action,
                        f"Next node of {agv} <br> reward: {reward} <br> action: {action} <br> in system: {dict_render['agvs'][agv]['in_system']}",
                    ]
                )
                if "target_node" in dict_render["agvs"][agv].keys():
                    data.append(
                        [
                            i,
                            agv,
                            agv == "0",
                            "target",
                            dict_render["agvs"][agv]["target_node"][0],
                            dict_render["agvs"][agv]["target_node"][1],
                            reward,
                            action,
                            f"Target node of {agv} <br> reward: {reward} <br> action: {action} <br> in system: {dict_render['agvs'][agv]['in_system']}",
                        ]
                    )
        df = pd.DataFrame(
            data,
            columns=[
                "step",
                "name",
                "main",
                "type",
                "x",
                "y",
                "reward",
                "action",
                "text",
            ],
        )
        fig = px.line(
            df,
            "x",
            "y",
            animation_frame="step",
            line_group="name",
            color="main",
            symbol="type",
            hover_name="text",
            range_x=[0, 1.1],
            range_y=[1.1, 0],
            height=600,
            width=1000,
        )
        fig.show()
