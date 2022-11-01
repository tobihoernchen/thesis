import random
from alpyne.client.alpyne_client import AlpyneClient
from .base_alpyne_zoo import ZooAgent, BaseAlpyneZoo
from .behaviors import MultiAgent, SingleAgent, RandomStationDispatcher
from ..utils.build_config import build_config


class Matrix(BaseAlpyneZoo):

    counter = 0
    global_client = None

    def __init__(
        self,
        model_path: str = None,
        startport=51150,
        fleetsize: int = 1,
        max_fleetsize: int = 1,
        sim_config: dict = dict(),
        max_steps: int = None,
        max_seconds: int = None,
        pseudo_dispatcher=True,
        routing_agent_death=False,
        dispatching_agent_death=False,
    ):
        self.fleetsize = fleetsize
        self.max_fleetsize = max_fleetsize

        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.stepcounter = 0

        self.metadata = dict(is_parallelizable=True)
        self.statistics = None
        self.dispatch = sim_config["dispatch"]
        self.pseudo_dispatcher = pseudo_dispatcher
        self.routing_ma = sim_config["routing_ma"]
        self.dispatching_ma = sim_config["dispatching_ma"]
        self.routing_agent_death = routing_agent_death
        self.dispatching_agent_death = dispatching_agent_death
        if routing_agent_death:
            self.routing_hwm = self.fleetsize
        if dispatching_agent_death:
            self.dispatching_hwm = 1000 + self.fleetsize

        self.client = None
        self.started = False
        self.model_path = model_path
        self.startport = startport

        self.config = self.get_config(sim_config)
        self.object_agents = self.get_agents()
        self.possible_agents = [agent.name for agent in self.object_agents]
        if routing_agent_death:
            self.possible_agents.extend([str(i) for i in range(self.fleetsize, 1000)])
        if dispatching_agent_death:
            self.possible_agents.extend(
                [str(i) for i in range(self.fleetsize + 1000, 2000)]
            )
        super().__init__(None, self.object_agents)
        self.reset()

    def _agent_selection_fn(self, alpyne_obs):
        if alpyne_obs.caller < 1000 and self.routing_agent_death:
            return self.routing_aliases[str(alpyne_obs.caller)]
        elif alpyne_obs.caller >= 1000 and self.dispatching_agent_death:
            return self.dispatching_aliases[str(alpyne_obs.caller)]
        else:
            return str(alpyne_obs.caller)

    def step(self, action):
        self.stepcounter += 1
        if int(self.agent_selection) < 1000 and self.routing_agent_death:
            self.agent_selection = self.routing_aliases_rev[self.agent_selection]
        elif int(self.agent_selection) >= 1000 and self.dispatching_agent_death:
            self.agent_selection = self.dispatching_aliases_rev[self.agent_selection]
        return super().step(action)

    def get_agents(self):
        agents = []
        if self.routing_ma:
            agents.extend(
                [
                    ZooAgent(str(i), MultiAgent(self.max_fleetsize, 5))
                    for i in range(self.fleetsize)
                ]
            )
            if self.routing_agent_death:
                self.routing_hwm = self.fleetsize
                self.routing_aliases = {str(i): str(i) for i in range(self.fleetsize)}
                self.routing_aliases_rev = {
                    str(i): str(i) for i in range(self.fleetsize)
                }
        else:
            agents.append(ZooAgent("2000", SingleAgent(self.max_fleetsize, 5)))
        if self.dispatch:
            if self.pseudo_dispatcher:
                if self.dispatching_ma:
                    agents.extend(
                        [
                            ZooAgent(str(i), RandomStationDispatcher(1), False)
                            for i in range(1000, 1000 + self.fleetsize)
                        ]
                    )
                else:
                    agents.append(
                        ZooAgent("2001", RandomStationDispatcher(self.fleetsize), False)
                    )
            else:
                if self.dispatching_ma:
                    agents.extend(
                        [
                            ZooAgent(str(i), MultiAgent(self.max_fleetsize, 5, True))
                            for i in range(1000, 1000 + self.fleetsize)
                        ]
                    )
                    if self.dispatching_agent_death:
                        self.dispatching_hwm = self.fleetsize + 1000
                        self.dispatching_aliases = {
                            str(i): str(i) for i in range(1000, 1000 + self.fleetsize)
                        }
                        self.dispatching_aliases_rev = {
                            str(i): str(i) for i in range(1000, 1000 + self.fleetsize)
                        }
                else:
                    agents.append(
                        ZooAgent("2001", MultiAgent(self.max_fleetsize, 5, True))
                    )
        return agents

    def get_config(self, config_args):
        conf = build_config(config_args, self.fleetsize)
        conf.obs_include_nodes_in_reach = True
        conf.coordinates = True
        return conf

    def seed(self, seed: int = None):
        if seed is not None:
            self.config.seed = seed
        else:
            self.config.seed = random.randint(0, 10000)

    def _start(self):
        if self.__class__.global_client is None:
            port = self.startport + int(self.__class__.counter)
            self.__class__.counter = (
                self.__class__.counter + 1
            )  # increment to change port for all envs created
            self.__class__.global_client = AlpyneClient(
                self.model_path, port=port, verbose=False
            )

        self.client = self.__class__.global_client
        self.run = self.client.create_reinforcement_learning(self.config)
        self.started = True
        super()._start(self.run)

    def reset(
        self,
        config=None,
        seed=None,
        return_info=False,
        options=None,
    ) -> "BaseAlpyneZoo.PyObservationType":
        if not self.started:
            self._start()
        self.stepcounter = 0
        self.seed(seed=seed)
        return super().reset(self.config, seed, return_info, options)

    def _save_observation(self, alpyne_obs, agent, obs, reward, done, info):
        if "statTitles" in alpyne_obs.names():
            self.statistics = {
                title: value
                for title, value in zip(alpyne_obs.statTitles, alpyne_obs.statValues)
            }
        not_in_system = "in_system" in info.keys() and not info["in_system"]
        if int(agent) < 1000 and self.routing_agent_death and not_in_system:
            self.remove_agent(agent)
            name = str(self.routing_hwm)
            original_name = self.routing_aliases_rev[agent]
            self.routing_aliases[original_name] = name
            self.routing_aliases_rev[name] = original_name  # bidirectional
            self.add_agent(ZooAgent(name, MultiAgent(self.max_fleetsize, 5)))
            self.routing_hwm += 1
        elif int(agent) >= 1000 and self.dispatching_agent_death and not_in_system:
            self.remove_agent(agent)
            name = str(self.dispatching_hwm)
            original_name = self.dispatching_aliases_rev[agent]
            self.dispatching_aliases[original_name] = name
            self.dispatching_aliases_rev[name] = original_name  # bidirectional
            self.add_agent(ZooAgent(name, MultiAgent(self.max_fleetsize, 5, True)))
            self.dispatching_hwm += 1
        return super()._save_observation(agent, alpyne_obs, obs, reward, done, info)

    def _terminal_alternative(self, observation) -> bool:
        """Optional method to add *extra* terminating conditions"""
        terminal_max_steps = (
            (self.stepcounter >= self.max_steps)
            if self.max_steps is not None
            else False
        )
        time = 0 if self.max_seconds is None else self.run.get_state()[1]["model_time"]
        terminal_max_seconds = (
            time >= self.max_seconds if self.max_seconds is not None else False
        )
        return terminal_max_steps or terminal_max_seconds

    def close(self):
        self.sim.stop()
