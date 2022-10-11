from alpyne.data.spaces import Action, Observation
import random
import numpy as np


class Dispatcher:
    def __init__(self) -> None:
        self.context = None

    def get_context(self, context):
        nodecontext = [c for c in context if c[-1] == 0]
        self.nodes = dict()
        [self.nodes.update({i: tuple(n[:2])}) for i, n in enumerate(nodecontext)]
        self.stations = {
            i: tuple(n[:2]) for i, n in enumerate(nodecontext) if n[2] == 1
        }
        self.nodeCoords = np.array(list(self.nodes.values()))

        pathcontext = [c for c in context if c[-1] != 0]
        self.context = dict()
        [
            self.context.update({self.getClosest(tuple(path[:2])): []})
            for path in pathcontext
        ]
        [
            self.context.update({self.getClosest(tuple(path[2:])): []})
            for path in pathcontext
        ]
        [
            self.context[self.getClosest(tuple(path[:2]))].append(
                self.getClosest(tuple(path[2:]))
            )
            for path in pathcontext
        ]
        [
            self.context[self.getClosest(tuple(path[2:]))].append(
                self.getClosest(tuple(path[:2]))
            )
            for path in pathcontext
        ]

    def getClosest(self, node):
        index = np.abs(self.nodeCoords - node).mean(1).argmin()
        return index

    def __call__(self, obs: Observation) -> Action:
        if (
            "networkcontext" in obs.names()
            and obs.networkcontext is not None
            and len(obs.networkcontext) > 0
        ):
            self.get_context(obs.networkcontext)

    def makeAction(self, actions, receiver: int = 2001):
        return Action(
            data=[
                {
                    "name": "actions",
                    "type": "INTEGER_ARRAY",
                    "value": list(actions),
                    "unit": None,
                },
                {
                    "name": "receiver",
                    "type": "INTEGER",
                    "value": receiver,
                    "unit": None,
                },
            ]
        )


class RandDispatcher(Dispatcher):
    def __init__(self, distance=2) -> None:
        super().__init__()
        self.distance = distance

    def __call__(self, obs: Observation) -> Action:
        super().__call__(obs)
        if self.context is None or self.distance is None:
            actions = random.sample(range(obs.n_nodes), len(obs.obs))
        else:
            lasts = [i[1:3] for i in obs.obs]
            nexts = [i[3:5] for i in obs.obs]
            actions = [self.getNode(l, n) for l, n in zip(lasts, nexts)]
        return self.makeAction(actions)

    def getNode(self, last, next):
        next = (
            self.getClosest(tuple(next))
            if next[0] != 0
            else self.getClosest(tuple(last))
        )
        last = self.getClosest(tuple(last))

        for i in range(self.distance):
            possible = list(self.context[next])
            if last in possible:
                possible.pop(possible.index(last))
            last = int(next)
            next = random.choice(possible)
        return next


class StationDispatcher(Dispatcher):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, obs: Observation) -> Action:
        receiver = int(obs.caller.replace("_Dispatching", "")) + 1000
        super().__call__(obs)
        if self.context is None:
            action = random.choice(range(obs.n_nodes))
        else:
            coords = random.choice(list(self.stations.values()))
            action = self.getClosest(tuple(coords))

        return self.makeAction(
            [
                action,
            ],
            receiver,
        )


# class CleverDispatcher(Dispatcher):

#     def __init__(self) -> None:
#         super().__init__()
#         self.stations = dict(
#             geo1 = (0.32954545454545453, 0.49767441860465117),
#             geo2 = (0.32954545454545453, 0.9395348837209302),
#             hsn = (0.7909090909090909, 0.7209302325581395),
#             wps = (0.9727272727272728, 0.7209302325581395),
#             rework = (0.5727272727272728, 0.7209302325581395),
#         )
#         self.workflow = ["geo1", "hsn", "wps", "rework", "geo2"]
#         self.states = dict()


#     def __call__(self, obs: Observation) -> Action:
#         super().__call__(obs)
#         assert self.context is not None
#         if int(obs.caller) >= 0:
#             if int(obs.caller) not in self.states.keys():
#                 self.states.update({int(obs.caller): self.workflow[0]})
#             state = self.states[int(obs.caller)]
#             nexts = [i[4:6] for i in obs.obs]
#             if obs.
