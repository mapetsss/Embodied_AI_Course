"""A fully observable agent: perceive, plan, act, remember."""
from collections import deque
from types import SimpleNamespace
from planner import find_path
from memory import Memory


class Agent:
    def __init__(self, env):
        self.env = env
        self.memory = Memory(env.pos)
        self.planning_calls = 0
        self.expanded_nodes = 0
        self.discovered_nodes = 0

    def perceive(self):
        # A detached snapshot prevents the planner from changing live state.
        width, height = self.env.width, self.env.height
        obstacles = frozenset(self.env.obstacles)
        return SimpleNamespace(
            pos=self.env.pos, goal=self.env.goal,
            ACTIONS=dict(self.env.ACTIONS),
            can_enter=lambda p: (0 <= p[0] < width and 0 <= p[1] < height
                                 and p not in obstacles))

    def run(self, max_steps=100, on_step=None):
        """Call on_step(record, summary) after each recorded action, if supplied."""
        if max_steps < 0:
            raise ValueError("max_steps must be nonnegative")
        executed = 0
        plan = deque()
        while self.env.pos != self.env.goal:
            if executed >= max_steps:
                raise RuntimeError("Step limit reached")
            observation = self.perceive()
            if not plan or not self._next_action_is_valid(plan[0], observation):
                search_stats = {}
                new_plan = find_path(observation, search_stats)
                self.planning_calls += 1
                self.expanded_nodes += search_stats["expanded_nodes"]
                self.discovered_nodes += search_stats["discovered_nodes"]
                if new_plan is None:
                    raise RuntimeError("Goal is unreachable")
                plan = deque(new_plan)
            action = plan.popleft()
            before = self.env.pos
            self.env.step(action)
            self.memory.record(before, action, self.env.pos)
            executed += 1
            if on_step is not None:
                on_step(self.memory.records[-1], self.memory.summary())
        result = self.memory.summary()
        result.update(planning_calls=self.planning_calls,
                      expanded_nodes=self.expanded_nodes,
                      discovered_nodes=self.discovered_nodes)
        return result

    def _next_action_is_valid(self, action, observation):
        dx, dy = observation.ACTIONS[action]
        target = (observation.pos[0] + dx, observation.pos[1] + dy)
        return observation.can_enter(target)
