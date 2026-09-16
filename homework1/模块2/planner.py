"""Breadth-first search: shortest number of grid moves."""
from collections import deque


def find_path(env, stats=None):
    """Return a shortest path and optionally fill search statistics."""
    if stats is not None:
        stats.clear()
        stats.update(expanded_nodes=0, discovered_nodes=1)
    if env.pos == env.goal:
        return []
    parents = {env.pos: None}
    queue = deque([env.pos])
    while queue:
        x, y = queue.popleft()
        if stats is not None:
            stats["expanded_nodes"] += 1
        for name, (dx, dy) in env.ACTIONS.items():
            if name == "stay":
                continue
            nxt = (x + dx, y + dy)
            if not env.can_enter(nxt) or nxt in parents:
                continue
            parents[nxt] = ((x, y), name)
            if stats is not None:
                stats["discovered_nodes"] += 1
            if nxt == env.goal:
                return _reconstruct_path(parents, nxt)
            queue.append(nxt)
    return None


def _reconstruct_path(parents, goal):
    actions = []
    current = goal
    while parents[current] is not None:
        current, action = parents[current]
        actions.append(action)
    actions.reverse()
    return actions
