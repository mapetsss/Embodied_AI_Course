"""Plan the fewest robot commands while treating heading as state."""
from collections import deque

HEADINGS = ("N", "E", "S", "W")
VECTORS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
COMMANDS = ("forward", "turn_left", "turn_right")


def find_robot_commands(env, initial_heading="E", stats=None):
    """Return a minimum-command plan, [] at goal, or None if unreachable."""
    if initial_heading not in HEADINGS:
        raise ValueError("Invalid heading: " + str(initial_heading))

    start = (env.pos, initial_heading)
    if stats is not None:
        stats.clear()
        stats.update(expanded_states=0, discovered_states=1)
    if env.pos == env.goal:
        return []

    parents = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if stats is not None:
            stats["expanded_states"] += 1
        for command in COMMANDS:
            nxt = _transition(env, state, command)
            if nxt is None or nxt in parents:
                continue
            parents[nxt] = (state, command)
            if stats is not None:
                stats["discovered_states"] += 1
            if nxt[0] == env.goal:
                return _reconstruct(parents, nxt)
            queue.append(nxt)
    return None


def _transition(env, state, command):
    pos, heading = state
    heading_index = HEADINGS.index(heading)
    if command == "turn_left":
        return pos, HEADINGS[(heading_index - 1) % 4]
    if command == "turn_right":
        return pos, HEADINGS[(heading_index + 1) % 4]
    dx, dy = VECTORS[heading]
    target = (pos[0] + dx, pos[1] + dy)
    return (target, heading) if env.can_enter(target) else None


def _reconstruct(parents, goal_state):
    commands = []
    current = goal_state
    while parents[current] is not None:
        current, command = parents[current]
        commands.append(command)
    commands.reverse()
    return commands
