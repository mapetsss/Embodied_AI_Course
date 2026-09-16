"""Translate absolute grid moves into relative robot instructions."""
HEADINGS = ("N", "E", "S", "W")
TARGET_HEADING = {"up": "N", "right": "E", "down": "S", "left": "W"}


def translate_actions(grid_actions, initial_heading="E"):
    if initial_heading not in HEADINGS:
        raise ValueError("Invalid heading: " + str(initial_heading))
    heading = HEADINGS.index(initial_heading)
    result = []
    for action in grid_actions:
        if action == "stay":
            continue  # Position equivalence only; no robot wait instruction.
        if action not in TARGET_HEADING:
            raise ValueError("Unknown grid action: " + str(action))
        target = HEADINGS.index(TARGET_HEADING[action])
        delta = (target - heading) % 4
        if delta == 3:
            result.append("turn_left")
        else:
            result.extend(["turn_right"] * delta)
        result.append("forward")
        heading = target
    return result
