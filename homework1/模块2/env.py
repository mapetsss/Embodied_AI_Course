"""Grid state and collision rules shared by planning and execution."""


class GridWorld:
    ACTIONS = {"up": (0, 1), "down": (0, -1),
               "left": (-1, 0), "right": (1, 0), "stay": (0, 0)}

    def __init__(self):
        self.width, self.height = 5, 5
        self.goal = (4, 4)
        self.obstacles = {(1, 1), (1, 2), (2, 2), (3, 2)}
        self.pos = (0, 0)
        self.steps = 0

    def can_enter(self, pos):
        x, y = pos
        return (0 <= x < self.width and 0 <= y < self.height
                and pos not in self.obstacles)

    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError("Unknown grid action: " + str(action))
        dx, dy = self.ACTIONS[action]
        target = (self.pos[0] + dx, self.pos[1] + dy)
        if self.can_enter(target):
            self.pos = target
        self.steps += 1
        return self.pos == self.goal

    def render(self, trajectory=(), start=None):
        """Display actual visited positions without changing world state."""
        visited = set(trajectory)
        for y in range(self.height - 1, -1, -1):
            row = []
            for x in range(self.width):
                pos = (x, y)
                row.append("A" if pos == self.pos else
                           "G" if pos == self.goal else
                           "#" if pos in self.obstacles else
                           "S" if pos == start else
                           "*" if pos in visited else ".")
            print(" ".join(row))
