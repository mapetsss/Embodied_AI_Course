"""Execute robot instructions independently of the translator."""


class Robot:
    HEADINGS = ("N", "E", "S", "W")
    VECTORS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}

    def __init__(self, env, heading="E"):
        if heading not in self.HEADINGS:
            raise ValueError("Invalid heading: " + str(heading))
        self.env = env
        self.heading = heading
        self.command_count = 0
        self.forward_count = 0

    def execute(self, action):
        if action in ("turn_left", "turn_right"):
            offset = -1 if action == "turn_left" else 1
            self.heading = self.HEADINGS[(self.HEADINGS.index(self.heading) + offset) % 4]
        elif action == "forward":
            dx, dy = self.VECTORS[self.heading]
            target = (self.env.pos[0] + dx, self.env.pos[1] + dy)
            if self.env.can_enter(target):
                self.env.pos = target
            self.forward_count += 1
        else:
            raise ValueError("Unknown robot action: " + str(action))
        self.command_count += 1
