import itertools
import unittest
from env import GridWorld
from planner import find_path
from translator import translate_actions
from robot import Robot
from robot_planner import find_robot_commands


class HomeworkTests(unittest.TestCase):
    def test_default(self):
        env = GridWorld()
        plan = find_path(env)
        self.assertEqual(len(plan), 8)
        commands = translate_actions(plan)
        self.assertEqual(len(commands), 10)
        robot = Robot(env)
        for command in commands:
            robot.execute(command)
        self.assertEqual(env.pos, (4, 4))

    def test_translation_equivalence(self):
        # Includes reverse turns, stay, boundaries and obstacle collisions.
        for heading in Robot.HEADINGS:
            for actions in itertools.product(GridWorld.ACTIONS, repeat=3):
                grid, physical = GridWorld(), GridWorld()
                robot = Robot(physical, heading)
                for action in actions:
                    grid.step(action)
                for command in translate_actions(actions, heading):
                    robot.execute(command)
                self.assertEqual(grid.pos, physical.pos, (heading, actions))

    def test_empty_unreachable_and_invalid(self):
        env = GridWorld()
        env.pos = env.goal
        self.assertEqual(find_path(env), [])
        self.assertEqual(translate_actions([]), [])
        env.pos = (0, 0)
        env.obstacles.update({(0, 1), (1, 0)})
        self.assertIsNone(find_path(env))
        with self.assertRaises(ValueError):
            translate_actions(["jump"])
        with self.assertRaises(ValueError):
            translate_actions([], "X")

    def test_heading_aware_plan_uses_fewer_commands(self):
        env = GridWorld()
        stats = {}
        commands = find_robot_commands(env, stats=stats)
        self.assertEqual(commands,
                         ["forward"] * 4 + ["turn_left"] + ["forward"] * 4)
        self.assertEqual(len(commands), 9)
        robot = Robot(env)
        for command in commands:
            robot.execute(command)
        self.assertEqual(env.pos, env.goal)
        self.assertGreater(stats["expanded_states"], 0)

    def test_robot_planner_edge_cases(self):
        for heading in Robot.HEADINGS:
            env = GridWorld()
            commands = find_robot_commands(env, heading)
            robot = Robot(env, heading)
            for command in commands:
                robot.execute(command)
            self.assertEqual(env.pos, env.goal)

        env = GridWorld()
        env.pos = env.goal
        self.assertEqual(find_robot_commands(env), [])
        env.pos = (0, 0)
        env.obstacles.update({(0, 1), (1, 0)})
        self.assertIsNone(find_robot_commands(env))
        with self.assertRaises(ValueError):
            find_robot_commands(GridWorld(), "X")


if __name__ == "__main__":
    unittest.main()
