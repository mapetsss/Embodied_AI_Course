import unittest
from env import GridWorld
from agent import Agent
from memory import Memory


class HomeworkTests(unittest.TestCase):
    def test_default_and_memory(self):
        env = GridWorld()
        agent = Agent(env)
        result = agent.run()
        self.assertEqual(result["final_pos"], (4, 4))
        self.assertEqual(result["steps"], 8)
        self.assertEqual(env.steps, 8)
        self.assertEqual(len(result["trajectory"]), 9)
        self.assertEqual(result["planning_calls"], 1)
        self.assertGreater(result["expanded_nodes"], 0)
        replay = GridWorld()
        for record in agent.memory.records:
            self.assertEqual(record.from_pos, replay.pos)
            replay.step(record.action)
            self.assertEqual(record.to_pos, replay.pos)

    def test_at_goal(self):
        env = GridWorld()
        env.pos = env.goal
        result = Agent(env).run(max_steps=0)
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["trajectory"], [(4, 4)])

    def test_unreachable(self):
        env = GridWorld()
        env.obstacles.update({(0, 1), (1, 0)})
        agent = Agent(env)
        with self.assertRaisesRegex(RuntimeError, "unreachable"):
            agent.run()
        self.assertEqual(agent.memory.records, [])

    def test_limit_and_snapshot(self):
        env = GridWorld()
        agent = Agent(env)
        observation = agent.perceive()
        env.obstacles.add((0, 1))
        self.assertTrue(observation.can_enter((0, 1)))
        with self.assertRaisesRegex(RuntimeError, "limit"):
            agent.run(max_steps=1)
        self.assertEqual(len(agent.memory.records), 1)

    def test_blocked_move_is_recorded(self):
        env = GridWorld()
        memory = Memory(env.pos)
        before = env.pos
        env.step("left")
        memory.record(before, "left", env.pos)
        self.assertEqual(memory.summary()["trajectory"], [(0, 0), (0, 0)])
        self.assertEqual(env.steps, 1)

    def test_replans_when_cached_path_is_blocked(self):
        env = GridWorld()
        agent = Agent(env)

        def add_dynamic_obstacle(record, summary):
            if record.step == 1:
                env.obstacles.add((0, 4))

        result = agent.run(on_step=add_dynamic_obstacle)
        self.assertEqual(result["final_pos"], (4, 4))
        self.assertEqual(result["planning_calls"], 2)
        self.assertNotIn((0, 4), result["trajectory"])


if __name__ == "__main__":
    unittest.main()
