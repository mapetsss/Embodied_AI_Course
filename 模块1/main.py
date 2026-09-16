"""Run homework 1: plan, translate, execute and verify."""
from env import GridWorld
from planner import find_path
from translator import translate_actions
from robot import Robot
from robot_planner import find_robot_commands


def main():
    grid_env, robot_env, optimized_env = GridWorld(), GridWorld(), GridWorld()
    plan = find_path(grid_env)
    if plan is None:
        raise RuntimeError("Goal is unreachable")
    commands = translate_actions(plan)
    robot = Robot(robot_env)
    search_stats = {}
    optimized_commands = find_robot_commands(optimized_env, stats=search_stats)
    if optimized_commands is None:
        raise RuntimeError("Goal is unreachable for robot")
    optimized_robot = Robot(optimized_env)
    for action in plan:
        grid_env.step(action)
    for command in commands:
        robot.execute(command)
    for command in optimized_commands:
        optimized_robot.execute(command)
    assert grid_env.pos == robot_env.pos == grid_env.goal
    assert optimized_env.pos == optimized_env.goal
    assert len(optimized_commands) <= len(commands)
    print("Grid actions:", plan)
    print("Robot actions:", commands)
    print("Grid final position:", grid_env.pos)
    print("Robot final position:", robot_env.pos, "heading:", robot.heading)
    print("Grid steps:", grid_env.steps, "robot commands:", robot.command_count)
    print("Optimized robot actions:", optimized_commands)
    print("Optimized final position:", optimized_env.pos,
          "heading:", optimized_robot.heading)
    print("Optimized robot commands:", optimized_robot.command_count,
          "expanded states:", search_stats["expanded_states"])
    print("Commands saved:", len(commands) - len(optimized_commands))
    print("PASS: final positions match")


if __name__ == "__main__":
    main()
