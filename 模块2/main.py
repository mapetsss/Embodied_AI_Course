"""Run homework 2 and verify the default assignment map."""
from env import GridWorld
from agent import Agent


def main():
    env = GridWorld()
    agent = Agent(env)
    start = env.pos
    print("Initial map (step 0):", start)
    print("Legend: A=agent S=start *=visited G=goal #=obstacle .=unvisited")
    env.render(trajectory=[start], start=start)
    print()

    def show_step(record, summary):
        status = " (goal reached)" if env.pos == env.goal else ""
        print(f"Step {record.step}: {record.from_pos} --{record.action}--> {record.to_pos}{status}")
        env.render(trajectory=summary["trajectory"], start=start)
        print()

    result = agent.run(on_step=show_step)
    print("Trajectory:", result["trajectory"])
    print("Final position:", result["final_pos"])
    print("Steps:", result["steps"])
    print("Planning calls:", result["planning_calls"])
    print("Expanded nodes:", result["expanded_nodes"])
    assert result["final_pos"] == (4, 4)
    assert result["steps"] == env.steps == 8
    assert len(result["trajectory"]) == 9
    assert result["planning_calls"] == 1
    print("PASS: goal (4, 4), 8 steps, 9 trajectory positions")


if __name__ == "__main__":
    main()
