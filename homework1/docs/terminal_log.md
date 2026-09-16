# 终端运行日志

以下记录来自用户提供的 PowerShell 运行结果。已还原粘贴过程中被 Markdown 转义或转为列表的字符，地图使用纯文本代码块保留排列；测试分隔线统一为横线。

```text
PS E:\A GDUT\dasanshang\jiqirengailun> powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
Running homework1
Grid actions: ['up', 'up', 'up', 'up', 'right', 'right', 'right', 'right']
Robot actions: ['turn_left', 'forward', 'forward', 'forward', 'forward', 'turn_right', 'forward', 'forward', 'forward', 'forward']
Grid final position: (4, 4)
Robot final position: (4, 4) heading: E
Grid steps: 8 robot commands: 10
PASS: final positions match
...
----------------------------------------------------------------------
Ran 3 tests in 0.003s

OK
Running homework2
Initial map (step 0): (0, 0)
Legend: A=agent S=start *=visited G=goal #=obstacle .=unvisited
. . . . G
. . . . .
. # # # .
. # . . .
A . . . .

Step 1: (0, 0) --up--> (0, 1)
. . . . G
. . . . .
. # # # .
A # . . .
S . . . .

Step 2: (0, 1) --up--> (0, 2)
. . . . G
. . . . .
A # # # .
* # . . .
S . . . .

Step 3: (0, 2) --up--> (0, 3)
. . . . G
A . . . .
* # # # .
* # . . .
S . . . .

Step 4: (0, 3) --up--> (0, 4)
A . . . G
* . . . .
* # # # .
* # . . .
S . . . .

Step 5: (0, 4) --right--> (1, 4)
* A . . G
* . . . .
* # # # .
* # . . .
S . . . .

Step 6: (1, 4) --right--> (2, 4)
* * A . G
* . . . .
* # # # .
* # . . .
S . . . .

Step 7: (2, 4) --right--> (3, 4)
* * * A G
* . . . .
* # # # .
* # . . .
S . . . .

Step 8: (3, 4) --right--> (4, 4) (goal reached)
* * * * A
* . . . .
* # # # .
* # . . .
S . . . .

Trajectory: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
Final position: (4, 4)
Steps: 8
PASS: goal (4, 4), 8 steps, 9 trajectory positions
.....
----------------------------------------------------------------------
Ran 5 tests in 0.001s

OK
```
