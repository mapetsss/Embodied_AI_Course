#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for assignment in 模块1 模块2; do
    echo "Running $assignment"
    python3 "$assignment/main.py"
    python3 "$assignment/test_homework.py"
done
