#!/bin/bash
cd "$(dirname "$0")"
python3 exp_centered_eb_table1.py "$@" > results/exp_centered_eb_table1.log 2>&1
echo "DONE: exit code $?" >> results/exp_centered_eb_table1.log
