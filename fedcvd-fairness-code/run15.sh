#!/bin/bash
export CONV=2
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
python3 run_brfss.py dense05 capped $S20 res4/dense05_capped.csv 150 > res4/log15a.txt 2>&1; echo dense05 >> res4/PROGRESS15
