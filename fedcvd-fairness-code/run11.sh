#!/bin/bash
export CONV=2
python3 run_brfss.py dense natural $(python3 -c "print(','.join(str(42+i) for i in range(20)))") res4/dense_natural.csv > res4/log11a.txt 2>&1; echo dense >> res4/PROGRESS11
