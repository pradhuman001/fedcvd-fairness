#!/bin/bash
export CONV=2 PREDS=res4/preds
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
python3 run2.py centdense natural $S20 res4/centdense_natural.csv 150 > res4/log13a.txt 2>&1; echo centdense >> res4/PROGRESS13
