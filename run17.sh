#!/bin/bash
export CONV=2
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
python3 run2.py centdense capped $S20 res4/centdense_capped.csv 150 > res4/log17a.txt 2>&1; echo done >> res4/PROGRESS17
