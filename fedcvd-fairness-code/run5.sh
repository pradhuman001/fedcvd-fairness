#!/bin/bash
export CONV=1 PREDS=res3/preds5
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
S10=$(python3 -c "print(','.join(str(42+i) for i in range(10)))")
python3 converge.py; echo converge >> res3/PROGRESS5
python3 run2.py fine capped $S20 res3/fine_capped.csv 150; echo fine_capped >> res3/PROGRESS5
python3 run2.py glob dir $S10 res3/glob_dir.csv 150; python3 run2.py glob dir03 $S10 res3/glob_dir03.csv 150; echo glob >> res3/PROGRESS5
python3 run2.py fine natural $S10 res3/fine_natural.csv 150; echo fine_natural >> res3/PROGRESS5
echo X > res3/DONE5
