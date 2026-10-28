#!/bin/bash
export CONV=1 PREDS=res3/preds
S10=$(python3 -c "print(','.join(str(42+i) for i in range(10)))")
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
for sc in dir03 dir; do for suite in fedsub centsub; do python3 run2.py $suite $sc $S10 res3/${suite}_${sc}.csv 150; done; echo $sc >> res3/PROGRESS4; done
python3 eg.py capped $S20 res3/eg_capped.csv 150; echo egcapped >> res3/PROGRESS4
python3 eg.py natural $S10 res3/eg_natural.csv 150; echo egnat >> res3/PROGRESS4
echo X > res3/DONE4
