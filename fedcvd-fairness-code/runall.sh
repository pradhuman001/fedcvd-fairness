#!/bin/bash
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
S10=$(python3 -c "print(','.join(str(42+i) for i in range(10)))")
for suite in core controls eo mlp agecut; do
  python3 run_brfss.py $suite capped $S20 res/${suite}_capped.csv 150
done
for suite in core controls eo mlp agecut; do
  python3 run_brfss.py $suite natural $S10 res/${suite}_natural.csv
done
echo ALLDONE > res/DONE
