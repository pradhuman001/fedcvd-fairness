#!/bin/bash
export CONV=1 PREDS=res3/preds
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
S10=$(python3 -c "print(','.join(str(42+i) for i in range(10)))")
for suite in core controls; do python3 run_brfss.py $suite capped $S20 res3/${suite}_capped.csv 150; python3 run_brfss.py $suite natural $S10 res3/${suite}_natural.csv; echo $suite >> res3/PROGRESS; done
for suite in central post lso; do python3 run2.py $suite capped $S20 res3/${suite}_capped.csv 150; python3 run2.py $suite natural $S10 res3/${suite}_natural.csv; echo $suite >> res3/PROGRESS; done
for suite in agecut; do python3 run_brfss.py $suite capped $S20 res3/${suite}_capped.csv 150; python3 run_brfss.py $suite natural $S10 res3/${suite}_natural.csv; echo $suite >> res3/PROGRESS; done
for cap in 75 300; do python3 run_brfss.py capabl capped $S20 res3/capabl_${cap}.csv $cap; done; echo capabl >> res3/PROGRESS
echo X > res3/DONE
