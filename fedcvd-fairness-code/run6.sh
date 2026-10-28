#!/bin/bash
export CONV=2 PREDS=res4/preds
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
S10=$(python3 -c "print(','.join(str(42+i) for i in range(10)))")
S5=$(python3 -c "print(','.join(str(42+i) for i in range(5)))")
for suite in core controls; do python3 run_brfss.py $suite capped $S20 res4/${suite}_capped.csv 150; python3 run_brfss.py $suite natural $S10 res4/${suite}_natural.csv; echo $suite >> res4/PROGRESS; done
for suite in central post lso; do python3 run2.py $suite capped $S20 res4/${suite}_capped.csv 150; python3 run2.py $suite natural $S10 res4/${suite}_natural.csv; echo $suite >> res4/PROGRESS; done
python3 run2.py fine capped $S20 res4/fine_capped.csv 150; echo fine_capped >> res4/PROGRESS
python3 run2.py glob dir $S10 res4/glob_dir.csv 150; python3 run2.py glob dir03 $S10 res4/glob_dir03.csv 150; echo glob >> res4/PROGRESS
python3 run2.py fedsub dir $S10 res4/fedsub_dir.csv 150; python3 run2.py centsub dir $S10 res4/centsub_dir.csv 150; python3 run2.py fedsub dir03 $S10 res4/fedsub_dir03.csv 150; python3 run2.py centsub dir03 $S10 res4/centsub_dir03.csv 150; echo dirsub >> res4/PROGRESS
python3 run2.py fine natural $S5 res4/fine_natural.csv 150; echo fine_natural >> res4/PROGRESS
for suite in agecut; do python3 run_brfss.py $suite capped $S20 res4/${suite}_capped.csv 150; python3 run_brfss.py $suite natural $S10 res4/${suite}_natural.csv; echo $suite >> res4/PROGRESS; done
for cap in 75 300; do python3 run_brfss.py capabl capped $S20 res4/capabl_${cap}.csv $cap; done; echo capabl >> res4/PROGRESS
echo X > res4/DONE
