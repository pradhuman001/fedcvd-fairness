#!/bin/bash
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
S10=$(python3 -c "print(','.join(str(42+i) for i in range(10)))")
python3 run2.py central natural $S10 res2/central_natural.csv
for suite in post lso mlp50; do
  python3 run2.py $suite capped $S20 res2/${suite}_capped.csv 150
  if [ $suite = lso ]; then SN=$S20; else SN=$S10; fi
  python3 run2.py $suite natural $SN res2/${suite}_natural.csv
done
echo X > res2/DONE
