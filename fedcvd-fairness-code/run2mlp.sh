#!/bin/bash
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
S10=$(python3 -c "print(','.join(str(42+i) for i in range(10)))")
python3 run2.py mlp50 capped $S20 res2/mlp50_capped.csv 150
python3 run2.py mlp50 natural $S10 res2/mlp50_natural.csv
echo X > res2/DONE
