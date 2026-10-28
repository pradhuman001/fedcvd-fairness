#!/bin/bash
while [ ! -f res/DONE ]; do sleep 15; done
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
S10=$(python3 -c "print(','.join(str(42+i) for i in range(10)))")
python3 run_brfss.py eoext capped $S20 res/eoext_capped.csv 150
python3 run_brfss.py eoext natural $S10 res/eoext_natural.csv
echo X > res/DONE2
