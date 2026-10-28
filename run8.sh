#!/bin/bash
export CONV=2 PREDS=res4/preds
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
S10=$(python3 -c "print(','.join(str(42+i) for i in range(10)))")
SB=$(python3 -c "print(','.join(str(52+i) for i in range(10)))")
python3 run2.py eopost capped $S20 res4/eopost_capped.csv 150; echo eopost_capped >> res4/PROGRESS8
python3 run2.py eopost natural $S10 res4/eopost_natural.csv 150; echo eopost_natural >> res4/PROGRESS8
python3 convcheck2.py > res4/convcheck2.log 2>&1; echo convcheck2 >> res4/PROGRESS8
python3 run_brfss.py core natural $SB res4/core_natural_b.csv; echo core_nat_b >> res4/PROGRESS8
python3 run2.py central natural $SB res4/central_natural_b.csv 150; echo central_nat_b >> res4/PROGRESS8
python3 run_brfss.py controls natural $SB res4/controls_natural_b.csv; echo controls_nat_b >> res4/PROGRESS8
echo X > res4/DONE8
