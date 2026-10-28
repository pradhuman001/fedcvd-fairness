#!/bin/bash
S20=$(python3 -c "print(','.join(str(42+i) for i in range(20)))")
for cap in 75 300; do python3 run_brfss.py capabl capped $S20 res/capabl_${cap}.csv $cap; done
echo X > res/DONE3
