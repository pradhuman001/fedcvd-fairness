#!/bin/bash
export CONV=2 PREDS=res4/preds_mlp
mkdir -p res4/preds_mlp
S10=$(python3 -c "print(','.join(str(42+i) for i in range(10)))")
S5=$(python3 -c "print(','.join(str(42+i) for i in range(5)))")
python3 run2.py mlp400 capped $S10 res4/mlp400_capped.csv 150; echo mlp_capped >> res4/PROGRESS7
python3 run2.py mlp400 natural $S5 res4/mlp400_natural.csv 150; echo mlp_natural >> res4/PROGRESS7
echo X > res4/DONE7
