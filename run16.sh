#!/bin/bash
export CONV=2 NSEED=20
python3 agegap.py capped > res4/agegap_capped20.log 2>&1; echo agegap20 >> res4/PROGRESS16
