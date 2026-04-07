#!/bin/bash
nohup python eval_model.py -da UOB -gi 3 > UOB_out.log 2>&1 &
nohup python eval_model.py -da USPTO -gi 5 > USPTO_out.log 2>&1 &
nohup python eval_model.py -da CLEF -gi 0 > CLEF_out.log 2>&1 &
nohup python eval_model.py -da staker -gi 4  > staker_out.log 2>&1 &
# nohup python eval_model.py -da JPO -gi 1 > JPO_out.log 2>&1 &
# nohup python eval_model.py -da acs -gi 6 > acs_out.log 2>&1 &

#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
#/recovery/bo/pys/i2m/output/custom_20260325