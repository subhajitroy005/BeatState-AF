# E01v2 oracle decision

Primary metric: `present_class_macro_f1`. Statistical unit: patient after seed aggregation.

## Model Means
- monolithic-32: 0.5461
- factored-32: 0.4637
- random-32: 0.4448

## Routes
- Compression factored-16 minus monolithic-32: mean -0.0125, 95% CI [-0.0929, +0.0582], non-inferior: False; random control at 16: False
- Same-budget 32 superiority versus monolithic and random: False

## VERDICT: STOP_CONFIRMED
