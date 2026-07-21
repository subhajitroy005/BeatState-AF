# E01v2 deployable decision

Primary metric: `present_class_macro_f1`. Statistical unit: patient after seed aggregation.

## Model Means
- monolithic-32: 0.1946
- factored-32: 0.1956
- random-32: 0.1923

## Routes
- Compression factored-16 minus monolithic-32: mean +0.0021, 95% CI [-0.0013, +0.0055], non-inferior: True; random control at 16: False
- Same-budget 32 superiority versus monolithic and random: False

## VERDICT: STOP_CONFIRMED
