# Metric Policy V2

E01v2 uses `present_class_macro_f1` as the primary patient-level metric.

For each patient, F1 is computed only for classes present in that patient's
reference labels, then averaged. This prevents AF-only or non-AF-only records
from receiving an artificial zero for an absent class.

Secondary fields are still reported:

- `af_f1`: AF-class F1 when the patient has AF reference beats; undefined
  otherwise.
- `sensitivity`: reported when AF reference beats exist.
- `ppv`: reported when AF predictions exist.
- `specificity`: reported when non-AF reference beats exist.
- `patient_has_af` and `patient_has_nonaf`: explicit patient composition flags.

All confirmatory paired tests aggregate model seeds within patient before
computing patient differences.
