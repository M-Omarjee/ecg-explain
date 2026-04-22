# Model card — ECG-Explain

## Model details

- **Architecture:** 1D ResNet, ~6M parameters, pre-activation residual blocks.
  See [`src/ecg_explain/models/resnet1d.py`](src/ecg_explain/models/resnet1d.py).
- **Input:** 12-lead ECG, 10 seconds at 100 Hz, bandpass-filtered (0.5–40 Hz)
  and z-normalised per lead.
- **Output:** raw logits over 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP).
  Apply sigmoid for per-class probabilities. Multi-label — a single ECG can
  belong to multiple classes.
- **Training:** AdamW, cosine LR schedule, multi-label BCE with class
  pos-weighting, gradient clipping, early stopping on validation macro AUROC.
  Reproducible via [`configs/baseline.yaml`](configs/baseline.yaml).

## Intended use

This model is a **research and educational artefact**. It is intended for:

- Demonstrating an interpretable approach to ECG classification
- Teaching about explainability methods (Grad-CAM) on biomedical signals
- Serving as a baseline for further research

## Out-of-scope use

This model **must not be used for clinical decision-making**. It has not been
validated against the standards required for medical devices (no regulatory
clearance, no prospective evaluation, no real-world deployment data). Clinical
use carries serious risks of patient harm.

## Training data

[PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) — 21,837 clinical
12-lead ECGs from 18,885 patients collected at the Physikalisch-Technische
Bundesanstalt (PTB) between 1989 and 1996. Each record was annotated by up to
two cardiologists using SCP-ECG statements, aggregated for this work into 5
diagnostic superclasses.

**Splits:** the official stratified folds — 1–8 for training, 9 for validation,
10 for test. Patient-level stratification ensures no patient appears in
multiple splits.

## Evaluation

## Evaluation

Headline metric: macro-averaged AUROC across the 5 superclasses on PTB-XL fold 10.

| Class  | AUROC  | F1 (@0.5) |
|--------|--------|-----------|
| NORM   | 0.9388 | 0.8450    |
| MI     | 0.9162 | 0.7084    |
| STTC   | 0.9263 | 0.7230    |
| CD     | 0.9216 | 0.6968    |
| HYP    | 0.8360 | 0.3985    |
| **Macro** | **0.9078** | **0.6744** |

Early stopping selected epoch 8. Training ran for 13 epochs total before
patience-5 early stopping triggered. Full training history in
`checkpoints/baseline/history.json`.

## Known limitations

- **Single dataset.** Trained and tested on PTB-XL only. Performance on data
  from different equipment, populations, or recording conditions is unknown.
- **Demographics.** PTB-XL was collected in a single German centre over 1989–1996,
  which constrains generalisability across age, ancestry, and modern equipment.
- **Class imbalance.** NORM is over-represented; HYP and CD are rarer. Per-class
  AUROC should always be reported alongside the macro number.
- **No rhythm-specific labels.** The 5-superclass formulation collapses
  important distinctions (e.g. NSTEMI vs STEMI within MI).
- **100 Hz sampling.** High-frequency morphology (subtle Q-waves, intra-QRS
  notching) is not resolvable.
- **Grad-CAM is approximate.** Attribution maps suggest *what the model focused
  on*, not what is *clinically diagnostic*. They can be misleading and should
  always be interpreted alongside the underlying ECG.

## Ethical considerations

- **Bias.** Models trained on a single-centre dataset can encode that centre's
  population, equipment, and labelling conventions. Deploying this model in any
  other setting without re-validation would be unsafe.
- **Automation bias.** Tools that present confident outputs without context can
  lead clinicians to over-trust them. The interpretability layer in this project
  is a deliberate counter-measure, not a feature add-on.
- **Data provenance.** PTB-XL is openly licensed for research use. Patient
  consent and ethical approval were obtained at the source institution.

## Authors

[Muhammed Omarjee](https://github.com/M-Omarjee), 2026.
