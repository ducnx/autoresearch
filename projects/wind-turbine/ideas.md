# Research Ideas: Anomaly Detection in Wind Turbines Using Public SCADA Data

> Last updated: 2026-04-21

---

## 1. Background & Motivation

Wind turbines operate in harsh, highly variable environments, generating massive volumes of SCADA (Supervisory Control and Data Acquisition) data at 10-minute intervals across dozens of sensors — covering temperatures, vibration, power output, pitch angle, yaw error, rotor speed, and more. Unplanned failures are expensive: a single offshore turbine outage can cost tens of thousands of dollars per day. The promise of ML-based anomaly detection is to provide early warnings weeks or months before a physical failure, shifting from reactive to predictive maintenance.

Despite significant research activity, there remain critical open problems:
- **Label scarcity**: real fault events are rare and poorly annotated.
- **Domain shift**: models trained on one turbine/site fail to generalize.
- **Interpretability gap**: operators don't trust black-box alerts.
- **Lack of standardized benchmarks**: hard to compare methods across the literature.
- **High false-positive rates**: alarm fatigue erodes operator trust.

---

## 2. Available Public Datasets

The following publicly available SCADA datasets can be used for evaluation and benchmarking.

| Dataset | Description | Access |
|---|---|---|
| **EDP Open Data** | Multi-year, multi-turbine SCADA + operator logs | [edp.com/open-data](https://www.edp.com/en/innovation/open-data/data) |
| **Kelmarsh Wind Farm** | 6 turbines, ~5 years, UK onshore, SCADA + events | [zenodo.org/record/5841834](https://zenodo.org/record/5841834) |
| **Penmanshiel Wind Farm** | 14 turbines, UK onshore, SCADA + logs | [zenodo.org/record/5946808](https://zenodo.org/record/5946808) |
| **Ørsted Anholt Offshore** | Large offshore farm, SCADA operational data | [orsted.com/data](https://orsted.com/en/our-business/offshore-wind/offshore-operational-data) |
| **Ørsted Westermost Rough** | Offshore, SCADA operational data | Same as above |
| **CAREtoCompare (B & C)** | Benchmark dataset for comparing anomaly detection algorithms | [zenodo.org/10958774](https://data.niaid.nih.gov/resources?id=zenodo_10958774) |
| **DSforWind (1a/1b/2a/2b)** | Multi-site, time-series SCADA records | [zenodo.org/5516552](https://zenodo.org/records/5516552) |
| **SMARTEOLE** | Farm-level SCADA + wake interaction data | [zenodo.org/7342466](https://zenodo.org/records/7342466) |
| **Engie La Haute Borne** | 4 turbines, 5+ years, open license | [opendata-renewables.engie.com](https://opendata-renewables.engie.com/) |
| **Kaggle Wind Turbine SCADA** | Single-turbine SCADA (simple but popular) | [kaggle.com](https://www.kaggle.com/berkerisen/wind-turbine-scada-dataset) |
| **Norrekaer Windfarm (DTU)** | Danish farm, academic quality | [data.dtu.dk](https://data.dtu.dk/articles/dataset/SCADA_data_from_Norre_m2_wind_farm/19076756) |
| **Hill of Towie Open Dataset** | Recent (2024), labeled events | [zenodo.org/14870023](https://zenodo.org/records/14870023) |
| **Levenmouth Turbine (ORE)** | Offshore demonstration turbine, 10-min SCADA | [pod.ore.catapult.org.uk](https://pod.ore.catapult.org.uk/data-collection/ldt-turbine-scada-10m) |

> A curated index with data loaders: [sltzgs/OpenWindSCADA](https://github.com/sltzgs/OpenWindSCADA)

---

## 3. State-of-the-Art Methods

### 3.1 Unsupervised / Reconstruction-Based Methods
- **Autoencoders (AE/LSTM-AE/VAE)**: Train on "normal" data; flag anomalies by high reconstruction error. LSTM variants capture temporal dependencies. VAEs add uncertainty quantification.
- **Isolation Forest (IF)**: Computationally efficient, fast to train, good for high-frequency outlier detection. Commonly used as a baseline and for hybrid pipelines.
- **DBSCAN / Mahalanobis Distance**: Statistical clustering-based methods, effective on low-dimensional features.

### 3.2 Deep Learning — Sequential & Temporal
- **Temporal Convolutional Networks (TCN)**: Dilated 1D convolutions for long-range dependencies with fixed receptive fields.
- **Transformer Autoencoders**: Self-attention for reconstruction-based anomaly detection; attention weights offer interpretability. Effective for multi-sensor time series.
- **Diffusion Models**: Emerging for high-fidelity "normal" signal generation; novelty as a generative model for anomaly scoring.

### 3.3 Graph-Based Spatiotemporal Models
- **Spatial-Temporal GNNs (ST-GNNs)**: Model sensors as graph nodes; edges = inter-sensor correlation. Graph Attention Networks (GAT) + LSTM/GRU or TCN for temporal modeling. Significantly outperforms non-graph baselines for multi-sensor SCADA.
- **SGG-DGCN**: Similarity graph generation combined with deep GCN for complex sensor coupling.
- **Causal Attention GNNs**: Use causal inference to identify which signals actually cause deviations, reducing false positives.

### 3.4 Transfer Learning & Domain Adaptation
- **Autoencoder fine-tuning**: Train on source turbine, adapt to target by fine-tuning decoder or threshold.
- **Anomaly-Space meta-models**: Map anomaly scores from multiple components into a shared feature space; train classifier across domains.
- **DMRACNN**: Deep Multi-scale Residual Attention CNN with domain adaptation for cross-turbine generalization.

### 3.5 Hybrid Physics–Data Models
- **Physics-Informed Neural Networks (PINNs)**: Embed physical constraints (power curve, Betz limit, aerodynamics) into the neural network loss function.
- **Residual learning**: Fit a physics model (e.g., power curve) then train ML on the physics residuals to detect anomalies.

### 3.6 Explainability (XAI)
- **SHAP**: Post-hoc feature attribution on anomaly scores to identify which sensors triggered alerts.
- **Attention weights**: Transformer attention maps for temporal and sensor-level interpretability.
- **Conformal prediction**: Uncertainty-bounded anomaly scores for principled threshold setting.

### 3.7 Self-Supervised & Contrastive Learning
- **Contrastive pretraining**: Learn normal operational representations without labels; anomalies surface as out-of-distribution encodings.
- **Masked autoencoding (MAE-style)**: Mask SCADA channels and predict them; reconstruction failure signals faults.

---

## 4. Research Ideas

---

### Idea 1: Cross-Turbine Self-Supervised Pretraining with Contrastive Learning

**Problem**: Each turbine requires its own labeled data for supervised learning. Labels are scarce and expensive. Models trained on one turbine rarely generalize.

**Proposed Approach**:
1. Pretrain a shared encoder (e.g., Transformer or ST-GNN) on multiple turbines from public datasets (Kelmarsh, Penmanshiel, EDP) using contrastive learning. Positive pairs = augmented views of same turbine time window; negative pairs = different turbines or different operating regimes.
2. Fine-tune the shared encoder on a target turbine using only normal-operation data (no fault labels needed).
3. Use reconstruction error or distribution-distance threshold for anomaly scoring.

**Research Questions**:
- Does multi-source contrastive pretraining generalize better than per-turbine training?
- What augmentation strategies work best for SCADA time series (jitter, masking, subsample)?
- Can a single pretrained encoder achieve comparable performance across turbine makes/models?

**Key Novelty**: Application of SimCLR/BYOL-style pretraining paradigm to multivariate industrial time series with explicit cross-turbine generalization objective.

**Datasets**: Kelmarsh + Penmanshiel (same operator, different turbine configurations), EDP open data.

**Baselines**: Per-turbine LSTM-AE, Isolation Forest, VAE.

---

### Idea 2: Causal Sensor Graph for Fault Localization in SCADA

**Problem**: Existing graph-based models use correlation graphs (symmetric), which cannot distinguish whether signal A causes signal B or vice versa. This confounds fault localization.

**Proposed Approach**:
1. Learn a directed causal graph (DAG) from SCADA signals using **Granger causality** or **NOTEARS/DAG-GNN** structure learning methods on normal operating data.
2. Integrate the learned causal graph into a **Causal GNN** architecture for anomaly detection and root-cause attribution.
3. When an anomaly is detected, trace the causal path in the graph to identify the most likely faulty subsystem.

**Research Questions**:
- Can causal graphs learned from SCADA data correspond to known physical subsystem relationships (gearbox → bearing temperature)?
- Does causal structure learning improve anomaly detection accuracy vs. undirected correlation graphs?
- Is the learned causal graph stable across seasons and operational regimes?

**Key Novelty**: First systematic application of differentiable causal structure learning (NOTEARS) to wind turbine SCADA for both anomaly detection and fault localization.

**Datasets**: EDP open data (has operator annotations for ground truth fault labeling).

**Baselines**: GAT with undirected correlation graph, LSTM-AE, SGG-DGCN.

---

### Idea 3: Physics-Residual Masked Autoencoder (PR-MAE)

**Problem**: Standard autoencoders learn arbitrary latent representations. Without physical constraints, they may learn shortcuts and fail to detect subtle physical anomalies. Physics-first models are too rigid for real noise.

**Proposed Approach**:
1. Fit a **physics-based normal behavior model** (power curve + temperature model) from SCADA data using a parametric or GPR model.
2. Compute **physics residuals** (observed – expected) as a normalized feature space.
3. Train a **Masked Autoencoder (MAE)** on the residual space: randomly mask a fraction of sensor channels and reconstruct them. Anomaly score = reconstruction error of the residuals.
4. This constrains the model to focus on deviations from known physics, not raw signal values.

**Research Questions**:
- Does working in residual space reduce false positives caused by environmental variability (wind speed changes)?
- Can the MAE-on-residuals approach detect early-stage bearing degradation that raw signal AEs miss?
- How sensitive is the approach to inaccuracies in the physics model (e.g., yaw misalignment bias)?

**Key Novelty**: Fusion of physics-based residual modeling with modern masked autoencoding in a principled two-stage pipeline for interpretable, physically consistent anomaly detection.

**Datasets**: Engie La Haute Borne (high-quality, multi-year; known power curve references), Kelmarsh.

**Metrics**: Precision-Recall AUC, detection lead time, false positive rate.

---

### Idea 4: Multi-Farm Benchmark with Standardized Evaluation Protocol

**Problem**: Published anomaly detection results are not comparable because: (a) different datasets, (b) different anomaly definitions, (c) different evaluation metrics. The literature cannot converge on what works.

**Proposed Approach**:
1. Curate a standardized benchmark from the top-4 richest public SCADA datasets (EDP, Kelmarsh, Penmanshiel, Engie La Haute Borne) with unified preprocessing, consistent anomaly labeling (sourced from operator logs + community annotations from OpenWindSCADA).
2. Define a standard evaluation protocol: point-adjust precision/recall, detection lead time in days, false positive rate, computational cost.
3. Implement and evaluate 8–10 representative algorithms (IF, LSTM-AE, VAE, Transformer-AE, GAT-LSTM, CARE-based, physics-residual).
4. Release benchmark code and results publicly to enable reproducible comparisons.

**Research Questions**:
- Which family of methods (reconstruction-based, density-based, graph-based) achieves best precision-recall tradeoff across diverse datasets?
- Is there a single method that generalizes well, or is performance strongly site-specific?
- How much does data quality (missing values, sensor drift) degrade each method?

**Key Novelty**: Unlike existing dataset papers, this provides a comprehensive, reproducible evaluation of detection methods, not just data access. Directly extends the CAREtoCompare approach to more datasets and more methods.

**Impact**: High — a benchmark paper with code is likely to be highly cited and adopted as the standard reference in the field.

---

### Idea 5: Lightweight Anomaly Foundation Model for Wind SCADA (WSCADA-FM)

**Problem**: Large pretrained time-series foundation models (e.g., TimesFM, Chronos, Moirai) show promise but are (a) trained on general corpora with no wind-specific inductive biases, and (b) too heavy for edge deployment in SCADA systems.

**Proposed Approach**:
1. Pretrain a **compact Transformer encoder** (e.g., PatchTST-style) on the entire corpus of public SCADA datasets (~10+ farms, ~50+ turbine-years).
2. Incorporate domain-specific inductive biases: diurnal/seasonal positional encodings, wind speed conditioning, turbine metadata tokens.
3. Fine-tune the foundation model on downstream tasks: anomaly scoring (reconstruction), fault classification, remaining useful life estimation.
4. Evaluate on held-out farms for zero-shot and few-shot generalization.

**Research Questions**:
- Does pretraining on wind-specific data outperform fine-tuning a general-purpose time series foundation model?
- What is the minimal model size that achieves competitive anomaly detection while being deployable on edge SCADA hardware?
- Can the foundation model encode operational regime context (curtailment, high-wind shutdown) to reduce false positives?

**Key Novelty**: First domain-specific compact foundation model for wind turbine SCADA, bridging recent foundation model research and industrial deployment constraints.

**Datasets**: All public datasets above for pretraining; EDP / Kelmarsh for evaluation.

---

### Idea 6: Uncertainty-Aware Anomaly Scoring with Conformal Prediction

**Problem**: Most models output a scalar anomaly score without uncertainty bounds. Operators have no principled way to set alert thresholds, leading to high false positive or false negative rates.

**Proposed Approach**:
1. Train a **VAE or Normalizing Flow** on normal SCADA data.
2. Apply **split conformal prediction** to calibrate anomaly score thresholds on a held-out calibration set of normal data.
3. At inference, provide **prediction intervals** on anomaly scores with statistical coverage guarantees (e.g., 95% confidence that the sample is abnormal before alerting).
4. Integrate with an **adaptive threshold** that adjusts for seasonal shifts in SCADA signal distributions.

**Research Questions**:
- Can conformal prediction reduce the false positive rate while maintaining detection sensitivity?
- How does the adaptive threshold behave across seasons (summer/winter temperature differences)?
- Can uncertainty estimates help operators prioritize alerts (high-confidence vs. borderline anomalies)?

**Key Novelty**: Rigorous statistical framework for threshold calibration in wind turbine anomaly detection — replacing ad-hoc tuning with coverage-guaranteed conformal intervals.

**Datasets**: Kelmarsh (long, multi-year), Engie La Haute Borne.

---

## 5. Research Gaps Summary

| Gap | Severity | Relevant Idea |
|---|---|---|
| No standardized benchmark across multiple farms | High | Idea 4 |
| Models don't generalize across turbine types/sites | High | Idea 1, 5 |
| Black-box alerts not actionable for operators | High | Idea 2 (causal), Idea 6 (uncertainty) |
| Purely data-driven models ignore physics | Medium | Idea 3 |
| No domain-specific pretrained foundation model | Medium | Idea 5 |
| False positive rate too high for industrial adoption | High | Idea 6, Idea 3 |
| Lack of fault localization (detect vs. diagnose) | Medium | Idea 2 |

---

## 6. Recommended Starting Points

Given the available compute and dataset accessibility, the following ideas are best to start with:

1. **Idea 4 (Benchmark)** — Relatively straightforward to implement; high impact; directly enables other ideas.
2. **Idea 3 (PR-MAE)** — Conceptually novel, incremental over existing work, uses widely available Engie + Kelmarsh datasets.
3. **Idea 1 (Cross-Turbine SSL)** — More ambitious; high novelty if results validate cross-farm generalization.

---

## 7. Key References

- Leszek et al. (2022). *OpenWindSCADA: A curated list of open wind turbine datasets.* Wind Energy. [doi.org/10.1002/we.2766](https://onlinelibrary.wiley.com/doi/full/10.1002/we.2766)
- Astolfi et al. (2024). *CAREtoCompare: A benchmark dataset for wind turbine anomaly detection algorithms.* Nature Scientific Data. [doi.org/10.1038/s41597-024-03067-9](https://www.nature.com/articles/s41597-024-03067-9)
- Various (2024). *Spatial-Temporal GNNs for SCADA anomaly detection.* MDPI Sensors / Energies.
- Various (2024). *Physics-Informed Neural Networks for wind turbine condition monitoring.* arXiv.
- Various (2024). *Cross-turbine domain adaptation via anomaly-space meta-models.* arXiv.
- McConville et al. (2021). *SMARTEOLE wind farm dataset.* Wind Energy Science. [doi.org/10.5194/wes-6-1427-2021](https://doi.org/10.5194/wes-6-1427-2021)

---

*Generated by literature review covering 2022–2025 publications. Priority and novelty assessments are as of April 2026.*
