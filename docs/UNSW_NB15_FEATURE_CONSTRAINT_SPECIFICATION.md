# UNSW-NB15 Feature and Constraint Specification

## 1. Purpose

This document defines the feature-handling and adversarial-constraint policy used for the UNSW-NB15 experiments in the study:

"Evaluating the Adversarial Robustness and Trustworthiness of Machine-Learning-Based Network Intrusion Detection under Realistic Attack Constraints"

The specification distinguishes between the prepared 45-column UNSW-NB15 train/test representation used in the experiments and the original 49-feature dataset documentation.

The prepared CSV schema is treated as the authoritative implementation schema.

---

## 2. Dataset Representation

The prepared UNSW-NB15 training and testing files contain 45 columns.

Training set:
- 175,341 records
- 45 columns

Testing set:
- 82,332 records
- 45 columns

The prepared representation does not contain the original:
- srcip
- sport
- dstip
- dsport
- Stime
- Ltime

Several remaining variables use abbreviated or case-normalized names relative to the original feature documentation.

Examples include:
- spkts / Spkts
- dpkts / Dpkts
- sload / Sload
- dload / Dload
- smean / smeansz
- dmean / dmeansz
- sinpkt / Sintpkt
- dinpkt / Dintpkt
- sjit / Sjit
- djit / Djit
- response_body_len / res_bdy_len
- ct_src_ltm / ct_src_ ltm
- label / Label

No renaming is performed for the experiments solely to reproduce the original documentation names.

---

## 3. Columns Excluded From Model Inputs

### 3.1 Identifier

`id`

Reason:
- Identifier rather than a network-behavioural feature.
- Excluding it reduces the possibility of learning record-order or dataset-specific artefacts.

### 3.2 Target variables

`label`
- Primary binary target.
- 0 = normal.
- 1 = attack.

`attack_cat`
- Attack-category label.
- Excluded from predictive model input.
- Retained for secondary attack-category analysis.

---

## 4. Categorical Features

The prepared dataset contains:

- `proto`
- `service`
- `state`

These variables are encoded for machine-learning models as required by the selected preprocessing pipeline.

Encoded numeric representations must not be interpreted as continuous semantic quantities during adversarial manipulation.

For constraint-aware attacks:
- protocol values must remain valid protocol categories;
- service values must remain valid service categories;
- state values must remain valid state categories;
- arbitrary interpolation between encoded category values is not considered a valid categorical manipulation.

---

## 5. Binary and Semantic Features

### 5.1 is_sm_ips_ports

Definition:
Indicates whether source and destination IP addresses and ports satisfy the equality condition represented by this feature.

Constraint policy:
- Valid values are restricted to {0, 1}.
- It is treated as a derived/semantic feature.
- It is not independently manipulated in the constraint-aware attack.

### 5.2 is_ftp_login

Definition:
Indicates whether an FTP session is accessed using user/password authentication.

Constraint policy:
- Valid values are restricted to the valid binary representation.
- Modification is permitted only where the corresponding service/context is compatible with FTP activity.
- It is not freely manipulated as an independent continuous variable.

---

## 6. Contextual and Derived Features

The following features are treated as contextual or derived:

- `ct_srv_src`
- `ct_srv_dst`
- `ct_dst_ltm`
- `ct_src_ltm`
- `ct_src_dport_ltm`
- `ct_dst_sport_ltm`
- `ct_dst_src_ltm`
- `ct_state_ttl`
- `ct_flw_http_mthd`
- `ct_ftp_cmd`

These features represent counts, relationships, or contextual properties derived from network activity and observation windows.

Constraint-aware policy:
- They are not independently perturbed.
- Their values are retained unless a future validated traffic-generation procedure can recompute them consistently from modified underlying activity.
- Independent manipulation is considered potentially inconsistent with the traffic-generating process.

This policy is an experimental approximation of feature dependency rather than a claim that these features are physically immutable in every operational environment.

---

## 7. Numerical Traffic Features

The following classes of features are treated as numerical traffic characteristics:

### Traffic volume and packet statistics

- `spkts`
- `dpkts`
- `sbytes`
- `dbytes`
- `sloss`
- `dloss`

### Timing and duration

- `dur`
- `sinpkt`
- `dinpkt`
- `sjit`
- `djit`
- `tcprtt`
- `synack`
- `ackdat`

### Traffic rates

- `rate`
- `sload`
- `dload`

### TTL and TCP characteristics

- `sttl`
- `dttl`
- `swin`
- `dwin`
- `stcpb`
- `dtcpb`

### Packet-size statistics

- `smean`
- `dmean`

### Application/content characteristics

- `trans_depth`
- `response_body_len`

For constraint-aware attacks, numerical features must remain within valid dataset-supported ranges and must satisfy any additional dependency rules defined by the attack procedure.

---

## 8. Dataset-Supported Numerical Ranges

The observed training-data ranges are recorded in:

`results/logs/unsw_nb15_audit.json`

These observed ranges are used as empirical validity bounds where appropriate.

Important distinction:

An observed minimum/maximum is a dataset-supported bound; it is not automatically equivalent to a physical network protocol limit.

Therefore, dataset-derived ranges must not be described as universal network constraints.

---

## 9. Conventional Feature-Space Attack

The conventional attack condition is intended to provide a less restrictive feature-space baseline.

The attack may manipulate eligible numerical feature representations within a predefined attack budget.

However:
- target columns are never modified;
- identifiers are never used;
- categorical variables are not treated as arbitrary continuous variables;
- generated samples must remain numerically valid;
- attack parameters are fixed before evaluation.

The purpose of this condition is to establish a conventional feature-space robustness baseline against which constraint-aware results can be compared.

---

## 10. Constraint-Aware Attack

The constraint-aware condition introduces progressively stronger restrictions.

Candidate adversarial samples must satisfy:

1. Valid numerical ranges.
2. Valid categorical values.
3. Valid binary values.
4. Restrictions on contextual/derived features.
5. Feature-dependency requirements where these can be justified from the available dataset representation.
6. Attack-specific semantic restrictions where applicable.

Invalid candidate perturbations are rejected or projected according to the predefined attack implementation.

Constraint violations and rejected candidates are recorded rather than silently discarded.

---

## 11. Features Unavailable For Constraint Evaluation

The following original UNSW-NB15 variables are absent from the prepared 45-column representation:

- `srcip`
- `sport`
- `dstip`
- `dsport`
- `Stime`
- `Ltime`

Consequently, constraints requiring direct access to these variables cannot be evaluated in the main prepared-data experiment.

This limitation must be reported explicitly in the manuscript.

---

## 12. Interpretation of Realism

The constraint-aware attack condition should be described as:

"constraint-aware feature-space evaluation"

rather than as a fully realistic host-space attack.

The experiment does not directly reproduce attacker actions on a live host or network.

A host-space experiment may be considered as an optional extension if sufficient time and validation resources are available.

---

## 13. Reproducibility

The following must be recorded for every adversarial experiment:

- dataset
- dataset split
- model
- random seed
- attack algorithm
- attack objective
- perturbation budget
- step size
- number of iterations
- modified feature set
- categorical handling
- constraint definitions
- constraint enforcement method
- number of rejected candidates
- attack success rate
- perturbation magnitude
- modified feature count
- model confidence
- calibration metrics

No attack result will be altered or selectively removed because it produces an undesirable outcome.

---

## 14. Status

Status: FROZEN FOR IMPLEMENTATION

This specification may only be changed if:
1. new empirical evidence demonstrates that a constraint is incorrect;
2. a reproducibility problem requires correction;
3. an implementation limitation makes a planned procedure infeasible.

Any change must be recorded in the project decision log.
