# best.pt: what it does, what is code, and what needs data

Weights are unchanged (SHA-1 `94507afb…`; a copy was kept outside the repo before any work).

## Metadata

| | |
|---|---|
| Task / architecture | detect, fine-tuned from `yolov8n.pt` (3,012,018 parameters) |
| Classes | 0 glove, 1 hairnet, 2 maskoff, 3 maskon, 4 no_glove, 5 no_hairnet. These match the code's mapping, so nothing was swapped |
| Training | `/content/dd-1/data.yaml` (Colab), 640 px, 30 epochs, batch 16, `fliplr 0.5`, mosaic 1.0 |
| Stored validation metrics | none (`best_fitness: null`). There is no recorded mAP or per-class recall, and no labelled test set is in the repo |

## Evidence: 132 frames (review_video.mp4 + demo.mp4 + two stills), real predictions

| Class | boxes ≥0.25 | ≥0.50 | p90 conf | Observation |
|---|---:|---:|---:|---|
| glove | 2 | **0** | 0.41 | The one real glove (demo.mp4 #300) scores **0.41**. Positive-glove recall is the main model weakness |
| no_glove | 105 | 67 | 0.69 | Correct on bare hands, including a bare hand at the head while removing a hairnet |
| hairnet | 61 | 41 | 0.70 | True positives 0.62–0.76, but **false positives 0.62–0.63 on bare brown hair** (review #85, #105) |
| no_hairnet | 49 | 24 | 0.62 | Correct after removal (0.79) |
| maskon / maskoff | 25 / 47 | 20 / 25 | 0.70 / 0.58 | Plausible |

- At ≥0.25 there were **18 same-location hairnet/no_hairnet conflicts**: the model emits both classes on one head.
- Vertical box centre within the person box (0 = top): hairnet p90 0.13 and mask p90 0.29. This is the basis
  for the head-band plausibility check.
- An off-person `no_hairnet 0.60` fired on a waste bin. It is ignored because no person contains it.

## What was a code defect (fixed)

| Defect | Effect | Fix |
|---|---|---|
| Model called at conf 0.15, sent as bare strings; the backend's 0.5 filter applied only to scored objects | a 0.16 `no_hairnet` counted as confirmed "missing" | scored `{class, confidence}`; 0.5 decision threshold on both sides |
| Positive and negative both kept; `missing` always won | contradictory output became a violation vote | best per item; if the positive and negative are within 0.10 of each other → **unknown** |
| PPE assigned to every person box containing it ≥50% | one person's item counted for neighbours | one owner (best containment ≥0.6, then smallest box) |
| A "hairnet"/"mask" box at hand height was accepted | not anatomically possible | must lie in the top 35% / 45% of the person box |
| Corridor rule deleted `no_*` labels in the detector | rules hidden in the detector | detector reports evidence; the backend applies per-zone requirements |
| Portrait phone letterboxed into a 640×360 canvas | person ≈⅓ of pixels; sink rule `x>320` covered a black bar | native aspect upload; normalised, content-box-aware coordinates |
| Persons too small for PPE evidence still voted | "not detected" could look like "missing" | height < 120 px → `ppe_reliable=false` → unknown |
| No landmarks → "not touching face" | false negative presented as fact | `face_touch = null` (unknown) |

## What thresholds cannot fix (data limitation)

- **Hairnet false positives on hair** are as confident as true hairnets (0.62 vs 0.62–0.76). No threshold
  separates them. In the production replay, these false positives delay the confirmed `no_hairnet`
  violation from about f13 to f33.
- **Glove recall:** with the 0.5 threshold, "gloves present" is effectively never confirmed. A gloved
  worker in production will show **UNKNOWN, not COMPLIANT**. That is the correct, safe behaviour, but it
  means a green light in the demo is unlikely unless the gloves look like the training data. Lowering the
  glove threshold to about 0.4 would be tuning on a single example. It is available
  (`PPE_CONFIDENCE`, applies to all items) but not recommended without validation data.

## Practical dataset / validation / retraining plan

1. **Validation set first (≈2 h):** capture 300–500 frames from *both* phones in their mounted positions.
   Include 3–5 people, each in every combination: all PPE / each item missing / gloves of every colour in
   use / hairnet vs bare hair (dark, light, tied) / mask on chin / hands near face / 2 people overlapping /
   far and near / back turned. Label in CVAT or Roboflow with the same 6 classes. Keep it **out of
   training**.
2. **Measure the current weights:** `yolo val model=best.pt data=val.yaml` gives per-class P/R and the
   confusion matrix. This is the honest baseline.
3. **Retrain:** add 1,000–2,000 site frames (hard negatives: bare hair, skin-coloured hands, bins, sinks)
   to the original data and fine-tune from `best.pt` at 640 for 50–80 epochs. Keep `fliplr` and turn mosaic
   off for the last 10 epochs. Save the result as `best_v2.pt`; **never overwrite `best.pt`**.
4. **Accept only on evidence:** same validation set, same thresholds. Require glove recall ≥0.8 and a
   hairnet false-positive rate on bare hair ≤5% before switching. Then rerun
   `performance/replay_validation.py` and the benchmark.

Retraining was not done here: there is no labelled site data, and you asked not to incur training costs.
