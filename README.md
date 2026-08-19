# Controlled benchmark for Schatzker classification on AP radiographs

Analysis code and derived data for:

> **Deep Learning-Based Schatzker Classification of Tibial Plateau Fractures on Anteroposterior
> Radiographs: A Controlled Benchmark on a Public Dataset**

This repository accompanies the manuscript. It contains every script used to produce the reported
numbers, the derived per-patient and per-image tables, and the per-seed results behind each figure
and table. It does **not** redistribute the source images; see [Data](#data).

---

## What the study does

A ResNet-50 is fine-tuned to assign Schatzker type from a single anteroposterior knee radiograph,
and four inexpensive controls are run on the identical cross-validation folds to ask how much of the
resulting accuracy is attributable to the fracture in the target bone:

| Control | Question it answers | Script |
|---|---|---|
| 1. Comparator with no pixel content | Is the image needed at all? | `code/pod_exp10_samesplit_baseline.py` |
| 2. Ablation of the target anatomy **and its complement** | Does the score need the tibia? | `code/pod_exp08_seeded.py`, `code/pod_prep03_tibonly.py`, `code/pod_exp17_ablation_pair.py` |
| 3. Regression on the expert mask alone | Does the annotation leak the label? | `code/local_mask_silhouette.py` |
| 4. Augmentation audit | Did the training recipe erase the distinction? | `code/pod_exp08_gpuaug2.py`, `code/pod_exp16_flip_compare.py` |

Two of the four returned findings against readings we had already committed to. Control 2 in
particular is the reason we recommend reporting ablations **in complementary pairs**: erasing the
tibial pixels leaves half the margin above chance, which on its own reads as the bone being unused,
while the complementary input (the tibia with everything else removed) matches the full crop.

---

## Data

The source dataset is **PlaTiF**, distributed under CC BY 4.0:

- Kazemi, A.; Same, K.; Zamanirad, A.; et al. *PlaTiF: tibial plateau fracture dataset* [dataset].
  Zenodo, 2025. <https://doi.org/10.5281/zenodo.18007397>
- Data descriptor: *Sci. Data* **2026**, *13*, 240. <https://doi.org/10.1038/s41597-026-06560-5>

Radiographs and segmentation masks are **not** redistributed here. Download them from Zenodo and
point the preprocessing scripts at the extracted `.mat` files.

`data/` contains only tables derived in this study:

| File | Contents |
|---|---|
| `Table1.csv` | Cohort and acquisition characteristics |
| `Table2.csv` | Balanced accuracy of every configuration on identical folds |
| `Table_S1.csv` | Per-patient characteristics (186 rows) |
| `Table_S2.csv` | Per-image characteristics and mask applicability (421 rows) |
| `Table_S3.csv` | Per-seed results for all ten configurations (50 rows) |
| `Table_S4.csv` | Non-imaging and mask-geometry baselines, per seed (75 rows) |
| `Table_S5.csv` | Pooled per-class recall, precision and confusion |
| `Table_S6.csv` | Data integrity findings and the action taken for each |
| `Supplementary_Tables.xlsx` | All six supplementary tables as one workbook |

`results/` holds the JSON/CSV outputs that the figures and the main text read from directly, so the
reported statistics can be checked without re-running any model.

---

## Reproducing the analysis

### Requirements

Python 3.11 with the packages in `requirements.txt`. The convolutional experiments need a CUDA GPU;
everything else runs on CPU in minutes.

```bash
pip install -r requirements.txt
```

### Order

1. **Preprocess** — `code/pod_prep02_antialias.py` builds the 448 px inputs (`full`, `roi`,
   `center`, `bg`) with an anti-aliasing prefilter; `code/pod_prep03_tibonly.py` adds the
   `tib` input, the exact complement of `bg`.
2. **Train** — `code/pod_exp08_seeded.py` runs the primary configurations (five seeds ×
   five folds each). `code/pod_exp08_gpuaug2.py` is a reimplementation that performs the
   augmentations on the GPU; it reproduces the original protocol to within +0.0001 balanced
   accuracy on the mask-ROI input and was used for the flip and tibia-only runs.
3. **Baselines and controls** — `code/pod_exp10_samesplit_baseline.py` (non-imaging comparator
   on the identical folds), `code/local_mask_silhouette.py` (annotation-only regression),
   `code/pod_exp12_gradcam.py` (attribution).
4. **Statistics** — `code/exp09_proper_stats.py`, `code/local_exp14_baseline_extras.py`,
   `code/local_exp15_cluster_and_recast.py` (patient-cluster bootstrap),
   `code/pod_exp16_flip_compare.py` (augmentation audit),
   `code/pod_exp17_ablation_pair.py` (erasure and its complement).
5. **Tables and figures** — `code/make_supplementary.py`, `code/make_table1.py`,
   `code/make_figures_v2.py`, `code/make_figure4.py`, `code/make_supp_figures.py`.

### Protocol, fixed in advance

Patient-level `StratifiedGroupKFold` (5 folds, `random_state = seed`), five seeds (0–4), ResNet-50
pretrained on ImageNet, AdamW with weight decay 1e-4, one-cycle schedule with max LR 3e-4, batch
size 16, class-weighted cross-entropy with label smoothing 0.05, 25 epochs, mixed precision.
Augmentation: horizontal flip p = 0.5, brightness 0.9–1.1 p = 0.5, translation ±12 px p = 0.3.
No hyperparameter search, no validation split, no early stopping. Predictions are aggregated to the
patient by averaging logits across that patient's images.

---

## Notes for readers

- **Paths are absolute** in most scripts (`/workspace/platif_p4` on the GPU host,
  `/Volumes/.../orthopedic_premed` locally). Edit the `ROOT` / `BASE` constants at the top of each
  file before running.
- **Every script is documented in English.** Each file opens with a docstring stating what it
  measures and why; the tables above map the controls to the scripts that implement them.
- Scripts prefixed `pod_` were run on a GPU host, `local_` on a laptop, `make_` produce manuscript
  assets, `eda_` are exploratory passes retained for provenance.
- Figure rendering requires the Arial font to be installed; the manuscript figures embed it.

## License

Code is released under the MIT License (`LICENSE`). Derived tables in `data/` are released under
CC BY 4.0, matching the licence of the source dataset, which must be cited if you use them.
