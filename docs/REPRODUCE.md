# Reproducing the results

```bash
pip install -r requirements.txt
```

## 1. Build the dataset

```bash
python run_all.py                       # harvest → mine → synthesise → assemble → render → validate → export
python build/10_bias_audit.py           # eleven-test shortcut audit → dataset/BIAS_AUDIT.md
```

`run_all.py` runs `build/01_harvest.py` … `build/08_export.py` in order. Rendering for
version 2 is `build/06_render.py --clean`.

## 2. Extra renders

```bash
python build/12_extra_real.py                                                  # supplementary real pool
python build/06_render.py --clean --records cache/records_extra.jsonl --out images_extra
python build/11_render_plotly.py                                               # Plotly, real test charts
python build/13_render_vega.py --workers 4                                     # Vega-Lite, real test charts
python build/11_render_plotly.py --split train --workers 6                     # Plotly, training charts
python build/11_render_plotly.py --records cache/records_extra.jsonl --workers 6
python build/06_render.py --clean --randstyle --split train --out images_mplrand
```

## 3. Train (Kaggle, 2× T4)

```bash
python train/make_split_notebooks.py    # writes notebooks/*.ipynb
```

| Notebook | Trains / evaluates |
|---|---|
| `nb5_real` | ResNet-101 and ResNet-50 on the balanced split |
| `nb5x_r101`, `nb5x_r50`, `nb5x_d121` | + supplementary real pool |
| `nb7_r101`, `nb7_d121` | + renderer augmentation (deployed members) |
| `nb6_zeroshot`, `nb6b_zeroshot_ts` | zero-shot Qwen VLM baselines |
| `nb4_vlm`, `nb4v2_vlm` | Qwen2-VL-2B + QLoRA generative model (version-1 / version-2 renders) |
| `nb7_{r101,d121}_s{2,3}` | extra training seeds of the deployed members (`train/seed_variance.py`) |

Each notebook writes `<tag>_best.pt` and `<tag>_probs.npz` (validation and test
probabilities). Put checkpoints under `results/nb5/`, `results/nb5x/<arch>/`,
`results/nb7/<arch>/` and probabilities under `results/probs/<tag>.npz`.

## 4. Evaluate

```bash
python train/cache_probs.py --view plotly    # Plotly-render probabilities for each member
python train/cache_probs.py --view vega      # Vega-Lite-render probabilities
python train/final_ensemble.py               # → results/final_ensemble.json
python train/stats_report.py                 # → results/stats_report.json, stats_table.tex
python train/grounding_eval.py               # → results/grounding.json
python train/ablate_marks.py                 # rendering-leak ablation → results/ablate_marks.csv
python paper/make_figures.py                 # → paper/figures/
```

The probabilities for every deployed and ablation member are already in `results/probs/`, so
step 4's ensemble and statistics run without retraining (they need `dataset/tables/` for
the labels).

## 5. Demo

```bash
python deploy/export_onnx.py     # → deploy/web/insightvlm_ensemble.onnx
python deploy/verify_onnx.py     # ONNX must match the evaluated predictions (4,000/4,000)
cd deploy/web && python -m http.server 8000
```

`app.py` is the equivalent local Gradio app (needs the checkpoints and `dataset/`).

## Where each number comes from

| Result | File |
|---|---|
| Main table, CIs, McNemar | `results/stats_report.json` |
| Deployed ensemble and alternatives | `results/final_ensemble.json` |
| Per-model training curves | `results/nb5*/…/nb5_results.json`, `results/nb7/*/nb5_results.json` |
| Zero-shot VLM predictions | `results/nb6/v2_zs_*.csv` |
| Grounding | `results/grounding.json`, `results/grounding_rows.csv` |
| Rendering leak | `results/ablate_marks.csv` |
