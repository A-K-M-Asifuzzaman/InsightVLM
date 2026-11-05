# InsightVLM-20K: Methods

**A. K. M. Asifuzzaman, Robiul Hasan Jisan, Fahim Khan Alif, Iftehad Kamal Iftee, and Mohammad Arfat**

Team Hocrux, Phitron AI/ML Thesis Track

[Live Demo](https://huggingface.co/spaces/AsifZaman1912/InsightVLM) · [Dataset (Zenodo)](https://doi.org/10.5281/zenodo.23968282) · [Back to README](../README.md)

* * *

***Abstract*—This document specifies how the charts, models and evaluations of InsightVLM-20K are produced. Charts are rendered by matplotlib and seaborn (26 chart types, 8 styles) with no label-dependent marks, and re-rendered by Plotly and Vega-Lite for evaluation, with Vega-Lite held out of training entirely. ResNet-101, ResNet-50 and DenseNet-121 are fine-tuned at 384 × 384 with a real-weighted sampler, label-preserving augmentation and checkpoint selection on real validation accuracy; the deployed model is the equal-weight ensemble of ResNet-101 and DenseNet-121. Zero-shot and generative vision–language baselines are described, together with the evaluation protocol (real-chart headline accuracy, bootstrap confidence intervals, paired exact McNemar tests, validation-only selection) and a Grad-CAM grounding analysis.**

***Index Terms*—Chart rendering, data augmentation, transfer learning, model ensembling, evaluation protocol, Grad-CAM, vision–language models.**

* * *

## I. Rendering

Charts are drawn by `build/06_render.py` with matplotlib [[20]] and seaborn on a 768 × 576 canvas, 26 chart types and 8 styles. Version 2 renders with `--clean`, which draws no label-dependent marks. Tight bounding-box cropping is disabled because it resizes the canvas after the axes transform is read, which would shift every evidence box.

**TABLE I**  
*RENDER SETS*

| Render set                    | Script                              | Used for                                      |
|-------------------------------|-------------------------------------|-----------------------------------------------|
| matplotlib, clean             | `06_render.py --clean`              | Training and the main test set                |
| matplotlib, randomised styles | `06_render.py --clean --randstyle`  | Training augmentation only                    |
| Plotly (5 templates)          | `11_render_plotly.py`               | Real test set; training augmentation          |
| Vega-Lite [[21]] (9 themes)   | `13_render_vega.py`                 | Real test set only; never seen in training    |

Randomised styles vary fonts, font sizes, line and marker sizes, colours, background tints, gridlines, spines and title placement; data and chart type are unchanged. Templates that imitate a training renderer (Plotly's seaborn template, and ggplot2 and fivethirtyeight in Vega-Lite) are excluded from the test renders.

## II. Models

### A. Supervised Backbones

ImageNet-initialised ResNet-101 [[11]], ResNet-50 and DenseNet-121 [[12]] are fine-tuned at 384 × 384 with the recipe in Table II.

**TABLE II**  
*TRAINING RECIPE*

| Component              | Setting                                                                                   |
|------------------------|-------------------------------------------------------------------------------------------|
| Optimiser              | AdamW [[22]], learning rate 3 × 10⁻⁴, weight decay 0.05                                   |
| Schedule               | One-cycle [[23]], 12 epochs                                                               |
| Regularisation         | Label smoothing 0.1 [[24]], mixed precision                                               |
| Hardware               | 2 × T4 (Kaggle)                                                                           |
| Batch composition      | About half of every batch is real (weighted sampler; real is 31% of the balanced split)   |
| Checkpoint selection   | Real validation accuracy (553 charts), mixed accuracy as tie-break                        |
| Augmentation           | Label-preserving only: mild crop and scale, colour jitter, blur, small erasures           |

No flips are used, because mirroring turns a positive correlation into a negative one and an increasing series into a decreasing one.

### B. Ablation Ladder

Each step is an ablation of the next: balanced split, then the supplementary real pool added, then renderer augmentation added (each epoch shows each training chart as one of its three renders). One model is trained per Kaggle kernel, because host memory grows across epochs and is not returned between models.

### C. Deployed Ensemble

The deployed model is the equal-weight average of the class probabilities of the two full-recipe models (ResNet-101 and DenseNet-121, trained with the pool and renderer augmentation). A label-shift correction towards real class frequencies is available, with its strength tuned on validation; the tuned strength is zero. Alternatives reported are the average of all seven trained models (86.2%) and greedy selection on validation (86.5%).

### D. Zero-Shot Baselines

Qwen2.5-VL-3B and -7B [[8]] and Qwen3-VL-8B [[9]] receive each chart together with the 20 class names and one-line definitions, and answer with one class name under greedy decoding.

### E. Generative Model

Qwen2-VL-2B [[7]] is fine-tuned with QLoRA [[10]] (r = 16, 4-bit NF4) for one epoch on 4,000 samples, trained to write the reasoning chain and insight, on version-2 renders (`nb4v2_vlm`).

## III. Evaluation Protocol

- **Headline.** Accuracy on the 1,192 real test charts, for the matplotlib, Plotly and Vega-Lite renders. Mixed accuracy is reported but is dominated by synthetic charts.
- **Floor and ceiling.** The chart-type rule (28.8%) and the statistical oracle on the data table (99.95%), rather than the 5% chance level.
- **Uncertainty.** 95% bootstrap confidence intervals (4,000 resamples) and paired exact McNemar tests [[25]] between systems on the same charts.
- **Selection on validation only.** The test split is scored once.

## IV. Grounding Analysis

For each real test chart with at least one evidence box, the Grad-CAM [[13]] peak of each deployed member (predicted class, last convolutional block) is mapped back to image pixels. A hit is a peak inside an evidence box. The control is a uniformly random point in the plot area, whose expected hit rate equals the boxes' share of the plot area (`train/grounding_eval.py`).

## V. Related Documents

**TABLE III**  
*SUPPLEMENTARY DOCUMENTS*

| Document                      | Contents                                                                    |
|-------------------------------|-----------------------------------------------------------------------------|
| [DATASET.md](DATASET.md)      | Taxonomy, verification gate, sources and licences, splits, files and fields |
| [RESULTS.md](RESULTS.md)      | All results, ablations, error analysis, grounding, rendering-leak study     |
| [REPRODUCE.md](REPRODUCE.md)  | Every command in order and the file behind each number                      |
| [DEMO.md](DEMO.md)            | How the demo works and how to run it                                        |

**License.** Code: [MIT](../LICENSE). Dataset: CC BY 4.0 for the corpus; real-source charts keep the licence of their source table; synthetic charts are CC0.

## Acknowledgment

This work was carried out by Team Hocrux in the Phitron AI/ML Thesis Track.

## References

Numbering follows the [README](../README.md#references).

[7] P. Wang *et al.*, "Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution," *arXiv:2409.12191*, 2024.

[8] S. Bai *et al.*, "Qwen2.5-VL technical report," *arXiv:2502.13923*, 2025.

[9] Qwen Team, "Qwen3-VL technical report," 2025.

[10] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.

[11] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.

[12] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 4700–4708.

[13] R. R. Selvaraju *et al.*, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626.

[20] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Comput. Sci. Eng.*, vol. 9, no. 3, pp. 90–95, 2007.

[21] A. Satyanarayan, D. Moritz, K. Wongsuphasawat, and J. Heer, "Vega-Lite: A grammar of interactive graphics," *IEEE Trans. Vis. Comput. Graphics*, vol. 23, no. 1, pp. 341–350, 2017.

[22] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.

[23] L. N. Smith and N. Topin, "Super-convergence: Very fast training of neural networks using large learning rates," in *Proc. SPIE Artif. Intell. Mach. Learn. Multi-Domain Oper. Appl.*, 2019.

[24] R. Müller, S. Kornblith, and G. Hinton, "When does label smoothing help?" in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019.

[25] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.