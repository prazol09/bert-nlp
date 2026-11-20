# Phase 3 — Extensions beyond the Assignment 3 spec: notes

**Status:** complete. Team decisions, **not** part of the required Assignment 3 deliverable. All results are **preliminary validation results**; the test split is not opened.
**Run it:** `python phase3_extensions.py` (about 15 minutes on an Apple M4, most of it the sweep) writes `results/phase3/ext/`. Walk-through: `phase3.ipynb`, sections 18–26. Plan: `docs/assignment/Phase-3/PHASE3_PLAN.md` §5.

## What was added
- **E1 Heavier noise.** Rates 0.01, 0.05, 0.10, 0.20 (the saved Assignment 2 files) plus 0.30, 0.50, 0.70, 0.90, generated with the same a2_default rule, seeds {11, 23, 47} and per-record RNG key, which gives 73 conditions.
- **E2 True loss.** `H_true = S / T_clean`: each condition's summed loss divided by the clean target count (35,971), the same denominator for every condition. `H = S / T` is kept beside it.
- **E2 Which operation hurts most.** Insertion, deletion and substitution ranked per model and rate on true loss, H, accuracy, cosine and distance; also plotted against the realized edit distance.
- **E3 Depth × epochs.** 1, 2 and 3 hidden layers, all 64 wide, 30 epochs, mean and concatenation, everything else as Assignment 3. The best epoch (lowest clean validation loss) is scored on all 73 conditions.
- **E4 Sentence tracer.** Any sentence through noise → edit distance → WordPiece → per-word vector drift → per-token predictions of both networks and the n-gram. Notebook parameters, an ipywidgets form, or `python phase3_extensions.py trace "…"`.

## Checks that passed
- The run **reproduces the required results** before reporting anything new: 74 neural rows (max |ΔH| 8.6e-8 on the GPU), the 37 n-gram control rows and the 37-row geometry audit (both exact).
- The noise generator regenerates a saved Assignment 2 file (`sub-p0.20-s47`, 500 records) exactly before writing new rates.
- The required `neural_next_token.py` run, after the model and training code gained the optional `n_layers` and `device` settings, still gives identical result tables and byte-identical checkpoints.
- 97 tests pass (14 new: default depth keeps the Assignment 3 layer names and initialisation, parameter counts for deeper networks, dropout 0 draws no random numbers, dropout only in training, early stopping, the 10K chunk is article-complete and contains the 500-record slice, true-loss arithmetic, ranking direction, controlled trace edits, deterministic variants).
- E (the frozen BERT table) has the same hash before and after the sweep.

## Results
**Heavier noise and true loss** (concatenation model, mean over 3 seeds; clean H = 7.28):

| | p = 0.20: H | p = 0.20: true loss | p = 0.90: H | p = 0.90: true loss |
|---|---|---|---|---|
| insertion | 9.17 | 17.88 | 10.22 | 39.72 |
| substitution | 9.04 | 15.61 | 9.71 | 24.39 |
| deletion | 8.40 | 11.08 | 7.04 | 3.51 |

- **H understates the damage, increasingly with p.** At p = 0.90 insertion, H (10.22) sits near ln K = 10.29 (chance) and has flattened, while the true loss keeps rising, because the text keeps growing (140k targets vs 36k clean).
- **Deletion turns around.** Its true loss peaks near p = 0.30 (≈ 11.2), falls below clean by p = 0.70 and reaches 3.5 at p = 0.90; accuracy recovers to 13.2% (clean 13.8%). This is not robustness. At p = 0.90 only half as many targets remain and 40% of them have no letters (punctuation, digits, end of record) against 20% in clean text. Loss cannot tell preserved text from removed text.
- **Which operation hurts most depends on the measurement.** For prediction it is **insertion > substitution > deletion**, in true loss for all three models at every rate. For static vector drift the order flips (deletion moves vectors most, insertion least, from p = 0.10 up). The three operations cause almost the same edit distance per character at each p, so this is not one operation simply editing more. The smallest drift going with the largest prediction damage is a counterexample to "small drift means robust prediction".
- **The n-gram control** has the lowest true loss under heavy insertion and substitution.
- Geometry at p = 0.90 deletion covers only 385 of 1,000 spans; the other 615 are fully deleted and undefined.

**Depth × epochs:**
- All six variants are best at **epoch 2**. More epochs only raise clean validation loss (1-layer concatenation: 7.28 at epoch 2 → 8.10 at 10 → 11.30 at 30) while training loss falls 6.48 → 2.95. Training loss has not levelled off by epoch 30 (still −1.3 to −2.0% per epoch).
- **Depth does not help:** best validation loss within 0.04 nats of 1 layer (concatenation 7.283 / 7.325 / 7.308; mean 7.505 / 7.480 / 7.518). Deeper networks overfit faster (concatenation validation loss at epoch 30: 11.3 / 14.1 / 15.1).
- **Depth does not change robustness:** the true-loss increase under noise differs by at most 0.69 nats across depths (against increases of up to 33 nats), with no consistent direction.
- Almost all ~2M trainable parameters are in the 64 → 29,525 output layer; an extra 64×64 layer adds 4,160. The limit is the 500 training records, not depth.

**Sentence trace ("A cow jumps over the fence"):** one edit on "cow" splits it differently per operation (coow → `co ##ow`, cpw → `cp ##w`, cw → `cw`) with cosine 0.42–0.44 to the clean word in all three cases. Insertion and substitution add a target, deletion does not. The networks' guesses on this out-of-domain sentence are mostly frequent tokens ("the", ",", "."), so it is an illustration only.

## R1 — Regularisation (dropout × weight decay on the same 500 records)
**Run:** `python phase3_extensions.py regularise` (about 55 minutes on the GPU) writes `results/phase3/ext/reg/`. Notebook §24.
**Setup:** dropout {0, 0.2, 0.5} after every hidden ReLU × Adam weight decay {0, 1e-5, 1e-4, 1e-3}; mean and concatenation, 1 hidden layer of 64, 30 epochs, everything else as Assignment 3. Per model the setting with the lowest clean validation loss wins; it is re-run at 2 and 3 layers, re-trained to convergence (up to 100 epochs, stop after 5 without improvement) and scored on all 73 noise conditions.

| | no regularisation | best setting | best epoch | clean validation loss | accuracy |
|---|---|---|---|---|---|
| concatenation | 7.283 (epoch 2) | dropout 0.5, weight decay 1e-4 | 30 (33 when allowed 100) | **6.776** (6.775) | 13.8% → 16.2% |
| mean pooling | 7.504 (epoch 2) | dropout 0.2, weight decay 1e-4 | 29 | **7.024** | 9.8% → 14.3% |

- **Overfitting is gone.** Validation loss improves until about epoch 30 and then stays flat (rise ≤ 0.02 nats), instead of climbing 4 nats. Longer training confirms the plateau.
- **Weight decay does the work.** Dropout alone still peaks at epoch 2–3 (concatenation 7.23 at dropout 0.5). Weight decay 1e-5 still overfits early; 1e-4 or 1e-3 removes the rise; the best cells add dropout on top (concatenation 6.95 → 6.78).
- **Depth still does not help:** the concatenation winner at 2 / 3 layers reaches 6.90 / 6.99 (1 layer 6.78); mean pooling 7.18 / 7.21 (1 layer 7.02).
- **Better fit, same noise penalty.** True loss is lower almost everywhere, but the increase over each model's own clean loss is within about 0.1 nats of the unregularised model for insertion and substitution up to p = 0.20 (slightly smaller at heavy noise: concatenation insertion p = 0.90 +31.4 vs +32.5; deletion +0.3 larger at p = 0.20). The curve moves down; it does not flatten.
- **Against the n-gram control** the regularised concatenation model has the lower true loss in **19 of 24** noise cells (10 before): every substitution and deletion cell and insertion up to p = 0.10. The n-gram still wins under insertion from p = 0.20.
- **Caveat:** the setting is chosen on the validation records it is reported on, so these numbers are optimistic. The test split stays sealed.
- The required `neural_next_token.py` run still gives identical tables and byte-identical checkpoints after adding dropout and early stopping (dropout 0 draws no random numbers).

## R2 — More data (a 10K-record training chunk)
**Run:** `python phase3_extensions.py scale` writes `results/phase3/ext/scale/`. Notebook §25. About 2.5 hours of GPU time (the logged wall time is longer because the laptop slept).
**Data:** the first 10,000 `train_full` records in file order, extended to the end of the article in progress: 10,010 records, 383 articles, 1,303,238 training targets (36.5× the slice). It contains all 500 slice records. Validation records and the 73 noise conditions are unchanged. Only 0.6% of validation targets are unseen in training (17.0% on the slice).
**Runs:** per model, no regularisation, the R1 winner, and the R1 winner's dropout with weight decay 1e-5; 1 layer of 64, up to 30 epochs, stop 3 epochs after the last improvement. The n-gram is rebuilt on the same chunk with its interpolation weights re-selected on clean validation (Assignment 2 procedure; not written to `configs/phase2.yaml`).

| clean validation loss | 500: none | 500: best (R1) | 10K: none | 10K: best | best accuracy |
|---|---|---|---|---|---|
| concatenation | 7.283 | 6.776 | 6.180 | **6.004** (dropout 0.5, wd 1e-5) | 16.2% → 18.2% |
| mean pooling | 7.504 | 7.024 | 6.401 | **6.138** (dropout 0.2, wd 1e-5) | 14.3% → 16.5% |
| n-gram control | 7.549 (λ = 1, 0, 0) | | 7.066 (λ = 0.75, 0.25, 0) | | |

- **More data is the bigger lever:** about −0.8 nats over the best regularised 500-record model, −1.3 over the required Assignment 3 model.
- **Less regularisation is needed with more data.** The R1 winners (weight decay 1e-4) are now worse than no regularisation (concatenation 6.25 vs 6.18, mean 6.47 vs 6.40); weight decay 1e-5 is best.
- **The n-gram becomes a real context model** on the chunk (it now weights the previous token) and improves to 7.07, still about a nat behind the networks on clean text.
- **The networks now beat the matched n-gram in 22 of 24 noise cells**, including heavy insertion, which the n-gram won in every earlier setting. The exceptions are deletion at p = 0.70 and 0.90 (shrinking-text regime; all within 0.11 nats).
- **More data, unlike regularisation, shrinks the noise penalty.** Concatenation's true-loss increase at 90% insertion falls from +31.4 (500 records, regularised) to +28.0, below the n-gram's +29.9; at 90% substitution +16.9 → +14.7 (n-gram +15.9). At p ≤ 0.20 insertion and substitution increases are about equal across models; under deletion the networks' increase is about 0.5 nats larger than the n-gram's at p = 0.20.
- Caveats: settings chosen and reported on validation (optimistic); one training run per setting; the per-run `seconds` include laptop sleep.

**R2b — depth on the 10K chunk.** Each model's best 10K setting (weight decay 1e-5) re-trained at 2 and 3 hidden layers (`python phase3_extensions.py scale --depth-only`, about 2 hours on the GPU); the 1-layer rows are the R2 runs, reused.

| best clean validation loss | 1 layer | 2 layers | 3 layers |
|---|---|---|---|
| concatenation (dropout 0.5, wd 1e-5) | **6.004** | 6.154 | 6.259 |
| mean pooling (dropout 0.2, wd 1e-5) | 6.138 | **6.119** | 6.176 |

- Concatenation gets worse with each layer even with 36× more data. Mean pooling gains 0.02 nats at 2 layers (the only time an extra layer helped in any experiment) and loses it at 3.
- Under heavy noise the 2-layer concatenation model is slightly better despite its worse clean loss (true loss at 90% insertion 33.56 vs 33.99); its increase over its own clean loss is smaller at all 16 insertion and substitution rates (by 0.02–0.58 nats), partly because it starts from a higher clean loss.
- Regularisation was tuned at 1 layer and not re-tuned for the deeper networks. Conclusion across all depth experiments: depth is neutral to harmful for these networks; data and the right regularisation matter far more.

## GPU
Apple M4 GPU (`mps`): one training epoch 1.1 s vs 3.4 s on one CPU thread; scoring 4–5× faster. Scoring on the GPU matches the CPU within 1e-7 nats. GPU training repeats exactly run to run on this machine, but takes a slightly different numerical path from the CPU: the 1-layer models' validation loss stays within 0.008 nats of the CPU run over 5 epochs, and the same epoch is kept. The required results therefore stay on the CPU.

## Caveats
- Extensions are team decisions beyond the assignment; the required 74 / 37 / 37 rows are unchanged and remain the deliverable.
- True loss charges all of a condition's loss to the clean text, so it falls when noise *removes* text (heavy deletion). Read it with T and the composition of the targets.
- The seed range is variation across 3 random corruptions of one trained model per variant, not a confidence interval. The sweep trains each variant once.
- Sweep variants were trained on the GPU and are not bit-identical to CPU training.
- p is the probability of editing each eligible letter. At p = 0.90 the text is mostly noise; these rates probe the extremes and are not realistic typo levels.

## Next steps
- **Depth with re-tuned regularisation** at 2 and 3 layers, if depth is to be pursued at all.
- **The full training split** (`train_full`, 17,556 records) is the natural next point on the data-size curve (500 → 10K → 17.6K), with weight decay re-tuned around 1e-5.

## Next network (recommendation, not built)
The pretrained BERT encoder, frozen: layer-wise clean/noisy cosine across its 12 layers (the static table E is layer 0) and masked-token cross-entropy (pseudo-perplexity). Training a transformer or an RNN from scratch on 500 records would overfit the same way the FFNNs do. An optional cheap bridge first: an FFNN with a 4- or 8-token context.

## Files
`configs/phase3.yaml` (`extensions:`), `src/ext/{conditions,scoring,trace,figures,run}.py`, `phase3_extensions.py`, `tests/test_ext.py`, `phase3.ipynb` §18–24,
`results/phase3/ext/{condition_rows,noise_summary,noise_realized,operation_ranking,geometry_audit,geometry_summary,sweep_history,sweep_condition_rows,sweep_noise_summary}.csv`, `ext_run_info.json`, `figures/ext_*.png`,
`derived/phase3/noisy_ext/` (36 files, not committed; hashes in `index.csv`), `derived/phase3/ext/*_best.pt` (not committed; hashes in `ext_run_info.json`).
R1: `src/ext/regularise.py`, `results/phase3/ext/reg/{reg_grid,reg_best,reg_history,reg_condition_rows,reg_noise_summary,reg_long,reg_long_history,reg_long_condition_rows,reg_long_noise_summary}.csv`, `reg_run_info.json`, `figures/reg_*.png`; checkpoints `derived/phase3/ext/reg/*.pt` (not committed; hashes in `reg_run_info.json`).
R2: `src/ext/scale.py`, `results/phase3/ext/scale/{scale_depth_runs,scale_depth_history,scale_depth_noise_summary,scale_depth_condition_rows,scale_runs,scale_best,scale_history,scale_condition_rows,scale_noise_summary,ngram_lambda_selection}.csv`, `chunk_manifest.json` (record IDs + hash), `scale_run_info.json`, `figures/scale_*.png`; checkpoints `derived/phase3/ext/scale/*.pt` (not committed; hashes in `scale_run_info.json`).
Small code changes outside `src/ext/` (defaults leave Assignment 3 unchanged): `src/neural/model.py` (`n_layers`, `dropout`), `src/neural/train.py` (`device`, `patience`, keeps only the best epoch's weights), `src/neural/evaluate.py` (scores on the model's device), `src/geometry/drift.py` (`build_pairs` accepts a condition list).
