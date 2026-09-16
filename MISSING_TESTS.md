# Missing Tests Analysis

This document catalogues every testing gap found across the project's test suite
after reading all 20 test files and all major source files. Gaps are organised by
subsystem. Severity labels are:

- **[CRITICAL]** — a bug here would be silent and very hard to debug
- **[HIGH]** — likely to matter in practice; easy to get wrong
- **[MEDIUM]** — correctness concern but usually caught at runtime quickly
- **[LOW]** — nice to have; regression protection

Questions that arose during analysis are collected at the bottom.

---

## 1. Quantizer Core (`quantizers/`)

### 1.1 Fixed-point per-tensor

**find_optimal_lsb — activation selection rule (prefer_high_lsb=True) [CRITICAL]**

`tests/test_fixedpoint_per_tensor.py` only tests the weight selection rule
(`prefer_high_lsb=False`: pick the candidate with most unique values; tie-break on
lowest SAD). The activation rule (`prefer_high_lsb=True`: among all candidates that
reach the global max unique-value count, pick the *highest* LSB) is not tested at
all. The two rules diverge whenever multiple LSB candidates tie on unique-value count —
which is exactly the common case for activations where ReLU pushes many values to zero
or to the same quantization grid. If the activation rule regressed, every QAT run would
silently calibrate with an overly fine grid (too low an LSB), wasting representable range.

Missing tests:
- Construct a weight tensor with two candidates tying on unique count; verify
  `prefer_high_lsb=False` picks the lower LSB (SAD tie-break).
- Construct an activation tensor with the same tie; verify `prefer_high_lsb=True`
  picks the higher LSB.
- Construct a case where all candidates reach full unique-value coverage (2^N);
  verify `prefer_high_lsb=True` still returns the largest candidate in the search
  range rather than the smallest.

**FixedPointQuantFn._integer_queue reset [CRITICAL]**

`FixedPointQuantFn` uses a class-level `deque` (`_integer_queue`) to smuggle integer
tensors from `forward()` to `symbolic()` during ONNX export. `reset_capture_state()`
clears this deque. There is no test that:
- Verifies `reset_capture_state()` actually clears a non-empty deque.
- Verifies that a second `torch.onnx.export()` call after a reset does not produce
  stale integer values from the first export.
- Verifies that calling `reset_capture_state()` on `SiLUQuantFn` and
  `CoefficientQuantFn` has the analogous effect (both have the same pattern).

If the deque is not reset between exports, the second ONNX graph silently picks up
integer tensors belonging to the first export, producing subtly wrong custom node
attributes. This is a real risk in notebook workflows.

**FixedPointPerTensorQuantizer — quantizer_role propagation [HIGH]**

`quantizer_role` is set to `"weight"`, `"activation"`, or `"bias"` at class level
on the injector subclasses (`WeightQuant`, `ActivationQuant`, `BiasQuant`). It is
passed to `find_optimal_lsb` to choose the selection rule. There is no test that
verifies the role is correctly threaded all the way through `Brevitas QuantConv2d →
injector → FixedPointPerTensorQuantizer.calibrate()`. A broken injector could silently
set `quantizer_role=""` and fall through to an undefined behaviour in `find_optimal_lsb`.

**RoundingMode fallback path [MEDIUM]**

`quantize_fixed_point` has a `rounding_mode` parameter with modes `ROUND`, `FLOOR`,
`CEIL`, `TRUNCATE`. Tests cover `ROUND` and `FLOOR`. `CEIL` and `TRUNCATE` have no
explicit arithmetic verification tests. The existing `TestRoundingModes` class covers
the API surface but does not assert specific output values for `CEIL` and `TRUNCATE`.

**Boundary: LSB search with a single-element tensor [MEDIUM]**

The LSB search computes unique values across the tensor. For a 1-element tensor every
candidate LSB gives unique_count=1. The selection rule (weight vs. activation) could
behave unexpectedly. Not tested.

**Boundary: tensors that are all identical (zero variance) [MEDIUM]**

All elements equal to a constant → all candidates tie at unique_count=1. The SAD
for some will also be 0 (if the constant rounds perfectly). What does the search
return? Not tested.

**Force recalibration in a realistic training loop [HIGH]**

`force_recalibration` is tested in isolation
(`tests/test_fixedpoint_per_tensor.py:TestUncalibratedEvalGuard`), but there is no
test that exercises the full cycle: calibrate → run several steps → set
`force_recalibration=True` → verify the next forward re-runs the search and updates
`search_result_lsb` — without triggering the uncalibrated-eval guard. The guard
checks `search_done` but `force_recalibration` resets it; the interaction needs a
test.

---

### 1.2 Coefficient per-tensor weights

**ONNX export for CoefficientPerTensorWeightQuantizer [CRITICAL]**

`tests/test_coefficient_per_tensor_weights.py` has no ONNX export test. The
`CoefficientQuantFn.symbolic()` method emits `Quantify::CoefficientQuant` custom
nodes just like `FixedPointQuantFn`. This path is completely untested. If `symbolic`
is broken, no error appears until someone tries to export a model that uses
`CoefficientPerTensorWeightQuant`, which may be an infrequent path.

**CoefficientQuantFn._integer_queue reset [HIGH]**

Same structural risk as `FixedPointQuantFn`. Not tested.

**Gradient flow through CoefficientQuantFn [MEDIUM]**

`tests/test_quantizer_gradients.py` covers gradients for fixed-point and SiLU
quantizers but not for `CoefficientPerTensorWeightQuantizer`. A regression in
`CoefficientQuantFn.backward()` (which uses STE pass-through) would be silent.

---

### 1.3 SiLU quantizer

**SiLU gradient with explicit expected values [MEDIUM]**

`tests/test_silu_quant.py:TestSiLUGradientFlow` checks that gradients are non-None
and non-zero but does not compare them against hand-calculated values. The `SiLUQuantFn`
backward uses an unusual STE variant (gradient clipping at the SiLU output range;
see source). A regression in the clip threshold would pass the current test.

**SiLUQuantFn._integer_queue reset [HIGH]**

Same structural risk as `FixedPointQuantFn`. Not tested.

---

### 1.4 AnnealingBlendFn

**Boundary at alpha=0.0 [CRITICAL]**

When `alpha=0.0`, `AnnealingBlendFn` should return the float input unchanged.
`tests/test_quantizer_gradients.py:TestSTEGradient` tests the `alpha=1.0` and
`alpha=0.5` cases but not `alpha=0.0`. At exactly zero, the output must be the
raw tensor (no quantisation at all), and the gradient must pass through unmodified.
If `AnnealingBlendFn.forward()` has an off-by-epsilon error at 0.0, the float warmup
phase would silently bleed in a small amount of quantisation error.

**STE slope at alpha between 0 and 1 [HIGH]**

The STE is supposed to have a constant slope of 1 regardless of alpha (see docstring
in `base_quantizer.py`). This is what makes gradients flow during the annealing ramp.
There is no test that verifies `d(output)/d(input) = 1` for intermediate alpha values
via `torch.autograd.gradcheck` or an explicit backward call.

**AnnealingBlendFn: requires_grad on the output tensor [MEDIUM]**

When `alpha=0.0`, no quantization happens, but the blend formula still calls into
the custom Function. Is `output.requires_grad` propagated correctly from the input?
Not tested.

---

## 2. QuantizerManager (`quantizers/manager.py`)

**request_snapshot [HIGH]**

`QuantizerManager.request_snapshot(quant_id)` is mentioned in comments inside
`base_quantizer.py` and the diagnostics module, but there is no test that calls it
and verifies a diagnostic event is triggered on the next forward pass of the named
quantizer.

**set_annealing_for_n_inferences skip_calibrated interaction [HIGH]**

`test_preserve_calibrated_quantizers.py` tests that `skip_gating_for_calibrated_quantizers`
skips already-calibrated quantizers. But the interaction between that call *and*
`set_annealing_for_n_inferences(skip_calibrated=True)` — which sets `annealing_alpha`
based on whether the quantizer was already calibrated — is only tested at a coarse
level. There is no test that explicitly verifies a calibrated quantizer ends up with
`annealing_alpha = 1.0` immediately after `_activate_qat()` while an uncalibrated one
starts at `0.0`.

**Global recalibration ordering with multiple simultaneous quantizers [MEDIUM]**

`tests/test_fixedpoint_manager.py:TestGlobalRecalibration` tests that all quantizers
get recalibrated. But the test uses sequential quantizers in a tiny model. There is no
test ensuring that a model with both weight and activation quantizers (interleaved in
the forward graph) recalibrates all of them correctly when some are `search_done=True`
and some are not.

**diagnostics_dir assignment propagation [MEDIUM]**

`_activate_qat` assigns `mgr.diagnostics_dir`. No test verifies that after this
assignment, a triggered diagnostics event actually writes to the configured directory
and does not silently fail (e.g., because the directory doesn't exist yet when the
first calibration fires).

**QuantizerManager state isolation between unit tests [HIGH]**

`QuantizerManager` is a singleton. If one test registers quantizers and fails without
cleanup, the next test inherits stale state. The existing tests call `mgr.reset()` in
fixtures, but this is not enforced at the fixture level for all test files. A test
that calls `QuantizerManager()` without first calling `reset()` may see leftover
quantizers from a previous test. This is a test-infrastructure gap that can cause
intermittent failures.

---

## 3. ONNX Export (`utils/onnx_export.py`)

**export_onnx_with_io — end-to-end [CRITICAL]**

`tests/test_fixedpoint_onnx_export.py` tests ONNX export via direct
`torch.onnx.export` calls in a minimal test fixture that constructs the export
manually. The **canonical** public function `export_onnx_with_io` (which adds dummy
I/O embedding, zero-bias injection, state reset, and the `Quantify` custom opset)
has no dedicated test. Specifically untested:
- The `embed_mode="metadata"` path and subsequent `load_embedded_io()` round-trip
  (embed, reload, compare arrays).
- The `embed_mode="initializer"` path.
- The zero-bias injection step (`inject_zero_biases`) — does it correctly add a zero
  bias to a `QuantLinear(bias=False)` layer without affecting forward output?
- The `reset_states=False` flag being honoured (no reset called).
- `dynamo=True` being properly rejected or handled for custom quantizers.

**load_embedded_io [HIGH]**

`load_embedded_io` decodes base64 arrays from ONNX metadata properties. No test
verifies:
- Shape and dtype are recovered correctly from a round-trip.
- An ONNX model that was *not* exported with `embed_mode="metadata"` raises the
  correct `KeyError` with a useful message.
- A float64 array survives the encode/decode round-trip.

**export_onnx_qcdq [MEDIUM]**

`export_onnx_qcdq` wraps Brevitas' standard QCDQ exporter. No test verifies it can
actually export a model and that the resulting ONNX graph contains
`QuantizeLinear`/`DequantizeLinear` nodes.

**reset_quantizer_states — all three quantizers [HIGH]**

`reset_quantizer_states()` tries to reset `FixedPointQuantFn`, `SiLUQuantFn`, and
`CoefficientQuantFn`. If one of the imports fails (e.g., module name changes), the
function silently swallows the `ImportError`. No test exercises all three resets and
verifies each deque is emptied.

---

## 4. Training Harness V2 (`training_harness/trainer_v2.py`)

This is the most undertested subsystem. The old v1 `Trainer` has an integration test
(`tests/test_training_harness.py`). The v2 `QATTrainerV2` — the harness actually used
in production — has **zero tests**.

### 4.1 Three-phase transition

**float warmup → QAT transition [CRITICAL]**

The core correctness guarantee of V2 is that quantization is fully disabled during
float warmup and then correctly activated at the right epoch. No test verifies:
- After `fit()` starts, all quantizers have `annealing_alpha = 0.0` during the first
  epoch.
- After `float_warmup_epochs` epochs, `_activate_qat()` is called exactly once.
- After activation, `_qat_active = True` and quantizers have non-zero alpha.
- `plateau_detector.step()` firing early (before `float_warmup_epochs`) still triggers
  `_activate_qat()`.

**float_warmup_epochs=0 path [HIGH]**

When starting from a PTQ checkpoint, `float_warmup_epochs=0` triggers `_activate_qat()`
before the training loop starts. No test verifies this path calls `_activate_qat()`
before the first epoch and does not call it again mid-loop.

**preserve_calibrated_quantizers in _activate_qat [HIGH]**

When `preserve_calibrated_quantizers=True`, already-calibrated quantizers skip the
`search_done=False` reset, are given alpha=1.0 immediately, and skip gating.
Uncalibrated ones go through normal reset+annealing. No test verifies this mixed
scenario inside `QATTrainerV2` (the unit tests in
`test_preserve_calibrated_quantizers.py` test the manager in isolation, not the
trainer calling it).

### 4.2 EMA integration

**EMA apply_to / restore during validation [HIGH]**

The trainer temporarily swaps EMA parameters into the model for validation, then
restores training parameters. No test verifies:
- After `_ema.apply_to(model)`, model parameters equal shadow model parameters.
- After `_ema.restore(model, stash)`, training parameters are exactly restored.
- When `_ema` is None (ema_decay=0.0), the code path runs without error.
- The EMA decay actually damps fast parameter changes (decays toward the
  exponential average, not away from it).

**EMA state_dict in checkpoint extra [MEDIUM]**

Each checkpoint bundle includes `{"ema_state_dict": self._ema.state_dict()}`.
`_post_training()` loads the best checkpoint and, if EMA was active, copies shadow
parameters into the main model. No test verifies that after this, `model.parameters()`
match what the EMA shadow contained at the best checkpoint epoch.

**EMAModel.update buffer sync [HIGH]**

`EMAModel.update()` blends parameters (EMA) but copies buffers directly. This means
BN running stats and quantizer calibration buffers (search_done, search_result_lsb,
annealing_alpha) in the shadow model get overwritten with the training model's values
on every update. If a quantizer recalibrates mid-training (force_recalibration), the
shadow model's calibration state is overwritten immediately. Is this the intended
behaviour? No test covers this interaction.

### 4.3 Reduce LR on Plateau — new feature (just added)

**reduce_lr_metric mode switching [CRITICAL]**

The just-added feature picks `mode="max"` for `val_acc`/`train_acc` and `mode="min"`
for everything else. No test verifies:
- `val_acc` → `ReduceLROnPlateau(mode="max")` (LR reduces when acc stops going up).
- `val_loss` → `ReduceLROnPlateau(mode="min")` (LR reduces when loss stops going down).
- An unknown metric name → falls through to `mode="min"` without error.
- When the chosen metric is absent from `all_metrics`, the fallback chain
  (`val_loss` → `train_loss` → 0.0`) is exercised correctly.

**reduce_lr_metric fallback chain [HIGH]**

If `reduce_lr_metric="val_acc"` but `val_loader` is None (no validation), `val_acc`
is absent from `all_metrics`. The fallback chain in `trainer_v2.py:363` uses
`all_metrics.get(metric, all_metrics.get("val_loss", all_metrics.get("train_loss", 0.0)))`.
Feeding 0.0 to a `mode="max"` scheduler every epoch would cause LR decay immediately.
No test covers this edge case.

**ReduceLROnPlateau actually fires [HIGH]**

There is no test that constructs a `QATTrainerV2` with `reduce_lr_on_plateau=True`,
feeds a metric that doesn't improve for `patience+1` epochs, and verifies that the LR
was actually multiplied by `factor`. This is the most basic sanity check for the new
feature.

### 4.4 Early stopping integration

**early_stopper gate: only after is_quantizing_everything_fully [HIGH]**

`fit()` only triggers early stopping when `QuantizerManager().is_quantizing_everything_fully`
is True. No test exercises the scenario where early stopping would have fired but was
suppressed because annealing was still in progress — and then does fire once annealing
completes.

**early_stopper restore_best_weights [MEDIUM]**

When the early stopper fires, it calls `self.early_stopper.restore(self.model)`.
No test verifies that the model parameters after restoration match the best epoch.

### 4.5 Epoch runner mechanics

**_run_epoch with dry_run=True [MEDIUM]**

`dry_run` limits the loop to `dry_run_batches` batches. No test verifies that with
`dry_run=True`, the loop exits early and still produces valid metric snapshots.

**after_step_hook and after_epoch_hook [MEDIUM]**

These hooks allow external code to observe or modify trainer state per-step and per-epoch.
No test verifies they are called with the correct arguments, or that raising an exception
inside a hook propagates correctly.

**_unpack_batch — dict batch format [MEDIUM]**

`_unpack_batch` handles `(list/tuple)` and `dict` batch formats. Tests use only the
tuple format. The dict format (keys `input`/`label`, `image`/`target`, etc.) is not
tested, including the error path when expected keys are absent.

**val_loader=None during fit() [HIGH]**

When `val_loader=None`, the val epoch is skipped, `val_metrics` is an empty dict,
and checkpoint monitoring falls back to `train_loss`. No test exercises a full
training loop without a val loader to verify no KeyError occurs in the metric
fallback chain.

### 4.6 Resume from checkpoint

**resume=True in fit() [HIGH]**

`checkpoint_mgr.resume()` restores model + optimizer + scheduler state and returns
`start_epoch`. No test exercises the `resume=True` path in `fit()` to verify that
training continues from the correct epoch index rather than from 0.

---

## 5. Training Utilities

### 5.1 EarlyStopping (`engine_utils.py`)

No test for `EarlyStopping` at all. Missing:

- **Basic: mode="min"** — counter increments when value is not smaller by `min_delta`;
  `step()` returns True after `patience` non-improving steps.
- **mode="max"** — counter increments when value is not larger by `min_delta`.
- **Counter reset on improvement** — a single improvement after 3 bad epochs resets
  counter to 0.
- **stopped_epoch recording** — verify `stopped_epoch` is set to the epoch passed
  to `step()` when stopping fires.
- **restore_best_weights=True** — verify that `restore()` loads the weights captured
  at the best epoch, not weights from after the decay.
- **restore called without model passed to step** — should emit the warning in
  `restore()`.
- **min_delta boundary** — exactly-equal values (within min_delta) should not count
  as improvement.

### 5.2 LossPlateauDetector (`engine_utils.py`)

Only used inside `QATTrainerV2.fit()`. No standalone test. Missing:

- **Basic firing** — feed `patience` identical values, verify returns True on the
  `patience`-th call.
- **plateau_triggered=True after firing** — subsequent calls should always return
  False even if the metric continues deteriorating.
- **Reset on improvement** — verify a descending sequence never fires.
- **first call initialises best_value** — verify no premature firing on epoch 0.

### 5.3 CheckpointManager (`checkpointing.py`)

All checkpoint functionality is untested. Missing:

**Top-K eviction [CRITICAL]**
- Save K+1 checkpoints with metric values where K+1th is worst; verify only K files
  remain on disk and the evicted one is gone.
- `monitor_mode="max"`: eviction evicts the *lowest* metric, not the highest.
- `monitor_mode="min"`: eviction evicts the *highest* metric value.

**_should_save logic [HIGH]**
- Full pool (K records) where new metric is worse than worst in pool: verify save is
  rejected.
- Full pool where new metric is better: verify it's admitted and one is evicted.

**save_every_n_epochs [MEDIUM]**
- Configure `save_every_n_epochs=3`; call save at epochs 0,1,2,3; verify a periodic
  file is written only at epoch 2 and 5 (0-indexed: epoch 2 means epoch+1=3, etc.).
  (Off-by-one in `(epoch + 1) % save_every_n_epochs == 0` could cause wrong cadence.)

**resume() correctness [HIGH]**
- Save a checkpoint at epoch 5; call resume(); verify the returned start_epoch is 6.
- resume() with no checkpoint should return 0.
- resume() with `reset_calibration=True` should zero all `search_done` buffers.

**load_best() [HIGH]**
- After saving two checkpoints (better and worse), verify `load_best()` loads the one
  with the better metric.
- Verify that after `load_best()`, model parameters match what was saved.

**index persistence [MEDIUM]**
- Create a CheckpointManager, save two checkpoints, destroy the instance, create a
  new CheckpointManager pointing at the same directory; verify it re-loads the index
  and knows about both previous checkpoints.

**ONNX export failure is non-fatal [MEDIUM]**
- If the model's forward fails during ONNX export (e.g., wrong dummy input shape),
  CheckpointManager should print a warning and continue rather than crashing the
  training loop. The current code catches `Exception` broadly but this behaviour is
  not tested.

### 5.4 EMAModel (`ema.py`)

No tests at all. Missing:

- **update()**: after one update with decay=0.9, shadow parameters should be
  `0.9 * initial + 0.1 * new`. Verify numerically.
- **update()**: buffers (e.g., a registered buffer tensor) should be copied exactly
  (not blended). Verify.
- **apply_to() / restore()**: after apply_to, model params equal shadow params.
  After restore, model params equal original training params.
- **state_dict() / load_state_dict()**: round-trip serialisation; shadow params
  survive a save/load cycle.
- **decay=1.0 edge case**: EMA never updates; shadow always equals initial model.
- **decay=0.0 edge case**: shadow is immediately overwritten with current model on
  each update.
- **to(device)**: shadow model is moved to the specified device without error.
- **Quantizer buffer handling**: after `update()`, shadow model's `search_done`
  buffer matches training model's `search_done`. This matters because if the training
  model recalibrates mid-QAT, the shadow should reflect the new calibration.

### 5.5 MetricsTracker / AverageMeter (`metrics.py`)

No tests. Missing:

- **AverageMeter.update with n>1**: verify weighted average.
- **AverageMeter after reset**: verify sum/count/val return to 0.
- **MetricsTracker.commit_epoch**: verify per-step values are averaged and the meter
  is cleared for the next epoch.
- **MetricsTracker.get_metric_series**: verify ordering when train and val commits
  are interleaved.
- **MetricsTracker.best_value mode="max"**: verify it returns the maximum, not the
  minimum.
- **MetricsTracker.summary()**: verify "best_val_acc" is max and "best_val_loss" is min.
- **MetricsTracker.all_epoch_dicts()**: verify train and val metrics for the same
  epoch are merged into one dict.

### 5.6 WarmupCosineScheduler (`schedulers.py`)

No tests. Missing:

- At step 0, LR should equal `warmup_start_lr`.
- At step `warmup_steps`, LR should equal the base LR.
- At step `total_steps`, LR should equal `eta_min`.
- LR is strictly monotone increasing during warmup.
- LR is strictly monotone decreasing during cosine phase.
- `progress` is clamped to 1.0 (no overshoot past `total_steps`).
- Multi-param-group support: each group has its own base LR.

### 5.7 QATWarmupScheduler (`schedulers.py`)

This is the old V1 scheduler (still in the codebase). Not tested beyond the old MNIST
integration test, which uses it only implicitly. Explicit missing:

- `step(epoch=float_warmup_epochs)` enables quant on the transition epoch (calls
  `_set_quant_enabled(model, enabled=True)`), not before.
- `step(epoch=float_warmup_epochs - 1)` leaves quant disabled.
- `step(epoch >= freeze_bn_after_epoch)` freezes BN exactly once (idempotent).
- `in_float_warmup` and `in_qat` properties toggle at the right epoch.

### 5.8 TrainingConsole (`console.py`)

No tests. The console is hard to test because it reads stdin. Missing:

- **_dispatch "lr" command**: after dispatching `lr 3e-5`, optimizer's param_group LR
  is updated.
- **_dispatch "patience" command**: updates `_plateau_lr_sched.patience`.
- **_dispatch "factor" command**: updates `_plateau_lr_sched.factor`.
- **_dispatch "stop" command**: sets `stop_requested=True`.
- **_dispatch "lr" with bad value**: does not crash; prints error.
- **_dispatch with no _plateau_lr_sched**: patience/factor commands print warning
  gracefully.
- **drain() processes all queued commands in order**: if two commands are queued,
  both are dispatched.
- **start() returns False when not a TTY**: no thread spawned.

### 5.9 collect_scale_factors / freeze_bn (`schedulers.py`)

- **freeze_bn**: verify that after calling `freeze_bn(model)`, BN running stats do
  not change when running more batches in train mode.
- **collect_scale_factors**: verify it returns a non-empty dict for a model with
  at least one Brevitas quant proxy that has a scale.
- **collect_scale_factors**: verify it gracefully handles a module where `proxy.scale()`
  raises an exception (the try/except is there but untested).

### 5.10 calibration.py

`run_calibration` is not tested at all. Missing:

- **Basic calibration**: after `run_calibration(model, loader, n_batches=5)`, the
  model's BN running stats have been updated and quantizer ranges are initialised.
- **n_batches respected**: verify the loader is iterated at most `n_batches` times
  even if the loader is longer.
- **reset_calibration=True**: verify `search_done` buffers are zeroed before the pass.
- **forward_fn override**: verify the custom forward function is used instead of
  `model(inputs)`.
- **model training state restored**: if the model was in `train()` before calibration,
  it should be back in `train()` after.
- **inspect_quant_ranges**: verify it returns a dict with scale and zero_point for a
  calibrated Brevitas model.

---

## 6. Data / Preprocessing Pipeline

### 6.1 `training_harness/mixup.py` (custom local module)

This module (`mixup_batch`, `cutmix_batch`, `apply_mixup_cutmix`) is separate from
timm's `Mixup` class (which the trainer uses). It appears to be legacy code or
an alternative path. No tests at all. Missing:

- **mixup_batch output shape**: output shape equals input shape.
- **mixup_batch lam is in [0,1]**: with alpha=0.2, lam should be Beta-distributed;
  after 1000 samples, all should be in [0,1].
- **mixup_batch is a valid convex blend**: `output = lam * x + (1-lam) * x[perm]`.
  When lam=1.0, output equals input exactly.
- **cutmix_batch**: the cut region of the output equals the shuffled batch's region;
  outside the cut, output equals the original batch.
- **cutmix_batch lam correction**: after cutting, lam is adjusted to reflect the true
  pixel fraction, not the Beta sample.
- **apply_mixup_cutmix with both disabled** (alpha=0): returns inputs and targets
  unchanged, lam=1.0.
- **apply_mixup_cutmix with only mixup**: always applies mixup, never cutmix.
- **apply_mixup_cutmix with both enabled**: roughly 50/50 selection over many calls.

### 6.2 DALI pipeline (`utils/dali_pipeline.py`)

Not tested. DALI is the primary data loader for ImageNet training. Missing:

- **Corrupt image resilience**: the commit "Make DALILoader resilient to
  corrupt/unreadable images" (b6cf7bf) introduced error handling. No test verifies
  that a directory with a corrupt image still loads cleanly.
- **Output tensor shapes**: train loader yields `(B, 3, 224, 224)` images (float32,
  normalised) and `(B,)` int64 labels.
- **Validation loader**: no augmentation; images are centre-cropped and normalised
  consistently.
- **RandAugment n/m params**: verify that changing `randaugment_n` and `randaugment_m`
  does not crash and changes the augmentation behaviour.

### 6.3 `utils/weight_mapping.py`

No tests for the MobileNet remapping functions. Missing:

- **load_pretrained_weights with shape mismatch**: verify the mismatched tensor is
  skipped and a warning is logged.
- **load_pretrained_weights with 1D→4D reshape**: verify a flat depthwise conv
  weight is reshaped to `(out, in, kH, kW)` before loading.
- **_remap_timm_mobilenetv1_sd**: after remapping a fake timm state dict
  (constructed manually with the right key names), verify the output keys match
  the expected QuantMobileNetV1 key names.
- **_remap_timm_mobilenetv2_sd**: same for QuantMobileNetV2. The staging is
  particularly complex (`_MV2_STAGING` maps flat block indices to `(stage, idx)`
  pairs); a remapping bug would silently produce zero initialisation for some layers.
- **load_timm_weights with wrong arch**: verify it raises `ValueError`.

---

## 7. Models (`models/`)

### 7.1 QuantMobileNetV1

No forward pass test. Missing:

- **Forward shape** `(B, 3, 224, 224) → (B, 1000)`.
- **No quantization in eval without calibration**: without calibration,
  `search_done=False` for all quantizers; in eval mode this should raise the
  uncalibrated-eval guard.
- **BN fusion**: `test_bn_fusion.py` covers MobileNetV1 fusion, but only the
  numerical equivalence path. There is no test that fuses BN into a V1 model and
  then runs a full QAT forward pass to verify outputs don't NaN.

### 7.2 QuantMobileNetV2

No forward pass test beyond BN fusion. Missing:

- **Inverted residual skip connection** with `use_res_connect=True`: verify
  `output = x + self.conv(x)` (not just `self.conv(x)`).
- **Inverted residual without skip** (`stride=2` blocks): verify no residual is added.
- **Forward shape** `(B, 3, 224, 224) → (B, 1000)`.
- **expand_ratio=1 (first block)**: timm's first block has no expansion conv; our
  model has it as an identity. The weight remapping for this block is special-cased
  in `_remap_timm_mobilenetv2_sd`. No test verifies the remapped weights match the
  expected layout.

### 7.3 QuantResNet (covered better but gaps remain)

- **`test_resnet_quant.py`** tests structure, shapes, calibration coverage, and float
  checkpoint loading. The missing gap is **post-calibration QAT forward**: after
  calibrating a ResNet, run a forward in train mode with `annealing_alpha=1.0` and
  verify the output differs from the float forward (proving quantization is active).

---

## 8. Utils (miscellaneous)

### 8.1 `utils/quantizer_diagnostics.py`

No tests. The module produces SVG/PNG plots and text logs. Missing:

- **_compute_metrics correctness**: construct x and quantized tensors manually;
  verify MAE, SQNR, clip_low_pct, and coverage_pct values are computed correctly.
- **MAX_PLOT_SAMPLES subsampling**: with a tensor larger than `MAX_PLOT_SAMPLES`,
  verify only 100,000 elements are transferred to CPU for plotting.
- **_append_log**: verify the log file is created and contains expected key strings
  (quantizer ID, trigger name, bit width, SQNR).
- **_save_search_plot with empty search_records**: should return early without error.
- **run_diagnostics integration**: with a small tensor, verify both log and plot
  files are written to the specified directory.

### 8.2 `utils/csv_logger.py`

Not tested. Missing:

- Write a few rows; verify the CSV has correct headers and values.
- Verify that appending rows to an existing file works correctly (no header duplication).

### 8.3 `utils/model_info.py`, `utils/workspace.py`

Not tested at all.

---

## 9. Regression / Integration Gaps

### 9.1 _make_quant_id collision detection [HIGH]

`_reset_and_register` raises `RuntimeError` if two different module paths produce the
same `quant_id`. No test exercises this safeguard. A model where two layers share a
non-unique id would fail at runtime with a confusing error. Should test:
- A model where all ids are unique → no error.
- A pathological model (or mock) that would produce a collision → RuntimeError.

### 9.2 _reset_and_register idempotency [MEDIUM]

Calling `_reset_and_register` twice on the same model (e.g., `evaluate()` followed
by `fit()`) should not double-register quantizers in the manager. Currently there is
`mgr.reset()` before re-registering, but no test verifies the manager has exactly
`N` quantizers (not `2N`) after two successive calls.

### 9.3 Mixed precision (AMP) + fake quantization [HIGH]

`trainer_v2.py` emits a warning that AMP may interact poorly with Brevitas
fake-quantization. There is no test that runs a short training loop with
`mixed_precision=True` and verifies:
- No exception occurs.
- The warning is actually emitted.
- Gradients are finite (no NaN from autocast type promotion inside a quantizer).

### 9.4 Brevitas calibration_mode context manager [HIGH]

`calibration.py:run_calibration` uses `brevitas.graph.calibrate.calibration_mode`.
This context manager changes how Brevitas observes statistics. No test exercises it
with a `FixedPointPerTensorQuantizer` to verify that after the context exits, the
quantizer's calibration state (LSB, search_done) reflects what was observed during
the calibration pass.

### 9.5 PTQ checkpoint chaining with BN fusion [HIGH]

`test_ptq_checkpoint_chaining.py` and `test_load_ptq_checkpoint_bn_fusion.py` test
these separately. But the full chain:
1. PTQ weights checkpoint (BN fused)
2. PTQ bias checkpoint (continuing from 1, BN already fused)
3. PTQ activations checkpoint (continuing from 2)
4. QAT training starting from 3 with `preserve_calibrated_quantizers=True`

is not tested end-to-end. There is no test that all three PTQ roles are preserved
when QAT activation fires with `preserve=True` in a chained setup.

### 9.6 Quantizer serialization with stale manager references [HIGH]

`test_quantizer_checkpoint_roundtrip.py` documents the known risk of stale quantizer
references after `load_state_dict` and recommends calling `_reset_and_register` as
mitigation. But the mitigation itself is not tested in a QATTrainerV2 context — only
in an isolated unit test. The specific scenario of:
1. Create trainer + model
2. Train 5 epochs (model gets registered, calibrated)
3. Save checkpoint
4. Load checkpoint into a new trainer
5. Verify `QuantizerManager` has the correct quantizers (not the old ones)

is untested.

---

## 10. Questions for the Author

These questions arose during analysis and should be answered before writing the
corresponding tests:

**Q1** — `reduce_lr_metric` fallback when metric is missing: in `trainer_v2.py:366`,
the fallback chain is `val_loss → train_loss → 0.0`. If `reduce_lr_metric="val_acc"`
but there is no `val_loader`, stepping with `0.0` into a `mode="max"` scheduler
every epoch would immediately trigger LR decay. Should the scheduler simply be skipped
when the metric is unavailable, or is 0.0 the intended sentinel?

**Q2** — `EMAModel.update()` copies buffers directly (`sb.copy_(mb)`). This means
quantizer calibration state (including `search_done=True` and `search_result_lsb`)
is overwritten in the shadow model on every training step. Is the intended behaviour
that the EMA shadow always reflects the training model's calibration state, or should
calibration buffers be excluded from the buffer copy (similar to how parameters are
blended rather than copied)?

**Q3** — In `CheckpointManager._export_onnx`, the dummy input falls back to
`torch.randn(1, 3, 32, 32)` if none is provided. For ImageNet models this produces a
(1, 3, 32, 32) input, which may fail ONNX export because the model expects (1, 3, 224, 224).
The failure is caught and printed as a warning. Is the intent to always provide the
dummy input from the trainer (via `_onnx_dummy_input`), making the fallback
effectively dead code?

**Q4** — `utils/mixup.py` (the local module) and the timm `Mixup` integration in
the trainer coexist. The trainer uses timm's `Mixup` when `mixup > 0 or cutmix > 0`.
The local `apply_mixup_cutmix` appears to be an older implementation. Is the local
`mixup.py` intentionally kept as a reference or is it dead code that should be
removed?

**Q5** — `calibration.py:run_calibration` uses `brevitas.graph.calibrate.calibration_mode`.
The V2 trainer does not call `run_calibration` — instead it relies on the custom
quantizers' own first-forward calibration (the `search_done` mechanism in
`BaseQuantizer`). Is `calibration.py` still in active use (e.g., called from
`find_perfect_lsbs_imagenet_ptq.py`), or is it also legacy code from V1?

**Q6** — `QATWarmupScheduler` in `schedulers.py` uses `disable_quant(model)` which
calls `_set_quant_enabled(model, enabled=False)`. This only toggles Brevitas-level
`disable_quant` attributes, not the custom quantizer annealing state. V2 uses
`QuantizerManager().disable_quantization()` instead. Is `QATWarmupScheduler` still
intentionally maintained (for V1 compatibility), or can it be removed?

**Q7** — `_reset_and_register` sets `module.inference_sequence_id = -1` for all
registered quantizers. Sequence IDs are assigned by the manager in the first forward
pass after registration. Is there a risk that if `_reset_and_register` is called
mid-training (e.g., after a console `load-best` command), the gating offsets
(N × gap passes) are recalculated from zero, causing all quantizers to re-gate
simultaneously rather than continuing with their existing offsets?

**Q8** — `EarlyStopping.restore()` restores best weights by calling
`model.load_state_dict(self._best_weights)`. For a quantized model, this may restore
`annealing_alpha` to the value at the best epoch (not necessarily 1.0). Is this
intended? Would we want to force `annealing_alpha=1.0` after restoring (to prevent
a half-annealed model being returned from `fit()`)?

**Q9** — `test_quantizer_checkpoint_roundtrip.py` documents the "double
load_state_dict risk" and "merging state dicts safe pattern". These are important
operational patterns. Should they be promoted to actual function-level utilities
(e.g., `safe_load_checkpoint(trainer, path)`) rather than leaving users to
implement them manually from docs?

**Q10** — The diagnostics system (`utils/quantizer_diagnostics.py`) generates SVG
and PNG plots. In a long ImageNet training run, each quantizer fires diagnostics at
calibration time and post-annealing. For a ResNet with ~50 quantizers × 2 triggers,
that's ~100 plot files. Is there a mechanism to configure which quantizers emit
diagnostics (e.g., skip bias quantizers)? And is there a risk of matplotlib memory
leaks if `plt.close(fig)` is not called — the current code does call it, but this
is not tested.

---

## 11. Summary Table

| Subsystem | Tests exist? | Coverage | Priority gaps |
|-----------|-------------|----------|---------------|
| fixedpoint_per_tensor math | Yes | ~80% | activation LSB rule, AnnealingBlendFn at alpha=0 |
| fixedpoint_per_tensor ONNX | Yes | ~60% | deque reset, export_onnx_with_io |
| coefficient ONNX export | No | 0% | entire ONNX path untested |
| silu_quant | Yes | ~75% | deque reset, explicit gradient values |
| QuantizerManager | Yes | ~65% | request_snapshot, singleton isolation |
| export_onnx_with_io | No | 0% | entire canonical export path untested |
| load_embedded_io | No | 0% | - |
| QATTrainerV2 | No | 0% | phase transition, EMA, reduce_lr_metric, resume |
| EarlyStopping | No | 0% | - |
| LossPlateauDetector | No | 0% | - |
| CheckpointManager | No | 0% | top-K eviction, resume, load_best |
| EMAModel | No | 0% | - |
| MetricsTracker / AverageMeter | No | 0% | - |
| WarmupCosineScheduler | No | 0% | - |
| TrainingConsole | No | 0% | - |
| calibration.py | No | 0% | - |
| mixup.py (local) | No | 0% | - |
| DALI pipeline | No | 0% | - |
| weight_mapping.py | No | 0% | MobileNet remapping, shape handling |
| quantizer_diagnostics.py | No | 0% | - |
| QuantMobileNetV1 forward | No | 0% | - |
| QuantMobileNetV2 forward | No | 0% | - |
| QuantResNet post-calibration QAT | No | 0% | - |
| reduce_lr_metric feature | No | 0% | mode switching, fallback, firing |
| BN fusion + QAT combo | No | 0% | fused model + full QAT forward |
| PTQ chain + QAT (end-to-end) | No | 0% | all three roles + preserve flag |
