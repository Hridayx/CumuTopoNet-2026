# Local workflows

The package provides matched 128-sample and full 1,024-sample workflows. Each uses its own configured workspace because their prepared arrays and split definitions are different.

## Matched 128-sample workflow

The matched workflow measures local preprocessing and training speed, selects a feasible sampling budget, prepares the feature cache, and runs learning-rate pilots. Validation macro-F1 selects the learning rate, with the lower value used for an exact tie. SupCon-enabled training uses a fixed coefficient of `0.5`.

Run `status --advance` after the pilots complete to prepare the main configurations. The same command reports completion state without requiring an external scheduler. Optional long-window, topology, and robustness runs can be requested after the main configurations finish.

## Full 1,024-sample workflow

The `standard-1024` suite contains the standard model comparisons and TDA sensitivity configurations. Run `prepare-sensitivity` before starting the sensitivity configurations.

The `extended-1024` suite adds held-out interference conditions, additional seeds, a spectrogram baseline, and an FP32 LSTM learning-rate comparison. After both LSTM pilots finish, `status --advance-lstm` selects the finite candidate with the lowest validation loss and prepares the final run.

Both suites use the same prepared full-window data and the same local runner.

## Operational notes

- Use a separate workspace when changing dataset paths, split settings, or feature definitions.
- Training and validation determine model and learning-rate choices; sensitivity configurations do not evaluate the test partition.
- Generated backgrounds and software interference are diagnostics, not measured RF captures.
- Use `status --recover` only when a local process has stopped and its claim needs to be inspected.
