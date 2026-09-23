# Method and model details

## Feature processing

Each waveform is centered and divided by its centered RMS using complex128 arithmetic. Inputs with power at or below `1e-24` map to zeros.

The higher-order-cumulant vector is `abs(C20), abs(C40), abs(C41), C42, abs(C60), C63`. Training-only scaling uses unit scale for effectively constant columns.

The temporal representation contains amplitude, principal phase divided by pi, and unwrapped phase increments divided by pi. The first increment is zero. Constant phase augmentation is sampled independently for each training example.

The TDA representation embeds normalized amplitude with dimension 3 and delay 5. Ripser computes H0 and H1 persistence diagrams. Infinite and zero-persistence intervals are excluded. Persistence-weighted Gaussian densities are sampled on a fixed image grid; constant inputs and empty diagrams produce zero images.

## Models

- HOC encoder: two linear blocks with normalization and nonlinear activation.
- Temporal encoder: five residual depthwise-convolution blocks with increasing dilation.
- TDA encoder: convolutional blocks followed by global pooling.
- Fusion models: concatenate the selected encoders, project to a shared embedding, and classify the result.
- Raw-IQ model: applies the temporal encoder directly to real and imaginary channels.
- LSTM model: one recurrent layer followed by a dense embedding and classifier.
- Spectrogram model: applies a compact convolutional network to log-power STFT inputs.

Parameter counts are computed from constructed models and stored with each completed run.

## Training

Training uses AdamW with a cosine learning-rate schedule, early stopping, deterministic seeded sampling, and optional CUDA mixed precision. Checkpoints include model, optimizer, scheduler, scaler, random-state, sampler, and progress data so interrupted runs can resume.

Cross-entropy remains unweighted in the matched workflow. SupCon-enabled configurations use a fixed coefficient of `0.5` and temperature `0.07`. The loss excludes self-comparisons and anchors without positive partners.

Learning-rate pilots compare `0.0003` and `0.001` for the full, temporal, raw-IQ, and LSTM models. Main runs start from fresh model initialization after the learning rate is selected.

## Evaluation

Reports include overall accuracy, balanced accuracy, macro-F1, per-class metrics, subgroup metrics, confusion matrices, paired recording-group intervals, and processing measurements when available.

Balanced accuracy averages recall over classes with support. Macro-F1 averages over the declared classes and assigns zero when a class is absent from both labels and predictions. Subgroup reports always include support counts.

Latency measurements separate CPU feature extraction, batch-one network inference, end-to-end resident-waveform processing, batch throughput, GPU allocation, and process memory. They do not include RF acquisition or disk ingestion.
