# Dataset format

The primary task uses seven classes in this fixed order:

| Index | Label | Folder prefix |
|---:|---|---|
| 0 | Air2S | AIR |
| 1 | Inspire2 | INS |
| 2 | MavicMini | MIN |
| 3 | MavicPro | MP1 |
| 4 | MavicPro2 | MP2 |
| 5 | Phantom4 | PHA |
| 6 | ParrotDisco | DIS |

Flight-mode suffixes are `FY` (Flying), `HO` (Hovering), and `ON` (SwitchedOn). Condition directories are `CLEAN`, `BLUE`, `WIFI`, and `BOTH`. Nested directories are accepted when these components are present in the path.

Each `.dat` file contains interleaved little-endian float32 values in `I0,Q0,I1,Q1,...` order, equivalent to little-endian complex64. The byte length must be divisible by eight and must contain at least one 1,024-sample window.

The full-window workflow expects the DroneDetect V2 layout used by the provided configuration. The `inventory` command reports unrecognized paths and missing class, mode, or condition combinations before preparation begins.

## Optional session metadata

Recordings from the same acquisition session can be grouped with a CSV file:

```csv
relative_path,session_id
nested/BLUE/MP1_FY/MA1_0110_00.dat,acquisition-001
```

The CSV must cover every recording exactly once with nonempty session IDs. Session links may combine existing recording families but cannot split them. Every resulting partition must retain all seven classes.

## Sampling and caches

Each recording is divided into equal-duration, non-overlapping bins. A deterministic recording-specific random generator selects one valid 1,024-sample start in each bin. Matched 128-sample inputs use the central portion of those parent windows.

Recording IDs follow the sorted inventory order. Window IDs combine the recording ID, sample offset, and window length. Feature-cache directory names include the feature version and relevant feature settings.

Cache files are written through temporary files followed by same-directory atomic replacement. Completed cache metadata records the explicit preparation request and required filenames. Loading validates file presence, array readability, shapes, finite values, aligned window IDs, and split separation.

Use a new workspace after changing the dataset location, sampling settings, feature configuration, or split metadata.
