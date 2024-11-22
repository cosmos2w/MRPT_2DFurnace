### `LatentRepresentation` Directory

This directory stores the low-dimensional representations of various physical fields extracted during the pre-training process, saved in `.csv` format.

#### File Format:

- Columns: `U` (the condition parameter of the case), `Field_info` (low-dimensional features)
- Includes separate files for training and testing sets.

#### Training Modes:

Files correspond to different pre-training modes:
- `ParallelMode`
- `MutualDecodingMode`
- `MutualEncodingMode`

### Note:

Existing files in these directories are for reference only and showcase the type of content stored ! ! !