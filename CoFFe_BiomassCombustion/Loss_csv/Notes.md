### `Loss_csv` Directory

This directory contains all training and testing loss data for the pre-training and sparse reconstruction tasks, stored in `.csv` format.

#### File Format:

- First Line: `Unified_weight = 5.0, n_field_info = ...` (This line stores training details and notes.)
- Subsequent Lines: `Epoch, Overall_Train_loss, Overall_Test_loss, ...`

#### Loss Files:

Different pre-training modes generate different numbers of loss files, primarily named with a `_Unified` suffix, which contains the reconstruction error information for the Unified system feature.

### Note:

Existing files in these directories are for reference only and showcase the type of content stored ! ! !