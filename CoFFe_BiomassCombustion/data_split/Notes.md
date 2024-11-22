
### `data_split` Directory

The `data_split` directory stores the raw data and the features data at intermediate stages of pre-training. 
All data files are saved in `.pic` format and can be read using the `pickle` package.

#### Contents:

- **`data_split.pic`**: This file contains six tensors for the Biomass combustion application:
  - `U_train`, `U_test`: Tensors storing the condition parameters for each case.
  - `Y_train`, `Y_test`: Tensors storing spatial coordinates for each case.
  - `G_train`, `G_test`: Tensors storing the specific values of various physical quantities.
  - Total cases: 301
  - field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2']

- **`data_split_NO.pic`**: Structurally similar to `data_split.pic`, this file also includes tensors for the distribution of nitrogen oxides in addition to the combustion-related physical fields.
  - Additional physical fields: Three tensors related to nitrogen oxides distribution.
  - Total cases: 41
  - field_names = ['T', 'P', 'Vx', 'Vy', 'O2', 'CO2', 'H2O', 'CO', 'H2', 'NH3', 'HCN', 'NO']

### Note:

Existing files in these directories are for reference only and showcase the type of content stored ! ! !