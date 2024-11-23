# CoFFe (Collaboratively Fused Feature)

Welcome to the CoFFe project repository! 

CoFFe stands for Collaboratively Fused Feature, a method designed for nonlinear data dimensionality reduction of multi-physics coupled systems.

CoFFe offers the following key contributions:
- High Efficiency and Flexibility: The model eliminates the need to process complete FOM data in each epoch and imposes no restrictions on input dimensions, computational domain shapes, or the discretization methods. The encoder utilizes sparse sampling to progressively cover the entire computational domain during training.
- Sparse Recognition Capability: The pre-trained model reliably identifies system states at a given condition from measurements, accommodating different sensor configurations in terms of number and location. 
- Fast Inference: In downstream tasks, the fine-tuning process converges quickly with varying types, numbers, and positions of measurements, enabling the retrieval of the unified system feature using only sparse data from partial physical field(s).
- Strong Extensibility: By capturing invariant features of multi-physical systems, the model is easily adaptable to downstream tasks requiring sparse and partial observations, such as parameter inversion, sensor arrangement optimization, and few-shot prediction of previously unobserved variables.

## Table of Contents

- [Usage Instructions](#usage-instructions)
- [Main Code Files](#main-code-files)
- [Execution Steps](#execution-steps)
  - [Pre-training](#pre-training)
  - [Export Features](#export-features)
  - [Sparse Reconstruction](#sparse-reconstruction)
- [File Paths](#file-paths)
- [Parameters and Configuration](#parameters-and-configuration)
- [Data Availability](#data-availability)
- [Contact Information](#contact-information)
- [License](#license)

## Usage Instructions

In each application folder of this project, you will find the corresponding executable codes, suggested file storage paths, and raw data.

## Main Code Files

The main code files include:

- `network.py`: Stores the neural network programs needed to be called by each training module. 
- `constant.py`: Stores the data structures used by each module.

### For Parallel Mode

1. `1_1_CoFFePretrain_ParallelMode.py`: The main program for pre-training using Parallel Mode. Running this file starts reading the dataset for pre-training data dimensionality reduction in Parallel Mode.
2. `1_2_ParallelMode_Export_Features.py`: After pre-training with Parallel Mode, this file reads the pre-trained network, outputs the obtained features, and standardizes them for further fine-tuning in sparse reconstruction tasks.
3. `1_3_ParallelMode_Sparse_Reconstruction.py`: Corresponds to the downstream sparse reconstruction task of the Parallel Mode.

### For Mutual Decoding Mode

1. `2_1_CoFFePretrain_MutualDecodingMode.py`: The main program for pre-training using Mutual Decoding Mode.
2. `2_2_MutualDecodingMode_Export_Features.py`: Reads the pre-trained network, outputs, and standardizes features after pre-training with Mutual Decoding Mode.
3. `2_3_MutualDecodingMode_Sparse_Reconstruction.py`: Corresponds to the downstream sparse reconstruction task of the Mutual Decoding Mode.

### For Mutual Encoding Mode

1. `2_1_CoFFePretrain_MutualEncodingMode.py`: The main program for pre-training using Mutual Encoding Mode.
2. `2_2_MutualEncodingMode_Export_Features.py`: Reads the pre-trained network, outputs, and standardizes features after pre-training with Mutual Encoding Mode.
3. `2_3_MutualEncodingMode_Sparse_Reconstruction.py`: Corresponds to the downstream sparse reconstruction task of the Mutual Encoding Mode.

## Execution Steps

### Pre-training

Choose the pre-training mode you wish to execute:
- **Parallel Mode**: Run `1_1_CoFFePretrain_ParallelMode.py`.
- **Mutual Decoding Mode**: Run `2_1_CoFFePretrain_MutualDecodingMode.py`.
- **Mutual Encoding Mode**: Run `3_1_CoFFePretrain_MutualEncodingMode.py`.

### Export Features

Further, run the corresponding export script:
- **Parallel Mode**: Run `1_2_ParallelMode_Export_Features.py`.
- **Mutual Decoding Mode**: Run `2_2_MutualDecodingMode_Export_Features.py`.
- **Mutual Encoding Mode**: Run `3_2_MutualEncodingMode_Export_Features.py`.

### Sparse Reconstruction

Next, run the corresponding sparse reconstruction script:
- **Parallel Mode**: Run `1_3_ParallelMode_Sparse_Reconstruction.py`.
- **Mutual Decoding Mode**: Run `2_3_MutualDecodingMode_Sparse_Reconstruction.py`.
- **Mutual Encoding Mode**: Run `3_3_MutualEncodingMode_Sparse_Reconstruction.py`.

## File Paths

During execution, the following paths are involved for data reading and storage of intermediate files:
- `data_split/`: Used to store raw data and features obtained from pre-training for use in downstream tasks.
- `LatentRepresentation/`: Used to output features obtained from pre-training.
- `Loss_csv/`: Stores training and testing error information.
- `Output_Net/`: Stores networks obtained from training.

Note: Each path contains a `Notes.md` file with more detailed descriptions.

## Parameters and Configuration

### Main Hyperparameters

During the pre-training process, the main hyperparameters include:
- `BATCH_SIZE`: The size of each training batch.
- `N_P_Selected`: The number of random sparse sensors in each pre-training epoch.
- `n_field_info`: The dimension of latent features to be extracted from field distribution data.
- `n_baseF`: The dimension of output layers of the DeepONet-based decoder.
- `Unified_Weight`: Contribution of the unified feature.
- `num_heads` and `num_layers`: FieldAttention layer parameters.

### Parameter Setting Suggestions

Suggested values for these parameters are provided at the beginning of the corresponding code files.
Users can modify and test them according to their own needs.

## Data Availability

This project provides three datasets, placed in the corresponding folders, available for download and use:
- `CoFFe_BiomassCombustion/Data/`
- `CoFFe_Electrolyzer/Data/`
- `CoFFe_ParticleDegradation/Data/`

## Contact Information

If you have any questions about the code architecture, data structure, or execution methods, please feel free to contact us.

**Responsible Person Email**: wanglz@mit.edu, silideng@mit.edu

## License

This project is licensed under the MIT License. See the LICENSE file for details.