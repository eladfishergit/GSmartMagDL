# GSmartMagDL v1.0

**Generalized Smartphone-Based Magnetic Mapping with Mutual Information-Guided Deep Learning**

GSmartMagDL is a MATLAB + PyTorch toolkit for simulation-based smartphone magnetic mapping and deep-learning reconstruction under crowdsourced, multi-user and multi-device acquisition. The repository accompanies the manuscript:

> *Generalized Smartphone-Based Magnetic Mapping with Mutual Information-Guided Deep Learning*
>
> Elad Fisher, Roger Alimi, Miki Vizel, Nadav Schneider, Itzik Klein
>
> Submitted to *Computers & Geosciences*.

The framework extends smartphone magnetic reconstruction from controlled single-device scans to a more realistic crowdsourced setting, in which multiple users and different smartphone devices collect sparse, irregular and partially overlapping observations over the same region of interest. These observations are fused into a unified sparse magnetic map, which is then reconstructed using a mutual information-guided deep network.

---

## 1. Repository purpose

This repository provides:

1. MATLAB scripts for generating theoretical magnetic maps, simulating heterogeneous smartphone trajectories, sampling the magnetic field along those trajectories, and producing fused observation maps.
2. A PyTorch inference script for loading a trained GSmartMagDL/SRResNet-style model and reconstructing a magnetic map from a fused input map.
3. A pretrained example checkpoint and one representative example dataset corresponding to the qualitative Kursk-style generalization example shown in the manuscript.
4. Example output figures showing the fused input map, the GSmartMagDL prediction and the theoretical reference map.

The MATLAB files implement the simulation and data-generation pipeline described in the manuscript. The Python demo is intended as the quickest way to verify the trained model on an example fused magnetic map.

---

## 2. Method overview

The complete workflow is:

```text
Theoretical magnetic map
        ↓
Multiple heterogeneous smartphone trajectory/device realizations
        ↓
Sampling of the magnetic field along irregular paths
        ↓
Probabilistic fusion into a sparse observation map
        ↓
GSmartMagDL reconstruction network
        ↓
Continuous reconstructed magnetic map
```

In the manuscript, each theoretical magnetic map is sampled by multiple trajectory/device realizations. The resulting partial observations are merged into a fused map, which is the input to the neural reconstruction model. The network reconstructs the continuous magnetic field while improving robustness to trajectory irregularity, device variability and geographic magnetic-background changes.

---

## 3. Recommended folder structure

The repository is organized as follows:

```text
gsmartmagdl/
├── README.md
├── LICENSE
├── requirements.txt
├── test_model.py
├── example_model.pth
│
├── example_data/
│   ├── low_res/
│   │   └── Bsim_tot_1.mat
│   └── labels/
│       └── Btheor_tot_1.mat
│
├── docs/
│   ├── kursk_fused_input.png
│   ├── kursk_gsmartmagdl_prediction.png
│   └── kursk_theoretical_reference.png
│
├── matlab/
│   ├── build_full_database_MV_HRDTraj_new_algo.m
│   ├── build_single_database_element_HRDTraj1.m
│   ├── build_ground_truth1.m
│   ├── buildWorkspace.m
│   ├── buildMatrix.m
│   ├── ParallelBfield.m
│   ├── ParallelDerby.m
│   ├── ParallelBulirsch.m
│   ├── ParallelGenerateReadings.m
│   ├── CreateTrajectories_all.m
│   ├── Earth_Mag_field_influence_for_new_algo.m
│   ├── compute_krig_interp_new_algo.m
│   ├── create_result_files_for_comparison.m
│   ├── create_result_files_for_comparison_with_Krigging.m
│   ├── calculateMetrics.m
│   ├── compute_simplified_lpips.m
│   ├── kriging.m
│   ├── variogram.m
│   ├── variogramfit.m
│   ├── fminsearchbnd.m
│   ├── fminsearchcon.m
│   └── my_norm.m
│
└── results/
    └── example_run/
```

For the minimal Python demo, the example fused map and its theoretical reference should be stored as:

```text
example_data/low_res/Bsim_tot_1.mat
example_data/labels/Btheor_tot_1.mat
```

The example used in the manuscript may originate from a different internal case index, such as `Bsim_tot_7.mat` and `Btheor_tot_7.mat`. For a single-sample public demo, these files can simply be copied or renamed to `Bsim_tot_1.mat` and `Btheor_tot_1.mat` so that the provided loader can run with `--num_samples 1`.

---

## 4. Requirements

### Python

Tested with:

```text
Python >= 3.8
PyTorch >= 1.12.1
numpy
scipy
scikit-learn
matplotlib
pytorch-ignite
```

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` can contain:

```text
numpy
scipy
scikit-learn
matplotlib
torch
pytorch-ignite
```

The inference demo can run on CPU. A CUDA-enabled GPU is recommended for large-scale training or batch inference.

### MATLAB

The MATLAB data-generation scripts were developed for MATLAB R2020b or newer. No mandatory MATLAB toolbox is required for the core scripts. The MATLAB code implements the magnetic-field simulation, trajectory generation, smartphone sampling, fusion preparation, Kriging baseline and evaluation utilities used in the manuscript.

---

## 5. Quick start: run the pretrained model on the example fused map

Clone the repository and enter the main folder:

```bash
git clone https://github.com/eladfishergit/gsmartmagdl.git
cd gsmartmagdl
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Make sure the following files exist:

```text
example_model.pth
example_data/low_res/Bsim_tot_1.mat
example_data/labels/Btheor_tot_1.mat
```

Then run:

```bash
python test_model.py --num_samples 1 \
  --data_dir ./example_data/low_res/ \
  --label_dir ./example_data/labels/ \
  --model_path ./example_model.pth \
  --output_dir ./results/example_run/
```

The script will:

1. load the fused magnetic input map from `example_data/low_res/`,
2. load the corresponding theoretical reference map from `example_data/labels/`,
3. normalize the data,
4. load the pretrained model checkpoint,
5. reconstruct the magnetic map,
6. save output figures and a MATLAB `.mat` file under `results/example_run/`.

Expected output files include:

```text
results/example_run/test_case_1_sim.png
results/example_run/test_case_1_prediction.png
results/example_run/test_case_1_theory.png
results/example_run/SRResNet_new_for_analysis.mat
```

---

## 6. Input data format

The Python inference script expects MATLAB `.mat` files with the following variable names:

### Fused input map

```text
File name: Bsim_tot_1.mat
Variable:  B_sim
Shape:     300 × 300
Meaning:   fused smartphone magnetic observation map
```

### Theoretical reference map

```text
File name: Btheor_tot_1.mat
Variable:  B_theor
Shape:     300 × 300
Meaning:   theoretical magnetic-field reference map
```

The fused input map is the effective network input. It represents the result of projecting multiple irregular trajectory/device observations onto a common grid and merging them into one sparse magnetic observation map.

To evaluate a different fused map, place the corresponding `.mat` files in the same folders and use the naming convention:

```text
Bsim_tot_1.mat, Bsim_tot_2.mat, ...
Btheor_tot_1.mat, Btheor_tot_2.mat, ...
```

Then set:

```bash
python test_model.py --num_samples <N>
```

where `<N>` is the number of sequential examples available.

---

## 7. Example corresponding to the manuscript figure

The repository includes a representative example from the held-out geographic generalization setting. This example corresponds to the qualitative comparison shown in the manuscript for Kursk, where the network reconstructs the dipole anomaly structure from a sparse fused input map.

Recommended image files for the repository documentation:

```text
docs/kursk_fused_input.png
docs/kursk_gsmartmagdl_prediction.png
docs/kursk_theoretical_reference.png
```

Example display:

| Fused input map | GSmartMagDL prediction | Theoretical reference |
|---|---|---|
| ![](docs/kursk_fused_input.png) | ![](docs/kursk_gsmartmagdl_prediction.png) | ![](docs/kursk_theoretical_reference.png) |

The corresponding `.mat` files should be placed under:

```text
example_data/low_res/Bsim_tot_1.mat
example_data/labels/Btheor_tot_1.mat
```

---

## 8. MATLAB data-generation pipeline

The MATLAB files implement the simulation and data-generation pipeline used to produce the paired examples for training, validation and comparison. The main stages are:

1. **Theoretical map generation** — creates the clean magnetic-field reference map from a set of localized magnetic sources and the regional magnetic background.
2. **Workspace construction** — defines the spatial grid and sensing domain.
3. **Trajectory generation** — creates multiple non-regular user trajectories over the same region of interest.
4. **Smartphone measurement simulation** — samples the theoretical magnetic field along each trajectory and applies device-dependent effects and measurement variability.
5. **Fusion preparation** — consolidates the partial trajectory/device observations into a fused magnetic map used as network input.
6. **Baseline and metric computation** — supports Kriging comparison and quantitative evaluation.

### MATLAB file roles

| File | Role |
|---|---|
| `build_full_database_MV_HRDTraj_new_algo.m` | Main batch script for generating paired theoretical and fused/simulated magnetic maps. |
| `build_single_database_element_HRDTraj1.m` | Generates one paired data element from a theoretical map and trajectory/device sampling. |
| `build_ground_truth1.m` | Generates theoretical magnetic-field components for randomized magnetic-source configurations. |
| `buildWorkspace.m` | Builds the spatial sensing grid and workspace dimensions. |
| `buildMatrix.m` | Converts sampled values into gridded matrix representations. |
| `ParallelBfield.m` | Computes magnetic-field values for finite-cylinder source configurations. |
| `ParallelDerby.m` | Auxiliary analytical magnetic-field computation. |
| `ParallelBulirsch.m` | Auxiliary elliptic-integral computation used by the magnetic-field model. |
| `ParallelGenerateReadings.m` | Generates magnetic readings at sensor positions. |
| `CreateTrajectories_all.m` | Generates non-regular trajectory realizations. |
| `Earth_Mag_field_influence_for_new_algo.m` | Applies regional Earth magnetic-field background effects. |
| `compute_krig_interp_new_algo.m` | Computes the Kriging interpolation baseline. |
| `create_result_files_for_comparison.m` | Creates result files for comparison among reconstruction methods. |
| `create_result_files_for_comparison_with_Krigging.m` | Creates result files including the Kriging baseline. |
| `calculateMetrics.m` | Computes reconstruction metrics. |
| `compute_simplified_lpips.m` | Computes a simplified perceptual similarity score. |
| `kriging.m` | Kriging interpolation function. |
| `variogram.m` | Empirical variogram estimation. |
| `variogramfit.m` | Variogram-model fitting utility. |
| `fminsearchbnd.m` | Bounded optimization helper. |
| `fminsearchcon.m` | Constrained optimization helper. |
| `my_norm.m` | Normalization helper. |

The manuscript provides the full methodological description of the simulation, including the multi-trajectory/device model, the fusion process and the training/validation protocol.

---

## 9. Python inference script

The file `test_model.py` defines the residual reconstruction network and performs inference using a pretrained checkpoint.

Main command-line arguments:

| Argument | Default | Description |
|---|---|---|
| `--label_dir` | `./example_data/labels/` | Path to theoretical reference `.mat` files. |
| `--data_dir` | `./example_data/low_res/` | Path to fused input `.mat` files. |
| `--model_path` | `./example_model.pth` | Path to the trained PyTorch checkpoint. |
| `--output_dir` | `./results/example_run/` | Output folder for plots and `.mat` result file. |
| `--num_samples` | `10` | Number of sequential examples to evaluate. |

For a single example, use:

```bash
python test_model.py --num_samples 1
```

For a custom dataset folder:

```bash
python test_model.py \
  --data_dir /path/to/fused_maps/ \
  --label_dir /path/to/theoretical_maps/ \
  --model_path /path/to/example_model.pth \
  --output_dir /path/to/output/ \
  --num_samples 5
```

---

## 10. Reproducing the published-style qualitative example

To reproduce a qualitative example similar to the manuscript figure:

1. Place the fused map under:

```text
example_data/low_res/Bsim_tot_1.mat
```

2. Place the theoretical reference map under:

```text
example_data/labels/Btheor_tot_1.mat
```

3. Place the trained model in:

```text
example_model.pth
```

4. Run:

```bash
python test_model.py --num_samples 1
```

5. Compare the generated files:

```text
results/example_run/test_case_1_sim.png
results/example_run/test_case_1_prediction.png
results/example_run/test_case_1_theory.png
```

These correspond to the fused input, the GSmartMagDL reconstruction and the theoretical magnetic-field reference.

---

## 11. Notes on training

The manuscript describes the training protocol used for the reported experiments, including the construction of 1000 paired theoretical/fused maps, an 800/200 internal train/validation split and held-out geographic validation. The provided checkpoint can be used directly for inference on the included example fused map.

---

## 12. Troubleshooting

### No data loaded

If the script prints:

```text
No data loaded. Please check your data directories.
```

verify that the `.mat` files are stored using the expected naming convention:

```text
example_data/low_res/Bsim_tot_1.mat
example_data/labels/Btheor_tot_1.mat
```

and that they contain the variables:

```text
B_sim
B_theor
```

### Running on a headless server

If matplotlib backend errors occur on a server without a graphical display, change:

```python
mlp.use('TkAgg')
```

to:

```python
mlp.use('Agg')
```

inside `test_model.py`.

### PyTorch checkpoint mismatch

The checkpoint is expected to match the SRResNet architecture defined in `test_model.py`, including 16 residual blocks and a single-channel input/output map.

---

## 13. Citation

If you use this code, please cite the accompanying manuscript:

```bibtex
@article{fisher_gsmartmagdl_2026,
  title   = {Generalized Smartphone-Based Magnetic Mapping with Mutual Information-Guided Deep Learning},
  author  = {Fisher, Elad and Alimi, Roger and Vizel, Miki and Schneider, Nadav and Klein, Itzik},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {Submitted}
}
```

---

## 14. License

This repository is intended for academic research and reproducibility. Please include an open-source license file in the repository, such as the MIT License, unless institutional or project-specific restrictions require a different license.

---

## 15. Contact

For questions, please contact:

**Elad Fisher**  
Hatter Department of Marine Technologies, Charney School of Marine Sciences, University of Haifa  
Email: eladfisher.mail@gmail.com
