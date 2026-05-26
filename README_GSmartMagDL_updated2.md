# GSmartMagDL

**Generalized Smartphone-Based Magnetic Mapping with Mutual Information-Guided Deep Learning**

GSmartMagDL is a computational framework for reconstructing continuous magnetic-field maps from sparse, heterogeneous, and crowdsourced smartphone magnetometer observations. The method is designed for realistic acquisition conditions in which multiple users and devices traverse the same region of interest along irregular, non-structured trajectories. The fused observation map is provided to a deep reconstruction network guided by a mutual-information-based loss, with the goal of preserving localized magnetic anomaly structure under heterogeneous input and geographically varying magnetic backgrounds.

![Conceptual workflow](docs/concept_gsmartmagdl.png)

The repository contains the MATLAB simulation and data-generation components used to create theoretical magnetic maps, heterogeneous trajectory/device samples, and fused observation maps, together with a Python/PyTorch inference example for loading a fused map and reconstructing the corresponding magnetic map using a trained GSmartMagDL model.

---

## 1. Repository purpose

This repository supports the manuscript:

**Generalized Smartphone-Based Magnetic Mapping with Mutual Information-Guided Deep Learning**

It provides the code components needed to demonstrate the GSmartMagDL reconstruction workflow. The MATLAB files implement the data-generation pipeline used to create theoretical magnetic maps, simulate multiple heterogeneous smartphone trajectories, sample the magnetic field along these trajectories, and fuse the partial observations into a single observation map. This fused map is the input to the reconstruction network.

The Python file `test_model.py` contains the PyTorch implementation needed to define and run the GSmartMagDL reconstruction network. The trained network weights are provided in `example_model.pth`. To make the repository directly runnable, we also provide one example test case from the Kursk validation setting:

- `Bsim_tot_1.mat` — the fused observation map used as the network input.
- `Btheor_tot_1.mat` — the corresponding theoretical magnetic map, provided only as a reference for visual comparison and evaluation.

Running the Python example loads the fused map, applies the trained GSmartMagDL model, and saves the reconstructed magnetic map together with plots of the input and reference maps.

---

## 2. Method overview

GSmartMagDL addresses magnetic reconstruction under crowdsourced acquisition, where measurements are collected opportunistically by different smartphones and users along irregular trajectories. The framework combines:

1. **Crowdsourced magnetic acquisition model** — multiple heterogeneous trajectory/device scans over the same region of interest.
2. **Probabilistic fusion** — consolidation of partial trajectory measurements into a unified sparse observation map.
3. **MI-guided deep reconstruction** — reconstruction of a continuous magnetic map using a mutual-information-guided learning objective that emphasizes spatially informative regions.
4. **Geographic generalization** — application of a trained model to held-out geomagnetic settings without location-specific retraining.

![GSmartMagDL framework](docs/gsmart_framework.png)

---

## 3. Repository structure

A suggested folder structure is:

```text
gsmartmagdl/
├── README.md
├── LICENSE
├── requirements.txt
├── example_model.pth
├── test_model.py
│
├── example_data/
│   ├── low_res/
│   │   └── Bsim_tot_1.mat
│   └── labels/
│       └── Btheor_tot_1.mat
│
├── matlab/
│   ├── buildWorkspace.m
│   ├── buildMatrix.m
│   ├── build_ground_truth1.m
│   ├── build_single_database_element_HRDTraj1.m
│   ├── build_full_database_MV_HRDTraj_new_algo.m
│   ├── ParallelBfield.m
│   ├── ParallelBulirsch.m
│   ├── ParallelDerby.m
│   ├── ParallelGenerateReadings.m
│   ├── CreateTrajectories_all.m
│   ├── Earth_Mag_field_influence_for_new_algo.m
│   ├── compute_krig_interp_new_algo.m
│   ├── compute_simplified_lpips.m
│   ├── create_result_files_for_comparison.m
│   ├── create_result_files_for_comparison_with_Krigging.m
│   ├── calculateMetrics.m
│   ├── kriging.m
│   ├── variogram.m
│   ├── variogramfit.m
│   ├── fminsearchbnd.m
│   ├── fminsearchcon.m
│   └── my_norm.m
│
└── docs/
    ├── concept_gsmartmagdl.png
    ├── gsmart_framework.png
    ├── kursk_fused_input.png
    ├── kursk_gsmartmagdl_prediction.png
    └── kursk_theoretical_reference.png
```

---

## 4. MATLAB data-generation pipeline

The MATLAB files implement the simulation and data-generation stages used to construct the paired examples for training, validation, and testing. These scripts generate theoretical magnetic fields, create heterogeneous user trajectories, simulate smartphone sampling and device effects, and produce fused observation maps.

The main MATLAB components are:

| Component | Main files | Role |
|---|---|---|
| Workspace and grid construction | `buildWorkspace.m`, `buildMatrix.m` | Defines the spatial grid and simulation domain. |
| Magnetic-field generation | `build_ground_truth1.m`, `ParallelBfield.m`, `ParallelBulirsch.m`, `ParallelDerby.m` | Computes theoretical magnetic maps from localized magnetic sources and the background field. |
| Trajectory generation | `CreateTrajectories_all.m` | Generates multiple irregular trajectories that represent unconstrained user motion. |
| Smartphone measurement simulation | `ParallelGenerateReadings.m`, `Earth_Mag_field_influence_for_new_algo.m`, `build_single_database_element_HRDTraj1.m` | Samples the theoretical magnetic field along trajectories and applies device-dependent measurement effects. |
| Dataset construction | `build_full_database_MV_HRDTraj_new_algo.m` | Builds paired examples consisting of fused smartphone observations and theoretical reference maps. |
| Baselines and evaluation | `compute_krig_interp_new_algo.m`, `kriging.m`, `variogram.m`, `variogramfit.m`, `calculateMetrics.m`, `compute_simplified_lpips.m`, `create_result_files_for_comparison.m`, `create_result_files_for_comparison_with_Krigging.m` | Computes baseline reconstructions and evaluation metrics used for comparison. |

The output of the MATLAB pipeline is a set of paired `.mat` files. For the Python example, the relevant files are:

```text
example_data/low_res/Bsim_tot_1.mat      # fused observation map, variable name: B_sim
example_data/labels/Btheor_tot_1.mat     # theoretical reference map, variable name: B_theor
```

---

## 5. Python inference example

The Python script `test_model.py` loads a fused magnetic observation map, applies the trained GSmartMagDL model, and saves the reconstructed map together with the input and theoretical reference plots.

### Input files

The script expects the following default folders:

```text
example_data/low_res/
example_data/labels/
```

Each fused input map should be named:

```text
Bsim_tot_<index>.mat
```

and should contain the variable:

```text
B_sim
```

Each theoretical reference map should be named:

```text
Btheor_tot_<index>.mat
```

and should contain the variable:

```text
B_theor
```

For the provided Kursk example, place the files as:

```text
example_data/low_res/Bsim_tot_1.mat
example_data/labels/Btheor_tot_1.mat
```

The trained model checkpoint should be placed in the repository root:

```text
example_model.pth
```

---

## 6. Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` is:

```text
numpy
scipy
scikit-learn
matplotlib
torch
```

The code can run on CPU for the provided inference example. A CUDA-enabled GPU is recommended for large-scale training or batch inference.

---

## 7. Running the Kursk example

After placing the model and example data in the folders described above, run:

```bash
python test_model.py --model_path ./example_model.pth --num_samples 1
```

The default command uses:

```text
--data_dir   ./example_data/low_res/
--label_dir  ./example_data/labels/
--output_dir ./results/example_run/
```

To use a different fused map or a different folder, pass the paths explicitly:

```bash
python test_model.py \
  --model_path ./example_model.pth \
  --data_dir ./example_data/low_res/ \
  --label_dir ./example_data/labels/ \
  --output_dir ./results/kursk_example/ \
  --num_samples 1
```

The output folder will include:

```text
test_case_1_sim.png          # fused observation map provided as model input
test_case_1_prediction.png   # GSmartMagDL reconstructed magnetic map
test_case_1_theory.png       # theoretical reference map
SRResNet_new_for_analysis.mat
```

---

## 8. Example corresponding to the manuscript figure

The provided example corresponds to the Kursk held-out geographic validation case shown in the manuscript. The fused map is the input to the trained GSmartMagDL model, while the theoretical map is used as a reference for comparison.

| Fused input map | GSmartMagDL reconstruction | Theoretical reference |
|---|---|---|
| ![Kursk fused input](docs/kursk_fused_input.png) | ![Kursk prediction](docs/kursk_gsmartmagdl_prediction.png) | ![Kursk theoretical reference](docs/kursk_theoretical_reference.png) |

---

## 9. Network architecture

The reconstruction backbone follows the modified SRResNet architecture used in the manuscript.

| Layer | Input shape | Output shape | Details |
|---|---:|---:|---|
| Initial convolution | 1 × 256 × 256 | 64 × 256 × 256 | Kernel 9 × 9, stride 1, padding 4 |
| ReLU | 64 × 256 × 256 | 64 × 256 × 256 | Activation |
| Residual blocks ×16 | 64 × 256 × 256 | 64 × 256 × 256 | Kernel 3 × 3 |
| Upsampling convolution 1 | 64 × 256 × 256 | 256 × 256 × 256 | Kernel 3 × 3, stride 1, padding 1 |
| ReLU | 256 × 256 × 256 | 256 × 256 × 256 | Activation |
| Upsampling convolution 2 | 256 × 256 × 256 | 1 × 256 × 256 | Kernel 9 × 9, stride 1, padding 4 |

The model contains 53 layers and approximately 1.36 million trainable parameters. In GSmartMagDL, this reconstruction backbone is trained with an MI-guided weighted loss that emphasizes spatially informative regions under heterogeneous crowdsourced input.

---

## 10. Code availability

The repository accompanies the manuscript and includes the main MATLAB and Python code used for simulation-based data generation, fused-map construction, baseline comparison, and neural-network inference.

The source code is available at:

```text
https://github.com/eladfishergit/gsmartmagdl
```

---

## 11. Citation

If you use this code, please cite the associated manuscript:

```bibtex
@article{fisher_gsmartmagdl,
  title   = {Generalized Smartphone-Based Magnetic Mapping with Mutual Information-Guided Deep Learning},
  author  = {Fisher, Elad and Alimi, Roger and Vizel, Miki and Schneider, Nadav and Klein, Itzik},
  journal = {Computers & Geosciences},
  year    = {submitted}
}
```

---

## 12. License

This repository is intended for academic research and reproducibility. Please see the `LICENSE` file for terms of use.

---

## 13. Contact

For questions, please contact:

**Elad Fisher**  
Hatter Department of Marine Technologies, Charney School of Marine Sciences, University of Haifa  
Email: `eladfisher.mail@gmail.com`
