# GSmartMagDL v1.0

*A MATLAB + PyTorch toolkit for generalized crowdsourced smartphone magnetic mapping with mutual-information-guided deep reconstruction.*

---

## 1. What is GSmartMagDL?

GSmartMagDL extends smartphone-based magnetic mapping from controlled single-device scans to a more realistic crowdsourced setting. The framework reconstructs a continuous magnetic map from sparse, irregular, and heterogeneous smartphone observations collected by multiple users and devices over the same region of interest.

The pipeline implemented in this repository is:

1. **Physics-consistent magnetic simulation (MATLAB)**  
   Generates theoretical magnetic maps over a representative domain and supports geographically varying background fields.

2. **Heterogeneous smartphone trajectory simulation (MATLAB)**  
   Generates multiple non-regular trajectory/device realizations per map, including sensor scaling, alignment effects, bias, orientation variability, and noise.

3. **Probabilistic occupancy-grid fusion (MATLAB / preprocessing)**  
   Merges multiple partial trajectory measurements into a unified sparse observation map.

4. **MI-guided deep reconstruction (Python / PyTorch)**  
   Uses an SRResNet-based reconstruction network with a mutual-information-guided learning formulation to emphasize spatially informative regions and preserve local magnetic anomaly structure.

5. **Generalization evaluation**  
   Supports testing on held-out geographic settings with different geomagnetic inclination, declination, and total-field magnitude, without location-specific retraining.

---

## 2. Repository contents

A recommended repository structure is:

```text
gsmartmagdl/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ example_model.pth
├─ test_model.py
├─ example_data/
│  ├─ labels/
│  │  ├─ Btheor_tot_1.mat
│  │  ├─ Btheor_tot_2.mat
│  │  └─ ...
│  └─ low_res/
│     ├─ Bsim_tot_1.mat
│     ├─ Bsim_tot_2.mat
│     └─ ...
├─ results/
│  └─ example_run/
├─ matlab/
│  ├─ simulation/
│  ├─ trajectories/
│  ├─ fusion/
│  └─ utilities/
└─ nn/
   ├─ test_model.py
   ├─ training/
   └─ checkpoints/
```

The current minimal Python inference example uses:

| File | Purpose |
|------|---------|
| `test_model.py` | Loads fused magnetic maps and theoretical maps from `.mat` files, applies the SRResNet model, and saves prediction plots and a MATLAB output file. |
| `example_model.pth` | Example trained PyTorch checkpoint for inference. |
| `example_data/labels/` | Expected folder for theoretical/ground-truth magnetic maps named `Btheor_tot_*.mat`. |
| `example_data/low_res/` | Expected folder for fused/simulated input maps named `Bsim_tot_*.mat`. |
| `results/example_run/` | Default output folder for prediction figures and `SRResNet_new_for_analysis.mat`. |

---

## 3. Requirements

### Python

Recommended environment:

```text
Python >= 3.8
PyTorch >= 1.12
numpy
scipy
scikit-learn
matplotlib
pytorch-ignite
```

Install the Python dependencies with:

```bash
pip install numpy scipy scikit-learn matplotlib torch pytorch-ignite
```

or, if a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

### MATLAB

The simulation and data-generation components were developed for:

```text
MATLAB R2020b or newer
```

No mandatory MATLAB toolboxes are required for the core simulation. The Statistics and Machine Learning Toolbox may be useful for optional analysis utilities.

### Hardware

Inference can be run on CPU or GPU. Training the full model is recommended on an NVIDIA GPU. In the experiments reported in the manuscript, training was performed on an NVIDIA A100 GPU, and inference required less than one second per fused observation map.

---

## 4. Quick start: Python inference demo

Place the trained checkpoint and the example data in the repository root using the following structure:

```text
gsmartmagdl/
├─ test_model.py
├─ example_model.pth
└─ example_data/
   ├─ labels/
   │  └─ Btheor_tot_1.mat
   └─ low_res/
      └─ Bsim_tot_1.mat
```

Run the default inference example:

```bash
python test_model.py
```

or specify custom paths:

```bash
python test_model.py \
  --label_dir ./example_data/labels/ \
  --data_dir ./example_data/low_res/ \
  --model_path ./example_model.pth \
  --output_dir ./results/example_run/ \
  --num_samples 10
```

The script expects each `.mat` file to contain the following variables:

| File pattern | Expected MATLAB variable | Meaning |
|-------------|--------------------------|---------|
| `Btheor_tot_*.mat` | `B_theor` | Theoretical/ground-truth magnetic map |
| `Bsim_tot_*.mat` | `B_sim` | Fused or simulated smartphone observation map |

The default map size used by the example script is `300 x 300`.

Outputs are saved to the selected output directory and include:

```text
test_case_*_prediction.png
test_case_*_sim.png
test_case_*_theory.png
SRResNet_new_for_analysis.mat
```

---

## 5. MATLAB simulation and data generation

The full workflow described in the manuscript relies on MATLAB code for generating the paired training and validation data. This part should include:

1. theoretical magnetic-field map generation;
2. heterogeneous multi-trajectory smartphone simulation;
3. trajectory/device parameter sampling;
4. probabilistic occupancy-grid fusion;
5. export of paired `.mat` files for Python training and testing.

The expected output format for the Python stage is:

```text
example_data/
├─ labels/
│  ├─ Btheor_tot_1.mat
│  ├─ Btheor_tot_2.mat
│  └─ ...
└─ low_res/
   ├─ Bsim_tot_1.mat
   ├─ Bsim_tot_2.mat
   └─ ...
```

where each `Btheor_tot_*.mat` file contains `B_theor`, and each `Bsim_tot_*.mat` file contains `B_sim`.

For reproducibility, the MATLAB files used to generate the theoretical maps, heterogeneous trajectories, sensor distortions, and fused observations should be included in the public repository, or the repository should clearly state which parts are not included and why.

---

## 6. Training overview

The manuscript reports training on 1000 paired examples, where each theoretical magnetic map is associated with 20 heterogeneous trajectory/device scans that are fused into a single sparse observation map.

The model uses an SRResNet-style residual reconstruction backbone. In the full training workflow, the reconstruction loss is weighted using a mutual-information-derived spatial weight map so that more informative regions contribute more strongly to optimization.

A typical training set structure is:

```text
data/
├─ train/
│  ├─ labels/
│  └─ low_res/
├─ validation/
│  ├─ labels/
│  └─ low_res/
└─ held_out_geographic_tests/
   ├─ Brazil/
   ├─ New_Zealand/
   └─ Kursk/
```

If a training script is provided, document it in the repository as:

```bash
python train_gsmartmagdl.py --config configs/train_default.yaml
```

and include the exact configuration files used to produce the manuscript results.

---

## 7. Reproducing the manuscript workflow

A complete reproduction package should allow the user to run:

1. **Generate theoretical maps and heterogeneous scans** using MATLAB.
2. **Fuse multi-trajectory observations** into sparse input maps.
3. **Train or load the GSmartMagDL model** using PyTorch.
4. **Run inference on reference and held-out geographic test sets**.
5. **Compute RMSE, SSIM, PSNR, LPIPS, and ECDF-based comparisons**.
6. **Export figures and `.mat` files for analysis.**

For the journal submission, the public repository should include at least:

- all code required to generate the synthetic dataset, or a clear explanation and sample generated data;
- the inference script;
- the trained example checkpoint;
- a small example dataset that allows `test_model.py` to run immediately;
- a clear open-source license;
- package versions or a `requirements.txt` file;
- instructions for reproducing the main numerical results.

---

## 8. Notes on reproducibility

The example inference script rescales the input and target maps using `MinMaxScaler` before applying the neural network. The same preprocessing convention should be used when preparing new `.mat` files for inference.

If running on a headless server, set a non-interactive Matplotlib backend before execution:

```bash
export MPLBACKEND=Agg
python test_model.py
```

or modify the script to use:

```python
import matplotlib
matplotlib.use("Agg")
```

before importing `matplotlib.pyplot`.

---

## 9. Code availability statement

The code associated with the manuscript is available at:

```text
https://github.com/eladfishergit/gsmartmagdl
```

Please cite the manuscript if you use this code.

---

## 10. Citation

```bibtex
@article{fisher_gsmartmagdl_2026,
  title   = {Generalized Smartphone-Based Magnetic Mapping with Mutual Information-Guided Deep Learning},
  author  = {Fisher, Elad and Alimi, Roger and Vizel, Miki and Schneider, Nadav and Klein, Itzik},
  journal = {Computers & Geosciences},
  year    = {2026},
  note    = {Submitted}
}
```

---

## 11. License

This project should be released under an open-source license. We recommend including a `LICENSE` file in the repository, for example under the MIT License, unless institutional restrictions require a different approved open-source license.

---

## 12. Contact

For questions, please contact:

Elad Fisher  
Hatter Department of Marine Technologies, Charney School of Marine Sciences, University of Haifa  
Email: eladfisher.mail@gmail.com
