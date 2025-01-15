# Fourier Domain Verification and Certified Training
The codebase for our work exploring verification and certified training for specifications defined in the Discrete Fourier Transform coefficient space.

## Table of Contents
- [Setup](#setup)
- [Input specification design and Encoding network](#input-specification-design-and-encoding-network)
- [Usage](#usage)

## Setup
Run `python3 -m venv <virtual_env_name> && source <virtual_env_name>/bin/activate && pip3 install -r requirements.txt`.

## Input specification design and Encoding network
This constitutes the main contribution of this work and implemented as a standalone module in [augmentations](augmentations/). 
The unit tests that validate the correctness of the adopted approach are in:
1. [test_fourier.py](augmentations/test_fourier.py): test_IDFT_encoding_correctness, test_equivalance_with_FACT. Run with `python3 -m augmentations.test_fourier -v`.
2. [test_bounds.py](augmentations/test_bounds.py): test_bounds_soundness (most important), test_brightness_bounds (important for results in Figure 5), test_contrast_bounds. 
    Require [AutoLiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) installation (with the provided patch application), thereafter run with `python3 -m augmentations.test_bounds -v`.
3. [test_cosine.py](augmentations/test_cosine.py): test_IDCT_encoding_correctness. Run with `python3 -m augmentations.test_cosine -v`.

Some of the above tests save the images produced by our encoding networks E and by numpy/scipy functions that E is being compared against; these visual results can be found in /tmp/FDV_tests, particularly /tmp/FDV_tests/fourier_augs, on local machine. The run of the above tests have been tested on cpu for the submission.


#### Perturbation samples from proposed specifications
The notebook to experiment with our specifications is [FD_auto_driving_aug_demo.ipynb](augmentations/FD_auto_driving_aug_demo.ipynb). Setup the notebook with: `ipython kernel install --user --name=<virtual_env_name> && python -m ipykernel install --user --name=<virtual_env_name>`. Then run it in from this directory with: `jupyter notebook`.
All examples shown in Figure 2 and 3 in our work can be generated using this notebook. Short videos demonstrating samples of different specifications (created using the same notebook) are available at the links below.

1. Additive and Multiplicative perturbations: [demo video link](https://drive.google.com/file/d/1-k8TmNKxVjH7GKl9h_PawAUPN_1nJYdv/view?usp=sharing),
2. Kernel-based spatial convolution produced perturbations: [demo video link](https://drive.google.com/file/d/16Srk9Ifnp3nXQzeckbmZT7lKjkO2602R/view?usp=sharing),
3. Two-inputs conditional specifications: [demo video link](https://drive.google.com/file/d/1HA6rgcnDn2pvtdic-vyfswet0P67jumi/view?usp=sharing).

## Usage
1. Run network (incomplete) verification: `python3 main.py --action eval --config <path-to-evals-config> --model_path <path-to-torch-model-file> --eps_linf <desired-pixel-space-linf-eps>`.
    On running incomplete verification, the augmented model in ONNX format gets saved in the network directory and vnnlibs for Fourier-perturbed inputs get saved in verifiers/vnnlibs for complete verification. Given the ONNX model and vnnlibs, do complete verification by:
        1. Installing alpha-beta CROWN from https://github.com/Verified-Intelligence/alpha-beta-CROWN,
        2. Create a CSV listing all the vnnlibs to be verified,
        3. Then, edit and run: `python abcrown.py --config=configs/FDVR_complete_verification_example_spec.yaml`.
2. Run network trainings: `python3 main.py --config <path-to-training-config> --eps_linf <desired-pixel-space-linf-eps> --epochs <num-training-epochs> <additional-args-as-needed-see-main.py>`.
3. Run bound propgation comparison: `python3 -m augmentations/bounds_comparison.py`.

**Reproducibility instructions:** Most configs used in our work are in [configs](configs/). The trained models reported in experiments are shared [here](https://drive.google.com/drive/folders/1hPgZthwLh78jrsQToSiJiqPdoM2eT7tU?usp=drive_link) and have the config.json that was used to train them in their respective directories, and could be reproducibly (to the best of our efforts) retrained from scratch following Usage 2 for those configs. The results in Figure 3 and 4 are obtained by Usage 1 with eval config for [the single input (ball) specifications + CIFAR10 dataset](configs/CIFAR10/evals/evals.py) and [the two inputs (segment) specifications + CoDAN dataset](configs/CODAN/evals/evals.py). Usage 3 reproduces the bound widths, computation time and verified accuracy results in Table 1.
