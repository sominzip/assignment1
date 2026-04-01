# Adversarial Attack Assignment

## Overview

This project implements adversarial attacks on image classification models using PyTorch.

The following attacks are included:

* FGSM (Fast Gradient Sign Method)

  * Untargeted
  * Targeted
* PGD (Projected Gradient Descent)

  * Untargeted
  * Targeted

---

## Requirements

Install dependencies using:

pip install -r requirements.txt

---

## How to Run

Run the main script:

python test.py

---

## Outputs

* Attack success rates will be printed in the console
* Visualization images will be saved in the `results/` folder

---

## Results

Examples of generated outputs:

* Original image
* Adversarial image
* Perturbation (scaled for visualization)

---

## Notes

* The model is trained from scratch (no pretrained model used)
* CIFAR-10 dataset is used for experiments
* Perturbations are scaled for better visualization

---
