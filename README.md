# bolt-fracture-1d-detection
1D waveform-based bolt fracture detection  (Laboratory research project under Professor Kai Xie)

This repository contains the optimal deep learning pipeline for 1D waveform classification to detect bolt fractures, achieving robust performance under complex laboratory conditions.

## Project Overview
The primary objective of this project is to classify 1D waveform data (ultrasonic/vibration echo sequences) to determine the structural integrity of bolts. 

During the development phase, an extensive comparative analysis was conducted involving **1D CNN**, **InceptionTime**, and **ConvNeXt1D** architectures. This repository highlights the final, optimal implementation centered around the high-capacity model, supported by rigorous experimental results.

## Model Selection & Results Summary
Through rigorous comparative experiments, the advanced architecture (detailed in `model.py`) demonstrated superior feature extraction capabilities on complex 1D morphological signals compared to baseline models.

**Key Experimental Findings:**
*(Note: Detailed performance charts and confusion matrices are available in the `/results` directory.)*
* **Baseline (1D CNN):** Achieved moderate accuracy but struggled to capture subtle long-term decay patterns in the waveforms.
* **InceptionTime:** Improved multi-scale feature capture but was computationally heavy for this specific signal length.
* **Optimal Architecture (Current Implementation):** Leveraged modern convolutional design principles, significantly improving accuracy and demonstrating remarkable robustness against signal variations.

## Deep Dive: Data Strategy & Noise Handling
A critical component of this project's success lies in the nuanced data preprocessing strategy, specifically designed for 1D structural signals.

**The Gaussian Noise Dilemma:** Standard data augmentation techniques often default to adding Gaussian noise to improve generalization. However, in this specific domain of bolt fracture detection, empirical analysis revealed that **blindly injecting Gaussian noise masks critical structural features**—such as the subtle micro-oscillations that distinguish a normal bolt from one with early-stage fractures. 

Therefore, this pipeline strategically excludes Gaussian noise from the augmentation process, opting instead for domain-specific transformations (like magnitude scaling and precise temporal shifts) that preserve the integrity of the crucial failure signatures. This decision was pivotal in pushing the model's accuracy past the 80% threshold.

## Repository Structure
* `model.py`: Contains the implementation of the optimal network architecture (ResNet1D + MultiScale + SEBlock).
* `dataset.py`: Data loading, preprocessing (normalization, padding), and custom augmentation logic.
* `train.py`: Main training and evaluation loop.
* `predict.py`: Standalone inference script for single-file prediction.
* `/results`: Visualizations of the comparative analysis, training curves, ROC, and confusion matrices.
