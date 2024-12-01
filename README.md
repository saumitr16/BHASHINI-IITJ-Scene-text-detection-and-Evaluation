# Scene Text Detection and Evaluation Framework

This repository provides a comprehensive framework for scene text detection and evaluation, combining methods and insights from CRAFT (Character Region Awareness for Text detection) and TedEval, a robust text evaluation tool.

---

## Overview

The repository includes implementations and evaluations of scene text detection methods tailored for Urdu text datasets. Two main components, CRAFT and TedEval, are utilized to detect and evaluate text instances in images.

- **CRAFT**: Modified to improve text detection accuracy, with support for custom configurations and datasets.
- **TedEval**: Used for detailed evaluation of the detected text regions against ground truths.

The implementation is based on references from:
- [CRAFT by Sakura Riven](https://github.com/sakura-riven/CRAFT-pytorch)
- [TedEval GitHub Repository](https://github.com/saumitr16/BHASHINI-IITJ-Scene-text-detection-and-Evaluation)

---

## Key Components

### 1. **CRAFT and TedEval Integration**
- The notebook demonstrates how to set up and use CRAFT and TedEval for evaluating text detection on sample datasets.
- A modified `detect.py` script enhances the original CRAFT detection algorithm with:
  - Adaptive image resizing for compatibility with model constraints.
  - Detailed polygon-based bounding box generation.
  - Enhanced image-to-tensor preprocessing for optimized detection.

### 2. **Custom Urdu Dataset**
- The model has been evaluated on a comprehensive Urdu text dataset. Results show that this implementation can handle challenges such as script complexity and varying image quality.

### 3. **Evaluation Insights**
- Using TedEval, the framework provides a detailed analysis of:
  - Precision, Recall, and F1-Score for text detection.
  - Comparison of detected text regions against ground truths.

---

## File Structure

1. **Notebooks**
   - `CRAFT and TedEval.ipynb`: Demonstrates the basic usage of CRAFT and TedEval on sample datasets.
   - `BSTD_V12_CRAFT_TEDEVAL.ipynb`: Performs extensive evaluations on the full Urdu dataset.

2. **Detection Script**
   - `detect.py`: A modified detection pipeline replacing the original file in the CRAFT repository. Supports custom thresholds and box restoration techniques.

3. **References**
   - CRAFT and TedEval repositories.

---

## Usage

### Setup
1. Clone the required repositories:
   ```bash
   git clone https://github.com/sakura-riven/CRAFT-pytorch
   git clone https://github.com/saumitr16/BHASHINI-IITJ-Scene-text-detection-and-Evaluation
