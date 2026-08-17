# AI Model Documentation & Guide

This directory contains the machine learning model files and class configuration mappings for the **AI-Based Skin Disease Detection** platform.

---

## 1. Where to Place Your Trained Model File

You can place your trained deep learning model directly in this directory with one of the following filenames:

- `model/model.keras` *(Recommended for modern TensorFlow 2.16+ / Keras 3)*
- `model/model.h5` *(Legacy HDF5 format)*

When the Flask backend starts, it automatically checks this folder:
1. It tries loading `model/model.keras`.
2. If not found, it attempts loading `model/model.h5`.
3. If neither is present, the app starts in **"AI Model Not Connected"** mode and clearly informs the user on the UI without fabricating fake results.

---

## 2. Generating a Starter Model (Quick Test)

To generate a real, ready-to-run MobileNetV2 architecture configured for the 7 HAM10000 classes, run the included generator script:

```bash
python create_sample_model.py
```

This creates `model/model.keras` instantly with standard ImageNet base transfer learning weights, enabling full testing of the preprocessing, inference, and visualization pipeline.

---

## 3. Supported Model Architectures & Transfer Learning

You can train any Convolutional Neural Network (CNN) in Keras or PyTorch (exported to ONNX/Keras/HDF5) using popular architectures:
- **MobileNetV2** (Fast, lightweight, ideal for web deployment)
- **ResNet50 / ResNet50V2** (Deep residual architecture)
- **EfficientNetB0 - B4** (State-of-the-art accuracy-to-parameter efficiency)
- **VGG16 / VGG19** (Classic deep CNN benchmark)
- **Custom Sequential CNN**

### Standard Model Input & Output Specifications:
- **Input Shape**: `(224, 224, 3)` (RGB image, resized & normalized to `[0.0, 1.0]` or standard `-1.0 to 1.0`)
- **Output Shape**: `(1, 7)` with `Softmax` activation returning probabilities for each class.

---

## 4. How to Customize Disease Class Names

To change, rename, or expand the list of predicted diseases:
1. Open `model/classes.json`.
2. Update the `classes` array. Ensure the indices (`id: 0, 1, 2...`) match your model's output neurons in training order.
3. Update the `name`, `short_name`, `category`, `risk_level`, and `description` fields.
4. Restart the Flask application (`python app.py`).

The frontend and API will dynamically read and present your new labels!

---

## 5. Dataset Sources for College Project Reference

- **HAM10000 Dataset**: 10,015 dermatoscopic images across 7 major diagnostic categories collected by the Medical University of Vienna and Cliff Rosendahl in Queensland.
  - [Harvard Dataverse Link](https://doi.org/10.7910/DVN/DBW86T)
  - [Kaggle Dataset Link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **ISIC Archive**: The International Skin Imaging Collaboration repository.
  - [ISIC Archive](https://www.isic-archive.com/)
