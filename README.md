# 🍱 Food Vision 101 — Image Classification Project

An image classification deep learning project built using **TensorFlow**, **Transfer Learning**, and **Mixed Precision Training** to classify images into **101 different food categories**. The model is trained on the popular [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

---

## 🔑 Key Details

- **Base Model:** `EfficientNetB0` (pre-trained on ImageNet)
- **Framework:** TensorFlow/Keras
- **Transfer Learning:** Fine-tuned the model's top layers
- **Mixed Precision Training:** Enabled to speed up training with minimal loss in accuracy
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (learning rate: `0.0001`)
- **Early Stopping:** Used to prevent overfitting and save optimal model
- **Dataset:** 75,750 training images and 25,250 test images from **Food-101** dataset

---

## 📊 Performance Metrics

| Metric     | Validation/Test |
|------------|-----------------|
| Accuracy   | **77.798%**     |
| Precision  | **77.805%**     |
| Recall     | **77.798%**     |
| F1 Score   | **77.689%**     |

> *Final values may vary depending on training duration and hyperparameter configuration.*

---

## 🧪 Features

- Classifies images into 101 popular food categories (e.g., pizza, burger, sushi)
- Transfer learning used for faster convergence and higher accuracy
- Mixed precision training for reduced memory usage and improved performance
- Custom image preprocessing pipeline using TensorFlow `tf.data`
- Model checkpointing and early stopping for better generalization
- TensorBoard integration for visualizing training logs

---

## 📁 Files in This Repository

- `Food_Vision_101.ipynb` — Main Jupyter notebook with training and evaluation code
- `fine_tuned.h5` — Saved complete Keras model
- `efficientnetb0_food101_fine_tuned_weights.weights.h5` — Saved model weights
- `README.md` — Project documentation and usage instructions
- `food_classes.txt` — List of 101 food class labels

---

## 🚀 Future Improvements

- Deploy the trained model using **Streamlit** or **Flask** as a web application
- Convert the model to **TensorFlow Lite** for mobile/edge deployment
- Upgrade to **EfficientNetB1/B3** or better architectures for increased accuracy
- Implement **data augmentation** techniques (random flip, zoom, rotation) to enhance generalization

---

## ▶️ Run in Google Colab

Click below to open this notebook in **Google Colab**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bMRo1iEDCKMpz2B5-uVF05JgV8zfLp4L?usp=sharing)

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---



