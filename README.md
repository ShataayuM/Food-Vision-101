# Food Vision 101 🍱🍕🥗

An image classification project that uses **TensorFlow** and **Transfer Learning** to classify images into **101 different food categories**. The model is trained on the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

---

## 🔑 Key Details

- **Base Model:** `EfficientNetB0` (pre-trained on ImageNet)
- **Framework:** TensorFlow/Keras
- **Transfer Learning:** Fine-tuned the model's top layers for high performance on food classification.
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (learning rate: `0.0001`)
- **Dataset:** 75,750 training images and 25,250 test images from **Food-101** dataset
- **Augmentation:** Horizontal flip, rotation, zoom, and brightness adjustments

---

## 📊 Performance Metrics

| Metric     | Validation/Test |
|------------|-----------------|
| Accuracy   | **~85–90%**     |
| Precision  | **~84–89%**     |
| Recall     | **~85–90%**     |
| F1 Score   | **~85–89%**     |

*(Final values may vary depending on training time and hyperparameters.)*

---

## 🧪 Features

- Classifies images into 101 popular food categories (e.g., pizza, burger, sushi, etc.)
- Uses transfer learning for faster convergence
- Custom image preprocessing and data pipeline using TensorFlow `tf.data`
- Model checkpointing and performance logging

---

## 📁 Files in This Repository

- `FoodVision101.ipynb` — Main Jupyter notebook with training code
- `food_model.h5` — Saved Keras model (optional)
- `README.md` — Project description and usage
- `food_classes.txt` — List of 101 food class labels

---

## 🛠️ Future Improvements

- Deploy the model using Streamlit or Flask as a web app
- Convert model to TensorFlow Lite for mobile
- Further tuning with EfficientNetB1/B3 for performance boost

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 🙋‍♂️ Author

Made with ❤️ by **Shataayu Mohanty**

---

## 📷 Sample Output

![Sample prediction](sample_output.png) *(Add a sample image if possible)*

