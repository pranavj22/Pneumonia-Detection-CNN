# 🫁 Pneumonia Detection AI from Chest X-Rays

## Overview
This project is a Deep Learning computer vision model built to detect signs of Pneumonia in chest X-ray images. It utilizes a Convolutional Neural Network (CNN) trained on thousands of medical images to classify lungs as either "Normal" or showing signs of "Pneumonia." The model is wrapped in an interactive web interface using Gradio, allowing users to upload an X-ray and receive an instant prediction with a confidence score.

![Pneumonia AI Web App]<img width="1883" height="872" alt="image" src="https://github.com/user-attachments/assets/6f23ff75-df90-4fa6-b8c6-f0cd9a22c571" />


---

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Web UI:** Gradio
* **Data Processing:** NumPy, PIL
* **Environment:** Google Colab (GPU Training)

---

## 📊 The Dataset
The model was trained on the widely used **Chest X-Ray Images (Pneumonia)** dataset from Kaggle, originally sourced from pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center. 

* **Total Images:** 5,863 (JPEG format)
* **Categories:** 2 (Normal, Pneumonia)
* **Dataset Link:** [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 🧠 Model Architecture
The custom CNN architecture was designed to extract complex spatial hierarchies from medical imagery while preventing overfitting. 

1. **Preprocessing:** Images resized to 150x150 pixels and normalized (pixel values scaled to 0-1).
2. **Feature Extraction (Convolutional Blocks):**
   * `Conv2D` (32 filters) + `MaxPooling2D`
   * `Conv2D` (64 filters) + `MaxPooling2D`
   * `Conv2D` (128 filters) + `MaxPooling2D`
3. **Classification Head:**
   * `Flatten` layer to convert 2D feature maps to 1D vectors.
   * `Dense` layer (128 neurons, ReLU activation).
   * `Dropout` (0.5) to randomly deactivate 50% of neurons and prevent memorizing the training data.
   * `Dense` output layer (1 neuron, Sigmoid activation for binary classification).

**Performance:** The model achieved an accuracy of **~97%** on the training data.
