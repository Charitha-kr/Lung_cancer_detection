
# Lung Cancer Detection Using Convolutional Neural Network (CNN)

This project implements a **Convolutional Neural Network (CNN)** to classify lung histopathology images into three categories:

* **Normal (lung_n)**
* **Lung Adenocarcinoma (lung_aca)**
* **Lung Squamous Cell Carcinoma (lung_scc)**

The goal is to demonstrate how **deep learning** can assist in early cancer detection using image-based analysis.

---

##  Dataset Structure

Your dataset should follow this folder structure:

```
lung_image_sets/
│
├── lung_n/       # Normal lung tissue images
├── lung_aca/     # Lung adenocarcinoma images
└── lung_scc/     # Lung squamous cell carcinoma images
```

Each folder contains raw histopathology images.

---

##  Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* OpenCV (cv2)
* scikit-learn
* Matplotlib

---

##  CNN Architecture Overview

The model consists of:

* 3 Convolutional + MaxPooling layers
* ReLU activations
* Flatten layer
* Dense layer (128 units)
* Dropout (0.5)
* Output Dense layer with Softmax activation (3 classes)

Loss Function: **Categorical Crossentropy**
Optimizer: **Adam**

---

##  Code Overview

### **1. Load and preprocess images**

* Reads all images from folders
* Resizes them to **224×224**
* Converts BGR → RGB
* Normalizes pixel values
* One-hot encodes labels

### **2. Split into training & testing**

* 80% training
* 20% testing

### **3. Train the CNN model**

* Trains for **10 epochs**
* Tracks accuracy and loss

### **4. Evaluate on test data**

Your model reported:

```
Accuracy ≈ 91.9%
Loss ≈ 0.215
```

---

## Visualizations Included

This project includes 6 types of visualizations:

1. Training Accuracy
2. Validation Accuracy
3. Training Loss
4. Validation Loss
5. Confusion Matrix
6. ROC Curves

These plots help analyze training progress and model performance.

---

## 🛠 How to Run This Project

### **1. Clone the repository**

```bash
git clone https://github.com/your-username/lung-cancer-cnn.git
cd lung-cancer-cnn
```

### **2. Install required libraries**

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn seaborn
```

### **3. Update dataset path**

In the script, edit:

```python
path = r"C:\Users\your_username\path_to_dataset\lung_image_sets"
```

### **4. Run the training script in Jupyter Notebook**

```bash
jupyter notebook
```

---

##  Future Improvements

* Add data augmentation
* Use transfer learning (e.g., VGG16, ResNet50)
* Improve dataset size and quality
* Add Grad-CAM for visual explainability

---

