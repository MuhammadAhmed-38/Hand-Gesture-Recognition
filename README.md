# 🧠 Real-Time Hand Gesture Recognition Using Deep Learning  
## A Comparative Study of Custom CNN, VGG16, and InceptionV3

---

## 🚀 Project Overview

This project presents a **rigorous, end-to-end deep learning system** for **real-time hand gesture recognition**, combining **model design, transfer learning, quantitative evaluation, and live deployment**.

Unlike typical academic demos, this work:
- Designs a **custom CNN from scratch**
- Applies **transfer learning** using **VGG16** and **InceptionV3**
- Performs **fair, metric-driven comparison**
- Implements **real-time webcam-based inference**
- Achieves **>95% classification accuracy** on a **balanced multi-class dataset**

This project reflects **production-oriented engineering practices**, not a proof-of-concept notebook.

---

## ✋ Problem Statement

Hand gesture recognition is a core **Human–Computer Interaction (HCI)** problem with applications in:
- Touchless interfaces
- Assistive technologies
- AR/VR systems
- Robotics
- Sign language interpretation

The key challenge is achieving **high accuracy without sacrificing real-time performance**.

---

## 🧩 Dataset

- **Source:** Kaggle – Leap Gesture Recognition Dataset + My Own Dataset (Combination)
- **Size:** 16 **GB**
- **Total Images:** 16,000  
- **Classes:** 8  
- **Distribution:** Perfectly balanced (2,000 images per class)

|Class ID| Gesture|
|--------|--------|
|   01   |  Palm  |
|   02   |    L   |
|   03   |  Fist  |
|   04   |  Thumb |
|   05   |  Index |
|   06   |   OK   |
|   07   |    C   |
|   08   |    V   |

> Balanced data ensures results reflect **model quality**, not dataset bias.

---

## 🏗️ Model Architectures

### 🔹 Custom CNN (Designed from Scratch)
**Purpose:** Speed, control, and deployability

- 6 convolutional layers
- Progressive filter scaling (32 → 512)
- MaxPooling for spatial reduction
- Dropout for regularisation
- Optimised for real-time inference

**Strength:**  
A hand-built CNN that **nearly matches pretrained ImageNet models** — a strong indicator of solid architecture design.

---

### 🔹 VGG16 (Transfer Learning)
- ImageNet-pretrained weights
- Frozen convolutional base
- Custom classification head
- RGB input processing

**Strength:**  
Stable, reliable baseline with strong feature extraction.

---

### 🔹 InceptionV3 (Transfer Learning)
- Multi-branch convolutional architecture
- Deep hierarchical feature learning
- Lowest validation loss among all models

**Strength:**  
Best generalisation, especially under noisy real-world input.

---

## 📊 Quantitative Results

|      Model      | Test Accuracy |
|-----------------|---------------|
| **Custom CNN**  | **95.94%**    |
| **VGG16**       | **95.63%**    |
| **InceptionV3** | **94.19%**    |

All models were trained and evaluated using the **same train–validation–test split and preprocessing pipeline** to ensure a fair comparison.

### Interpretation
- Custom CNN performance **competes with ImageNet-pretrained models**
- Pretrained models are powerful but **not automatically superior**
- Architecture choice must consider **speed vs generalization trade-offs**

---

## 🎥 Real-Time Gesture Recognition

The system supports **live webcam inference** with:
- HSV-based skin segmentation
- Noise removal via morphological operations
- Dynamic hand ROI extraction
- Confidence-aware predictions
- Real-time visualisation

> This is **deployment-ready logic**, not offline-only evaluation.

---

## 🧠 Critical Insights

- RGB input significantly outperformed grayscale
- `categorical_crossentropy` produced superior convergence
- Transfer learning accelerates training but increases model size
- Model depth alone does **not** guarantee better performance
- Custom architectures remain highly competitive when tuned correctly

---

## 📂 Project Structure (High-Level)

```
Hand gesture/
│
├── Artificial Intelligent.ipynb # Full training, evaluation, and deployment pipeline
├── AI_Coursework_CN7023_Term2_2025_v01.docx
├── README.md
└── reduced_data/ # Dataset (intentionally excluded due to size)

```

## 📸 Result's Screenshots

## 🧪 Data Preparation & Validation

### Dataset Sample & Distribution
![Sample Image](Results%20Images/Sample%20Image.png)
![Data Balancing](Results%20Images/Data%20Balancing.png)

### Train–Test–Validation Split
![Train Test Split](Results%20Images/Test_And_Train_Split.png)
![Data Visualisation](Results%20Images/Train_Test_Validate_Data_Visualisation.png)

### Training vs Validation Performance
![Training vs Testing](Results%20Images/Testing_Accuracies_And_Loss.png)
![Validation Summary](Results%20Images/Validation_Summary.png)


## 📊 CNN Model Results
![CNN Accuracy](Results%20Images/CNN_Accuracy.png)
![CNN Learning Curve](Results%20Images/CNN_Learnig_Curve.png)
![CNN Confusion Matrix](Results%20Images/CNN_Confusion_Matrix.png)

## 📊 VGG16 Results
![VGG16 Accuracy](Results%20Images/VGG16_Accuracy.png)
![VGG16 Learning Curve](Results%20Images/VGG16_Learnig_Curve.png)
![VGG16 Confusion Matrix](Results%20Images/VGG16_Confusion_Matrix.png)

## 📊 InceptionV3 Results
![InceptionV3 Accuracy](Results%20Images/InceptionV3_Accuracy.png)
![InceptionV3 Learning Curve](Results%20Images/InceptionV3_Learnig_Curve.png)
![InceptionV3 Confusion Matrix](Results%20Images/InceptionV3_Confusion_Matrix.png)

## 🎥 Real-Time & Predictions
![CNN Webcam](Results%20Images/CNN_Webcam_Prediction.png)
![CNN Live Webcam](Results%20Images/CNN_Live_Webcam_Prediction.png)
![Sample Image](Results%20Images/Sample%20Image.png)


## ▶️ How to Run the Project

This section explains how to **reproduce the experiments** and **run real-time hand gesture recognition** from scratch.

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/MuhammadAhmed-38/Hand-Gesture-Recognition.git
cd Hand-Gesture-Recognition

### 2️⃣ Set Up a Python Environment (Recommended)
python3 -m venv venv
source venv/bin/activate   # macOS / Linux

### 3️⃣ Install Required Dependencies
pip install tensorflow opencv-python numpy matplotlib scikit-learn pandas seaborn jupyter

**Note:**
1- TensorFlow automatically selects CPU or GPU depending on availability.
2- GPU acceleration is recommended but not mandatory.

### 4️⃣ Download the Dataset
This repository does NOT include the dataset due to size constraints.
    1- Download the Leap Gesture Recognition Dataset from Kaggle
    2- Extract it locally on your system
After extraction, update the dataset path inside the notebook (Artificial Intelligent.ipynb) to match your local directory.

### 5️⃣ Launch Jupyter Notebook
jupyter notebook

**Then open**

Artificial Intelligent.ipynb

### 6️⃣ Run the Notebook (Important Order)
Run the notebook cells sequentially from top to bottom. The pipeline includes:
    1- Dataset loading and preprocessing
    2- Data balancing and train–validation–test split
    3- Training:
        Custom CNN
        VGG16 (Transfer Learning)
        InceptionV3 (Transfer Learning)
    4- Model evaluation and metric comparison
    5- Visualisation of results and learning curves

### 7️⃣ Run Real-Time Webcam Gesture Recognition
To enable real-time inference:
    1- Ensure your webcam is connected
    2- Run the webcam prediction cell in the notebook
    3- Perform gestures in front of the camera within the defined ROI
The system will:
    1- Segment the hand region
    2- Perform live prediction
    3- Display predicted gesture and confidence score in real time
Press **q** to exit the webcam window.

### 8️⃣ Reproducibility Notes
    1- All models were trained using the same preprocessing pipeline and dataset split
    2- Results are reproducible given the same dataset and environment
    3- Minor accuracy variations may occur due to random initialisation

### 9️⃣ Optional: Experimentation
    1- You can further experiment by:
    2- Adjusting CNN architecture depth
    3- Fine-tuning pretrained layers
    4- Modifying batch size or learning rate
    5- Testing additional gestures or datasets


## 🔭 Current Limitations & Future Work

### Current Limitations
- The dataset is limited to controlled lighting conditions; real-world performance may degrade under extreme lighting or background noise.
- Models were trained on static images; temporal information from video sequences was not exploited.
- Hyperparameter tuning was limited due to computational constraints.
- The real-time webcam prediction module was tested on a limited set of gestures and environments.

### Future Work
- Extend the system to **video-based gesture recognition** using CNN + LSTM or Transformer-based architectures.
- Apply **data augmentation and domain adaptation** to improve robustness in real-world scenarios.
- Experiment with **lightweight models** (e.g., MobileNet, EfficientNet) for edge-device deployment.
- Deploy the model as a **web or mobile application** with real-time inference.
- Integrate explainability techniques (e.g., Grad-CAM) to visualise model attention.

---

## 🎓 Intended Audience

This project is intended for:
- AI / Machine Learning Engineers
- Computer Vision Researchers
- MSc / BSc students studying Artificial Intelligence or Data Science
- Recruiters and technical interviewers evaluating applied deep learning skills

It is especially relevant for those interested in **transfer learning, model comparison, and real-time computer vision systems**.

---

## 📚 Citation & Usage

This repository is provided for **academic, learning, and portfolio demonstration purposes**.

If you use this work in:
- academic assignments
- research projects
- reports or publications  

Please cite or reference this repository appropriately.

Commercial use or redistribution of the dataset must comply with the **original dataset license**.

---

## 👤 Author

**Muhammad Ahmed**  
MSc Artificial Intelligence (Distinction)  
University of East London  

Background in:
- Deep Learning
- Computer Vision
- Natural Language Processing
- Applied AI Systems  

---

## ⭐ Feedback & Contributions

Feedback, suggestions, and improvements are welcome.

If you would like to:
- report an issue
- suggest an enhancement
- contribute code or documentation  

Please open an issue or submit a pull request.

⭐ If you find this project useful, consider starring the repository.
