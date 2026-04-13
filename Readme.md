# 🧠 Fatty Liver Severity Classification using VGG19

## 📌 Project Overview

This project focuses on **multi-class classification of fatty liver severity** using ultrasound images. The goal is to replicate and compare results from a research paper using deep learning models.

We implemented a **VGG19-based Convolutional Neural Network (CNN)** and compared it with a reference paper that uses **VGG16**.

---

## 📄 Reference Paper

**Title:**  
Convolutional Neural Network Classification of Ultrasound Parametric Images for Liver Steatosis Grading  

**Link:**  
https://link.springer.com/article/10.1007/s11517-024-03187-4  

---

## 🧠 Methodology

| Component | Details |
|----------|--------|
| Model (Paper) | VGG-16 (Pretrained CNN with Transfer Learning) |
| Model (Trained on our dataset) | VGG-19 (Pretrained CNN with Transfer Learning) |
| Task | Multi-class Fatty Liver Severity Classification |
| Input | Ultrasound Images |
| Dataset | https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset?resource=download |

---

## 📂 Dataset

- Ultrasound images categorized into:
  - **Benign**
  - **Malignant**
  - **Normal**

- Dataset split:
  - 80% Training
  - 20% Testing

---

## ⚙️ Model Details

### 🔹 Paper Approach
- Uses **VGG16**
- Input: Parametric images (moment maps)
- ROI-based classification
- Hard voting for final prediction

---

### 🔹 Our Approach(Trained same model on our dataset for comparitive analysis)
- Uses **VGG19**
- Input: Raw ultrasound images
- Transfer learning applied
- End-to-end classification

---

## 📊 Results

### 🔥 Overall Performance

| Metric | Paper (VGG16) | Trained on our dataset (VGG19) |
|------|--------------|----------------|
| Accuracy | ~63% | **67.35%** |
| Precision | Moderate | 0.67 (weighted avg) |
| Recall (Sensitivity) | ~60–63% | 0.67 (weighted avg) |
| F1-score | ~0.60 | **0.61** |
| Specificity | Not reported | **0.7602** |
| AUC Score | Not reported | **0.8821** |

---

## 📊 Class-wise Performance

| Class    | Precision | Recall | F1-score |
|------    |---------- |--------|--------- |
| Class 0  |   0.45    | 0.23   |  0.30    |
| Class 1  | **0.70**  | **0.99** | **0.82** |
| Class 2  | **1.00**  | 0.20     | 0.33 |

---

## 📉 Confusion Matrix
[[ 9 31 0]
[ 1 86 0]
[10 6 4]]


---

## 🧠 Analysis & Insights

### 🔥 1. Accuracy Improvement
- Our model achieves **67.35% accuracy**, outperforming the paper (~63%)
- Reasons:
  - Larger dataset
  - Deeper architecture (VGG19 vs VGG16)

---

### ⚠️ 2. Class Imbalance Issue
- Model heavily biased toward **Class 1**
- Very high recall for Class 1 (0.99)
- Poor recall for:
  - Class 0 (0.23)
  - Class 2 (0.20)

---

### 📉 3. Minority Class Problem
- Severe cases (Class 2) poorly detected
- High precision but low recall
- Indicates model rarely predicts this class

---

### 🔥 4. Strong Feature Learning
- High AUC score (**0.8821**)
- Indicates strong class separability
- Model effectively learns features from ultrasound images

---

### ⚖️ 5. Paper vs Our Model

| Aspect      | Paper         | Our dataset result |
|------       |------         |--------|
| Balance     | More balanced | Imbalanced |
| Accuracy    | Lower         | Higher |
| Input Type | Parametric images | Raw images |
| Generalization| Controlled | Data-driven |

---

## 🧠 Key Insight

- Paper focuses on **balanced classification using feature engineering**
- Our model focuses on **higher accuracy using deep learning directly**

---

## 🎯 Conclusion

The VGG19-based model achieved higher accuracy (**67.35%**) compared to the VGG16-based approach (~63%) reported in the reference paper. This improvement is attributed to the use of a larger dataset and deeper network architecture.

However, the model exhibits **class imbalance**, leading to poor recall for minority classes. While the model demonstrates strong feature learning capability (**AUC = 0.88**), further improvements are required to enhance performance across all severity levels.

---

## 🚀 Future Work

- Apply **class balancing techniques**
- Use **ResNet / DenseNet architectures**
- Improve detection of minority classes
- Explore **ROI-based learning (like paper)**

---

## 👨‍💻 Author

- Project developed as part of research on **Fatty Liver Severity Detection using Deep Learning**

---
