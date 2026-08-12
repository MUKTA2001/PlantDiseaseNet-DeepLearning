# PlantDiseaseNet-DeepLearning

A deep learning framework for **plant disease image classification** using the PlantVillage dataset. The project explores and compares multiple deep learning architectures, including a custom CNN, ConvNeXt-Tiny, EfficientNet-B4, and Swin-Tiny, followed by ensemble-based prediction using soft-voting fusion.

## 📌 Overview

Plant diseases can significantly affect agricultural productivity and crop quality. Automated image-based disease classification can assist in identifying plant diseases efficiently from leaf images.

This project develops and evaluates multiple deep learning models for classifying diseases affecting **Tomato, Potato, and Pepper** plants.

The workflow includes:

* Dataset exploration and exploratory data analysis (EDA)
* Image preprocessing and augmentation
* Duplicate/leakage detection
* Training multiple deep learning architectures
* Model performance evaluation
* Ensemble prediction using soft voting
* ROC-AUC analysis
* Confusion matrix analysis
* Cross-validation
* Statistical model comparison

## 🌱 Dataset

The project uses the **PlantVillage dataset** and focuses on three crops:

* 🍅 Tomato
* 🥔 Potato
* 🌶️ Pepper

The selected subset contains:

* **15 disease/health classes**
* **20,638 images**

The classes include healthy plants as well as several disease categories affecting Tomato, Potato, and Pepper.

## 🧠 Models

The project evaluates four different architectures:

### 1. Basic CNN

A custom convolutional neural network is implemented as a baseline model for comparison with the more advanced architectures.

### 2. ConvNeXt-Tiny

A modern convolutional neural network architecture designed to provide strong image classification performance while retaining the advantages of convolution-based models.

### 3. EfficientNet-B4

EfficientNet-B4 is evaluated as a higher-capacity transfer-learning model using a larger input resolution of **380 × 380 pixels**.

### 4. Swin-Tiny

Swin-Tiny is included as a Transformer-based vision architecture, allowing comparison between convolutional and Transformer-based approaches.

## 🔬 Ensemble Learning

To improve the robustness of predictions, the project combines predictions from multiple models using **soft-voting fusion**.

The ensemble combines model probability outputs rather than relying on a single classifier.

The project also evaluates the agreement and differences between individual models using statistical analysis.

## 🔄 Workflow

```text
PlantVillage Dataset
        │
        ▼
Dataset Exploration
        │
        ▼
EDA & Class Distribution Analysis
        │
        ▼
Image Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
Duplicate / Data Leakage Detection
        │
        ▼
Train / Validation / Test Split
        │
        ├───────────────┐
        ▼               ▼
   Basic CNN       ConvNeXt-Tiny
        │               │
        ├───────────────┤
        ▼               ▼
 EfficientNet-B4     Swin-Tiny
        │               │
        └───────┬───────┘
                ▼
        Model Evaluation
                │
                ▼
       Soft-Voting Fusion
                │
                ▼
      Statistical Evaluation
                │
                ▼
        Final Analysis
```

## 🛠️ Technologies Used

* **Python 3**
* **PyTorch**
* **Torchvision**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **SciPy**
* **PIL**
* **ImageHash**
* **Statsmodels**

## ⚙️ Hardware & Environment

The notebook is designed to run in a GPU-enabled environment.

During the recorded experiment, the notebook used:

```text
Device: CUDA
GPU: Tesla T4
```

The experiment also uses a fixed random seed:

```python
SEED = 42
```

to improve reproducibility.

## 📊 Evaluation Metrics

The models are evaluated using several performance measures, including:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix
* ROC curves
* AUC
* Cross-validation
* McNemar's statistical test

These evaluations provide both overall and class-level insight into model performance.

## 🔍 Data Preprocessing

The project performs several preprocessing steps before model training.

### Image Resizing

Images are resized according to the requirements of the selected architecture.

For example:

```text
224 × 224
```

is used for the standard models, while EfficientNet-B4 uses:

```text
380 × 380
```

### Data Augmentation

Training images undergo augmentation techniques such as:

* Random horizontal flipping
* Random rotation
* Color-based transformations
* Image normalization

This helps improve model generalization and reduce overfitting.

## 🧹 Data Leakage Detection

The project also investigates possible duplicate images between dataset splits using **perceptual image hashing**.

This step helps identify potentially duplicated or highly similar images that could otherwise lead to overly optimistic evaluation results.

## 📈 Visualization

The project generates several visualizations for analysis, including:

* Class distribution
* Crop-level dataset distribution
* Image characteristics
* Confusion matrices
* ROC curves
* Model performance comparisons

High-resolution figures are also generated for academic/research reporting.

## 🔬 Cross-Validation

A **5-fold Stratified Cross-Validation** procedure is used to further evaluate model behavior across different subsets of the data.

Stratification ensures that class distributions are considered during the fold construction.

## 📐 Statistical Analysis

The project includes **McNemar's test** to statistically compare the predictions of different models.

This provides an additional perspective beyond simply comparing accuracy values.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/PlantDiseaseNet-DeepLearning.git
cd PlantDiseaseNet-DeepLearning
```

### 2. Install the required packages

```bash
pip install torch torchvision numpy pandas matplotlib seaborn pillow scipy scikit-learn imagehash statsmodels
```

### 3. Download the Dataset

Download the PlantVillage dataset and update the dataset path in the notebook:

```python
DATASET_PATH = "/path/to/PlantVillage"
```

### 4. Open the Notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Then open:

```text
ml-project-tp.ipynb
```

### 5. Run the Notebook

Execute the cells sequentially to reproduce:

1. Dataset exploration
2. EDA
3. Visualization
4. Preprocessing
5. Data augmentation
6. Leakage detection
7. Model training
8. Model evaluation
9. Ensemble fusion
10. Statistical analysis

## 📁 Suggested Repository Structure

```text
PlantDiseaseNet-DeepLearning/
│
├── README.md
├── notebooks/
│   └── plant_disease_classification.ipynb
│
├── results/
│   ├── figures/
│   ├── confusion_matrices/
│   └── roc_curves/
│
├── models/
│   └── README.md
│
├── requirements.txt
│
└── .gitignore
```

## 🎯 Project Objectives

The main objectives are to:

* Develop an automated plant disease classification system.
* Analyze disease patterns across Tomato, Potato, and Pepper crops.
* Compare traditional CNNs with modern architectures.
* Evaluate ConvNeXt, EfficientNet, and Swin Transformer models.
* Investigate the effectiveness of ensemble learning.
* Reduce the risk of data leakage during evaluation.
* Perform rigorous statistical model comparisons.
* Build a reproducible deep learning experimentation pipeline.

## 🔮 Future Work

Potential improvements include:

* Testing additional agricultural datasets.
* Incorporating field-condition images instead of laboratory-style images.
* Evaluating additional Transformer architectures.
* Applying explainable AI techniques such as Grad-CAM.
* Developing a lightweight model for mobile deployment.
* Building a web or mobile application for real-time disease prediction.
* Investigating domain adaptation for real-world agricultural environments.

## 📜 License

This project is intended for **academic, educational, and research purposes**.

Add an appropriate open-source license if you plan to distribute the source code publicly.

## 👩‍💻 Author
**Trishita Paul**
**Meharunnasa Mukta**

Computer Science and Engineering
International Islamic University Chittagong (IIUC)
