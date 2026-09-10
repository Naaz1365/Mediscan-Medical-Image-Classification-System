# MediScan — Medical Image Classification System

## 1. Abstract
MediScan is a deep-learning based educational prototype for classifying chest X-ray images into Normal and Pneumonia categories. The system uses ResNet50 transfer learning and Grad-CAM explainability. A Streamlit interface allows a user to upload an image and view the predicted class and confidence together with a heatmap.

## 2. Problem Statement
Manual screening of medical images can be time-consuming. This project demonstrates how computer vision can support automated image classification while also providing an interpretable visualization.

## 3. Objectives
- Build a CNN-based medical image classifier.
- Use transfer learning with ResNet50.
- Evaluate with standard classification metrics.
- Generate Grad-CAM explanations.
- Provide a simple web interface.
- Document limitations and ethical considerations.

## 4. Methodology
Dataset → preprocessing/augmentation → ResNet50 transfer learning → validation → test evaluation → Grad-CAM → Streamlit deployment.

## 5. Dataset
PneumoniaMNIST / MedMNIST. The training script downloads the dataset automatically.

## 6. Technologies
Python, PyTorch, Torchvision, MedMNIST, Scikit-learn, Matplotlib, Seaborn, Grad-CAM, Streamlit and MLflow.

## 7. Results
Run the supplied notebook and paste the generated values from `metrics.json` here:
- Accuracy: ______
- Precision: ______
- Recall: ______
- F1-score: ______
- ROC-AUC: ______

Insert generated figures:
- Accuracy/Loss curves
- Confusion matrix
- ROC curve
- 10–12 Grad-CAM examples

## 8. Web Application
The Streamlit app provides image upload, prediction, confidence score and Grad-CAM visualization.

## 9. Model Comparison
For the required comparison, train/record a second lightweight architecture (e.g., ResNet18) using the same preprocessing and test split, then compare Accuracy, F1 and ROC-AUC. The supplied ResNet50 is the primary model.

## 10. Limitations and Ethics
This prototype is not a medical device. It must not be used for clinical decisions. Dataset bias, distribution shift, image quality and interpretability limitations must be considered.

## 11. Conclusion
MediScan demonstrates an end-to-end deep-learning workflow for medical image classification with explainability and a web interface.
