# MediScan — Model Card

## Model
ResNet50 transfer-learning classifier for binary chest X-ray classification.

## Dataset
PneumoniaMNIST from the MedMNIST collection. Classes: Normal and Pneumonia.

## Intended Use
Educational/research demonstration of image classification and explainability.

## Not Intended For
Clinical diagnosis, treatment decisions, patient triage, or use without qualified medical professionals.

## Input
Chest X-ray image resized to 224×224.

## Explainability
Grad-CAM is used to visualize image regions that contribute to the model output.

## Evaluation
Run the training notebook to generate actual Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix and ROC curve. Do not replace these with invented values.

## Limitations
- Dataset distribution may differ from real hospital populations.
- Model can inherit dataset and labeling bias.
- Image quality and acquisition differences can affect predictions.
- Grad-CAM is an explanation aid, not proof of clinical reasoning.
