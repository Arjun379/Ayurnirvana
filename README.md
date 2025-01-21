# Ayurnirvana

# Ayurvedic Foliar Taxonomy via Synergistic Texture-Gradient Descriptors and Cascaded Dimensionality Optimization: Comparative Analysis of Boosted Ensembles, Stochastic Forests, and Hyperplane Classifiers for Multispectral Phytomorphological Patterns
![Untitled design](https://github.com/user-attachments/assets/6552ad1e-2416-4c0a-a6ce-aaa79f744e27)
## Overview
This project aims to classify various plant species using advanced image processing techniques and machine learning models. The methodology involves data augmentation, feature extraction using Histogram of Oriented Gradients (HOG) and Gray-Level Co-occurrence Matrix (GLCM), and classification using Support Vector Machines (SVM) and XGBoost. The results demonstrate high accuracy and robustness in plant species identification.

## Key Highlights
- Utilized OpenCV, Scikit-learn, and XGBoost libraries for image processing, feature extraction, and machine learning.
- Achieved over 90% accuracy in plant species classification using HOG features and SVM model.
- Optimized model performance through hyperparameter tuning with GridSearchCV and RandomizedSearchCV.
- Demonstrated effective data augmentation and feature extraction techniques, enhancing model robustness and accuracy.

## Project Structure
The project is organized into several key steps:
1. **Data Augmentation**: Applied rotation, flipping, and Poisson noise to augment the dataset.
2. **Feature Extraction**: Extracted HOG and GLCM features from the images.
3. **Normalization**: Applied Min-Max normalization to the extracted features.
4. **Dimensionality Reduction**: Used PCA (Principal Component Analysis) and NCA (Neighborhood Component Analysis) to reduce feature dimensionality.
5. **Model Training and Evaluation**: Trained and evaluated SVM and XGBoost models using cross-validation.
6. **Hyperparameter Tuning**: Optimized model hyperparameters using GridSearchCV and RandomizedSearchCV.
7. **Performance Metrics**: Calculated accuracy, precision, recall, and F1 score for model evaluation.

## Results

### SVM with HOG Features
- **Cross-Validation Results**:
  - Mean Accuracy: 93.07%
  - Mean Precision: 0.93
  - Mean Recall: 0.93
  - Mean F1 Score: 0.93
  - Mean Computation Time: 20.3090 seconds

- **Best Hyperparameters**:
  - C: 10
  - Gamma: 0.01
  - Kernel: rbf

- **Training Metrics**:
  - Accuracy: 100.00%
  - Precision: 100.00%
  - Recall: 100.00%
  - F1 Score: 100.00%
  - Computational Time: 1.0347 seconds

- **Testing Metrics**:
  - Accuracy: 90.28%
  - Precision: 90.44%
  - Recall: 90.28%
  - F1 Score: 90.29%
  - Computational Time: 1.0900 seconds

### XGBoost with HOG Features
- **Cross-Validation Results**:
  - Mean Accuracy: 79.62% ± 0.33
  - Mean Precision: 0.80 ± 0.00
  - Mean Recall: 0.80 ± 0.00
  - Mean F1 Score: 0.79 ± 0.00
  - Mean Computation Time: 4.4615 seconds

- **Best Hyperparameters**:
  - Subsample: 0.8
  - N_estimators: 250
  - Min_child_weight: 1
  - Max_depth: 6
  - Learning_rate: 0.0699999999999999
  - Colsample_bytree: 0.8

- **Training Metrics**:
  - Accuracy: 100.00%
  - Precision: 100.00%
  - Recall: 100.00%
  - F1 Score: 100.00%
  - Computational Time: 13.6231 seconds

### XGBoost with GLCM Features
- **Cross-Validation Results**:
  - Mean Accuracy: 74.24% ± 1.08
  - Mean Precision: 0.76 ± 0.01
  - Mean Recall: 0.74 ± 0.01
  - Mean F1 Score: 0.74 ± 0.01
  - Mean Computation Time: 0.0024 seconds

- **Best Hyperparameters**:
  - Colsample_bytree: 0.8060664215236861
  - Learning_rate: 0.15771142658314577
  - Max_depth: 4
  - Min_child_weight: 1
  - N_estimators: 207
  - Subsample: 0.9829269087326896

- **Training Metrics**:
  - Accuracy: 100.00%
  - Precision: 1.00
  - Recall: 1.00
  - F1 Score: 1.00
  - Computational Time: 7.5347 seconds

## Visualizations
- **Confusion Matrix**: Visualized the performance of the models using confusion matrices.
- **Learning Curves**: Plotted learning curves to analyze model performance over different training sizes.
- **ROC and Precision-Recall Curves**: Generated ROC and Precision-Recall curves for model evaluation.

## Conclusion
The project successfully demonstrates the application of advanced image processing and machine learning techniques for plant species classification. The results highlight the effectiveness of HOG and GLCM features, as well as the robustness of SVM and XGBoost models in achieving high classification accuracy.

## Future Work
- Explore additional feature extraction techniques such as deep learning-based features.
- Experiment with other machine learning models and ensemble methods.
- Collect more data to improve model generalization and performance.

