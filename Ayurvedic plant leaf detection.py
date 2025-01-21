#!/usr/bin/env python
# coding: utf-8

# # DATA AUGMENTATION

# In[1]:


import os
import cv2
import numpy as np
from scipy.stats import poisson
from tqdm import tqdm
import cv2
import os
import shutil
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import exposure
import pandas as pd
from scipy.ndimage import gaussian_filter
from PIL import Image
import seaborn as sns
import matplotlib as pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import time
import joblib
import time
from skimage.feature import hog
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import label_binarize
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns


# In[2]:


# Function to apply rotation, flipping, and Poisson noise to an image
def augment_image(image):
    # Rotation
    angle = np.random.uniform(-30, 30)
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    # Horizontal flipping
    if np.random.choice([True, False]):
        rotated_image = cv2.flip(rotated_image, 1)

    # Poisson noise
    noisy_image = poisson.rvs(rotated_image, random_state=None)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

# Function to perform data augmentation for a leaf class
def augment_leaf_class(class_folder, output_folder, augmentation_factor=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(class_folder)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_folder, filename)
            original_image = cv2.imread(img_path)

            # Augment and save images
            for i in range(augmentation_factor):
                augmented_image = augment_image(original_image)
                output_filename = f"{os.path.splitext(filename)[0]}_aug_{i+1}.png"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, augmented_image)

# Main function to perform data augmentation for all leaf classes
def main():
    dataset_folder = 'dataset'
    augmented_dataset_folder = 'augmented_dataset'
    augmentation_factor = 5

    if not os.path.exists(augmented_dataset_folder):
        os.makedirs(augmented_dataset_folder)

    for class_folder in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_folder)
        if os.path.isdir(class_path):
            output_class_folder = os.path.join(augmented_dataset_folder, class_folder)
            augment_leaf_class(class_path, output_class_folder, augmentation_factor)

if __name__ == "__main__":
    main()


# In[ ]:





# # SPLITTING OF DATASET 

# In[2]:


import os
import shutil
from sklearn.model_selection import train_test_split

# Step 1: Split the dataset into training and testing sets
dataset_path = 'augmented_dataset'
split_dataset_path = 'splitted_dataset'

# Create the 'splitted_dataset' folder if not exists
if not os.path.exists(split_dataset_path):
    os.makedirs(split_dataset_path)

for plant_folder in os.listdir(dataset_path):
    plant_path = os.path.join(dataset_path, plant_folder)
    split_plant_path = os.path.join(split_dataset_path, plant_folder)

    # Create 'train_data' and 'test_data' folders
    train_folder = os.path.join(split_plant_path, 'train_data')
    test_folder = os.path.join(split_plant_path, 'test_data')

    if not os.path.exists(split_plant_path):
        os.makedirs(split_plant_path)

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Split the images into training and testing sets
    images = os.listdir(plant_path)
    train_images, test_images = train_test_split(images, test_size=0.3, random_state=42)

    # Copy images to the respective folders
    for image in train_images:
        shutil.copy(os.path.join(plant_path, image), os.path.join(train_folder, image))

    for image in test_images:
        shutil.copy(os.path.join(plant_path, image), os.path.join(test_folder, image))

print("Step 1: Dataset split completed successfully.")


# In[ ]:





# # GRAYSACALE APPLICATION

# In[3]:


import cv2
import os

# Step 2: Apply grayscale to images in 'train_data' and 'test_data' folders
splitted_dataset_path = 'splitted_dataset'
grayscaled_dataset_path = 'Grayscaled_dataset'

# Create 'Grayscaled_dataset' folder if not exists
if not os.path.exists(grayscaled_dataset_path):
    os.makedirs(grayscaled_dataset_path)

for plant_folder in os.listdir(splitted_dataset_path):
    plant_path = os.path.join(splitted_dataset_path, plant_folder)
    grayscaled_plant_path = os.path.join(grayscaled_dataset_path, plant_folder)

    # Create 'gray_train_data' and 'gray_test_data' folders
    gray_train_folder = os.path.join(grayscaled_plant_path, 'gray_train_data')
    gray_test_folder = os.path.join(grayscaled_plant_path, 'gray_test_data')

    if not os.path.exists(grayscaled_plant_path):
        os.makedirs(grayscaled_plant_path)

    if not os.path.exists(gray_train_folder):
        os.makedirs(gray_train_folder)

    if not os.path.exists(gray_test_folder):
        os.makedirs(gray_test_folder)

    # Apply grayscale to images in 'train_data' folder
    for image_name in os.listdir(os.path.join(plant_path, 'train_data')):
        image_path = os.path.join(plant_path, 'train_data', image_name)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(gray_train_folder, image_name), gray_img)

    # Apply grayscale to images in 'test_data' folder
    for image_name in os.listdir(os.path.join(plant_path, 'test_data')):
        image_path = os.path.join(plant_path, 'test_data', image_name)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(gray_test_folder, image_name), gray_img)

print("Step 2: Grayscale applied to images successfully.")


# # GAUSSIAN FILTER APPLICATION

# In[4]:


import cv2
import os

# Step 3: Apply Gaussian filter to images in 'gray_train_data' and 'gray_test_data' folders
grayscaled_dataset_path = 'Grayscaled_dataset'
gaussian_dataset_path = 'Gaussian_dataset'

# Create 'Gaussian_dataset' folder if not exists
if not os.path.exists(gaussian_dataset_path):
    os.makedirs(gaussian_dataset_path)

for plant_folder in os.listdir(grayscaled_dataset_path):
    grayscaled_plant_path = os.path.join(grayscaled_dataset_path, plant_folder)
    gaussian_plant_path = os.path.join(gaussian_dataset_path, plant_folder)

    # Create 'gauss_train_data' and 'gauss_test_data' folders
    gauss_train_folder = os.path.join(gaussian_plant_path, 'gauss_train_data')
    gauss_test_folder = os.path.join(gaussian_plant_path, 'gauss_test_data')

    if not os.path.exists(gaussian_plant_path):
        os.makedirs(gaussian_plant_path)

    if not os.path.exists(gauss_train_folder):
        os.makedirs(gauss_train_folder)

    if not os.path.exists(gauss_test_folder):
        os.makedirs(gauss_test_folder)

    # Apply Gaussian filter to images in 'gray_train_data' folder
    for image_name in os.listdir(os.path.join(grayscaled_plant_path, 'gray_train_data')):
        image_path = os.path.join(grayscaled_plant_path, 'gray_train_data', image_name)
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gauss_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        cv2.imwrite(os.path.join(gauss_train_folder, image_name), gauss_img)

    # Apply Gaussian filter to images in 'gray_test_data' folder
    for image_name in os.listdir(os.path.join(grayscaled_plant_path, 'gray_test_data')):
        image_path = os.path.join(grayscaled_plant_path, 'gray_test_data', image_name)
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gauss_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        cv2.imwrite(os.path.join(gauss_test_folder, image_name), gauss_img)

print("Step 3: Gaussian filter applied to images successfully.")


# In[ ]:





# # RESIZED IMAGE

# In[5]:


import cv2
import os

# Step 4: Resize images in 'gauss_train_data' and 'gauss_test_data' folders
gaussian_dataset_path = 'Gaussian_dataset'
final_splitted_data_path = 'final_splitted_data'
target_size = (128, 128)

# Create 'final_splitted_data' folder if not exists
if not os.path.exists(final_splitted_data_path):
    os.makedirs(final_splitted_data_path)

for plant_folder in os.listdir(gaussian_dataset_path):
    gauss_plant_path = os.path.join(gaussian_dataset_path, plant_folder)
    final_plant_path = os.path.join(final_splitted_data_path, plant_folder)

    # Create 'res_train_data' and 'res_test_data' folders
    res_train_folder = os.path.join(final_plant_path, 'res_train_data')
    res_test_folder = os.path.join(final_plant_path, 'res_test_data')

    if not os.path.exists(final_plant_path):
        os.makedirs(final_plant_path)

    if not os.path.exists(res_train_folder):
        os.makedirs(res_train_folder)

    if not os.path.exists(res_test_folder):
        os.makedirs(res_test_folder)

    # Resize images in 'gauss_train_data' folder
    for image_name in os.listdir(os.path.join(gauss_plant_path, 'gauss_train_data')):
        image_path = os.path.join(gauss_plant_path, 'gauss_train_data', image_name)
        gauss_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(gauss_img, target_size)
        cv2.imwrite(os.path.join(res_train_folder, image_name), resized_img)

    # Resize images in 'gauss_test_data' folder
    for image_name in os.listdir(os.path.join(gauss_plant_path, 'gauss_test_data')):
        image_path = os.path.join(gauss_plant_path, 'gauss_test_data', image_name)
        gauss_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(gauss_img, target_size)
        cv2.imwrite(os.path.join(res_test_folder, image_name), resized_img)

print("Step 4: Images resized successfully.")


# In[ ]:





# # HOG FEATURE EXTRACTION

# In[6]:


import cv2
import os
import pandas as pd

from skimage.feature import hog
from sklearn.preprocessing import MinMaxScaler

# Step 5: Extract HOG features and save to CSV
final_splitted_data_path = 'final_splitted_data'
hog_train_csv_path = 'hog_train_features.csv'
hog_test_csv_path = 'hog_test_features.csv'

# Lists to store HOG features and labels
hog_train_features = []
hog_train_labels = []
hog_test_features = []
hog_test_labels = []

# Initialize HOG descriptor with specified parameters (cell_size and bins)
cell_size = (16, 16)
bins = 180
hog = cv2.HOGDescriptor(_winSize=(128, 128), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=cell_size, _nbins=bins)

# Extract HOG features for 'res_train_data' folder
for plant_folder in os.listdir(final_splitted_data_path):
    res_train_folder = os.path.join(final_splitted_data_path, plant_folder, 'res_train_data')

    for image_name in os.listdir(res_train_folder):
        image_path = os.path.join(res_train_folder, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        hog_features = hog.compute(img)
        hog_train_features.append(hog_features)
        hog_train_labels.append(plant_folder)

# Extract HOG features for 'res_test_data' folder
for plant_folder in os.listdir(final_splitted_data_path):
    res_test_folder = os.path.join(final_splitted_data_path, plant_folder, 'res_test_data')

    for image_name in os.listdir(res_test_folder):
        image_path = os.path.join(res_test_folder, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        hog_features = hog.compute(img)
        hog_test_features.append(hog_features)
        hog_test_labels.append(plant_folder)

# Convert lists to pandas DataFrames
hog_train_df = pd.DataFrame(hog_train_features)
hog_train_df['label'] = hog_train_labels

hog_test_df = pd.DataFrame(hog_test_features)
hog_test_df['label'] = hog_test_labels

# Save HOG features to CSV
hog_train_df.to_csv(hog_train_csv_path, index=False)
hog_test_df.to_csv(hog_test_csv_path, index=False)

print("Step 5: HOG features extracted and saved to CSV successfully.")


# In[ ]:





# # MIN-MAX NORMALIZATION OF HOG FEATURES

# In[7]:


from sklearn.preprocessing import MinMaxScaler

# Step 6: Minmax normalization of HOG features
normalized_hog_train_csv_path = 'norm_hog_train_features.csv'
normalized_hog_test_csv_path = 'norm_hog_test_features.csv'

# Load HOG features from CSV
hog_train_df = pd.read_csv(hog_train_csv_path)
hog_test_df = pd.read_csv(hog_test_csv_path)

# Extract labels
train_labels = hog_train_df['label']
test_labels = hog_test_df['label']

# Extract HOG features
train_features = hog_train_df.drop(columns=['label'])
test_features = hog_test_df.drop(columns=['label'])

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform on training data
normalized_train_features = scaler.fit_transform(train_features)
normalized_hog_train_df = pd.DataFrame(normalized_train_features)
normalized_hog_train_df['label'] = train_labels

# Transform test data
normalized_test_features = scaler.transform(test_features)
normalized_hog_test_df = pd.DataFrame(normalized_test_features)
normalized_hog_test_df['label'] = test_labels

# Save normalized HOG features to CSV
normalized_hog_train_df.to_csv(normalized_hog_train_csv_path, index=False)
normalized_hog_test_df.to_csv(normalized_hog_test_csv_path, index=False)

print("Step 6: Minmax normalization of HOG features completed successfully.")


# In[ ]:





# # PCA & NCA ON NORMALIZED HOG FEATURES 

# In[8]:


from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis


# Step 7: PCA and NCA on normalized HOG features
pca_hog_train_csv_path = 'pca_hog_train_features.csv'
nca_hog_train_csv_path = 'NCA_hog_train_features.csv'
pca_nca_hog_test_csv_path = 'NCA_hog_test_features.csv'

# Load normalized HOG features from CSV
normalized_hog_train_df = pd.read_csv(normalized_hog_train_csv_path)
normalized_hog_test_df = pd.read_csv(normalized_hog_test_csv_path)

# Extract labels
train_labels = normalized_hog_train_df['label']
test_labels = normalized_hog_test_df['label']

# Extract normalized HOG features
train_features = normalized_hog_train_df.drop(columns=['label'])
test_features = normalized_hog_test_df.drop(columns=['label'])

# Step 7.1: PCA on normalized HOG features
pca = PCA(n_components=50)  # Choose an appropriate number of components
pca_train_features = pca.fit_transform(train_features)
pca_hog_train_df = pd.DataFrame(pca_train_features)
pca_hog_train_df['label'] = train_labels

# Save PCA features to CSV
pca_hog_train_df.to_csv(pca_hog_train_csv_path, index=False)

# Step 7.2: NCA on normalized HOG features
nca = NeighborhoodComponentsAnalysis()
nca_train_features = nca.fit_transform(pca_train_features, train_labels)
nca_hog_train_df = pd.DataFrame(nca_train_features)
nca_hog_train_df['label'] = train_labels

# Save NCA features to CSV
nca_hog_train_df.to_csv(nca_hog_train_csv_path, index=False)

# Apply PCA and NCA on normalized HOG features for test set
pca_test_features = pca.transform(test_features)
pca_nca_test_features = nca.transform(pca_test_features)

# Save NCA features for test set to CSV
pca_nca_hog_test_df = pd.DataFrame(pca_nca_test_features)
pca_nca_hog_test_df['label'] = test_labels
pca_nca_hog_test_df.to_csv(pca_nca_hog_test_csv_path, index=False)

print("Step 7: PCA and NCA on normalized HOG features completed successfully.")


# In[ ]:





# # Calculation of Accuracy , precision , recall & F1 Score(On training Dataset)

# In[9]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Step 9: Cross-validation with SVM and StratifiedKFold
n_splits = 5
svm_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'computation_time': []}

# Load NCA features from CSV
nca_hog_train_df = pd.read_csv(nca_hog_train_csv_path)

# Extract labels
labels = nca_hog_train_df['label']
features = nca_hog_train_df.drop(columns=['label'])

# Initialize SVM classifier
svm_classifier = SVC()

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(stratified_kfold.split(features, labels), 1):
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

    start_time = time.time()

    # Train SVM classifier
    svm_classifier.fit(X_train, y_train)

    # Predict on the training set
    predictions = svm_classifier.predict(X_train)

    # Calculate metrics
    accuracy = accuracy_score(y_train, predictions) * 100
    precision = precision_score(y_train, predictions, average='weighted')
    recall = recall_score(y_train, predictions, average='weighted')
    f1 = f1_score(y_train, predictions, average='weighted')
    computation_time = time.time() - start_time

    # Store metrics for each fold
    svm_metrics['accuracy'].append(accuracy)
    svm_metrics['precision'].append(precision)
    svm_metrics['recall'].append(recall)
    svm_metrics['f1'].append(f1)
    svm_metrics['computation_time'].append(computation_time)

    # Print metrics for each fold
    print(f"\nFold {fold}:")
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Computation Time: {computation_time:.4f} seconds')

# Calculate mean metrics
mean_accuracy = sum(svm_metrics['accuracy']) / n_splits
mean_precision = sum(svm_metrics['precision']) / n_splits
mean_recall = sum(svm_metrics['recall']) / n_splits
mean_f1 = sum(svm_metrics['f1']) / n_splits
mean_computation_time = sum(svm_metrics['computation_time']) / n_splits

# Print mean metrics
print("\nMean Metrics:")
print(f'Mean Accuracy: {mean_accuracy:.2f}%')
print(f'Mean Precision: {mean_precision:.2f}')
print(f'Mean Recall: {mean_recall:.2f}')
print(f'Mean F1 Score: {mean_f1:.2f}')
print(f'Mean Computation Time: {mean_computation_time:.4f} seconds')


# In[ ]:





# # Standard Deviation

# In[10]:


import numpy as np

# Step 10: Calculate mean and standard deviation of metrics
std_accuracy = np.std(svm_metrics['accuracy'])
std_precision = np.std(svm_metrics['precision'])
std_recall = np.std(svm_metrics['recall'])
std_f1 = np.std(svm_metrics['f1'])

# Print mean and standard deviation of metrics
print("\nMetrics Statistics:")
print(f'Standard Deviation of Accuracy: {std_accuracy:.2f}')
print(f'Standard Deviation of Precision: {std_precision:.2f}')
print(f'Standard Deviation of Recall: {std_recall:.2f}')
print(f'Standard Deviation of F1 Score: {std_f1:.2f}')


# In[ ]:





# # Computational Time per fold 

# In[11]:


import matplotlib.pyplot as plt

# Step 11: Calculate and plot time metrics
fold_numbers = range(1, n_splits + 1)

# Plot computational time for each fold
plt.figure(figsize=(10, 6))
plt.plot(fold_numbers, svm_metrics['computation_time'], marker='o', linestyle='-', color='b')
plt.title('Computational Time per Fold')
plt.xlabel('Fold')
plt.ylabel('Computation Time (seconds)')
plt.grid(True)
plt.show()


# In[ ]:





# # Metric Distribution Across Folds

# In[12]:


import seaborn as sns

# Step 12: Create Box Plot for Metric Distribution
metric_data = {
    'Accuracy': svm_metrics['accuracy'],
    'Precision': svm_metrics['precision'],
    'Recall': svm_metrics['recall'],
    'F1 Score': svm_metrics['f1']
}

metric_df = pd.DataFrame(metric_data)

# Create a box plot
plt.figure(figsize=(12, 8))
sns.boxplot(data=metric_df)
plt.title('Metric Distribution Across Folds')
plt.ylabel('Metric Value')
plt.show()


# In[ ]:





# # Extraction Of Best Hyperparameters

# In[13]:


from sklearn.model_selection import GridSearchCV

# Step 14: Hyperparameter Tuning with Grid Search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}

# Load NCA features from CSV
nca_hog_train_df = pd.read_csv(nca_hog_train_csv_path)

# Extract labels
labels = nca_hog_train_df['label']
features = nca_hog_train_df.drop(columns=['label'])

# Initialize SVM classifier
svm_classifier = SVC()

# Initialize GridSearchCV
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, verbose=2, n_jobs=-1)

# Perform grid search
grid_search.fit(features, labels)

# Print best hyperparameters
print("\nBest Hyperparameters:")
print(grid_search.best_params_)


# In[ ]:





# # SVM model trained with best hyperparameters

# In[14]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Best hyperparameters obtained from GridSearchCV
best_params = {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}

# Load NCA features from CSV
nca_hog_train_df = pd.read_csv(nca_hog_train_csv_path)

# Extract labels
train_labels = nca_hog_train_df['label']
train_features = nca_hog_train_df.drop(columns=['label'])

# Initialize SVM classifier with best hyperparameters
svm_classifier = SVC(**best_params)

# Train the model
start_time = time.time()
svm_classifier.fit(train_features, train_labels)
training_time = time.time() - start_time

# Predict on the training set
train_predictions = svm_classifier.predict(train_features)

# Calculate metrics
accuracy = accuracy_score(train_labels, train_predictions) * 100
precision = precision_score(train_labels, train_predictions, average='weighted') * 100
recall = recall_score(train_labels, train_predictions, average='weighted') * 100
f1 = f1_score(train_labels, train_predictions, average='weighted') * 100

print("Training Metrics:")
print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}%')
print(f'Recall: {recall:.2f}%')
print(f'F1 Score: {f1:.2f}%')
print(f'Computational Time: {training_time:.4f} seconds')


# In[ ]:





# In[15]:


from joblib import dump

# Save the trained model
model_filename = 'svm_model.joblib'
dump(svm_classifier, model_filename)

print(f"Trained model saved as {model_filename}")


# In[ ]:





# # Confusion Matrix with Classification Report(SVM + HOG )

# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Load the test dataset
nca_hog_test_df = pd.read_csv(pca_nca_hog_test_csv_path)

# Extract labels
test_labels = nca_hog_test_df['label']
test_features = nca_hog_test_df.drop(columns=['label'])

# Predict on the test set
test_predictions = svm_classifier.predict(test_features)

# Create confusion matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)

# Get unique labels
unique_labels = sorted(nca_hog_train_df['label'].unique())

# Create a DataFrame for the confusion matrix with serial numbers
conf_matrix_df = pd.DataFrame(conf_matrix, index=range(1, len(unique_labels) + 1), columns=range(1, len(unique_labels) + 1))

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Print each plant name according to the serial number
print("Plant Names:")
for i, plant_name in enumerate(unique_labels, 1):
    print(f"{i}. {plant_name}")


# In[ ]:





# # Calculate TP, TN, FP, FN for each class

# In[17]:


# Initialize dictionaries to store TP, TN, FP, FN for each class
class_metrics = {}

# Get unique labels
unique_labels = sorted(nca_hog_train_df['label'].unique())

# Iterate through each class
for label in unique_labels:
    # Initialize counters
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    # Iterate through test labels and predictions
    for true_label, pred_label in zip(test_labels, test_predictions):
        if true_label == label and pred_label == label:
            tp += 1
        elif true_label != label and pred_label != label:
            tn += 1
        elif true_label != label and pred_label == label:
            fp += 1
        elif true_label == label and pred_label != label:
            fn += 1
            
    # Store metrics for the class
    class_metrics[label] = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

# Print TP, TN, FP, FN for each class
for label, metrics in class_metrics.items():
    print(f"\nClass: {label}")
    print(f"True Positives (TP): {metrics['TP']}")
    print(f"True Negatives (TN): {metrics['TN']}")
    print(f"False Positives (FP): {metrics['FP']}")
    print(f"False Negatives (FN): {metrics['FN']}")


# In[ ]:





# In[18]:


import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Start timing
start_time = time.time()

# Predict on the test set
test_predictions = svm_classifier.predict(test_features)

# Calculate metrics
accuracy = accuracy_score(test_labels, test_predictions) * 100
precision = precision_score(test_labels, test_predictions, average='weighted') * 100
recall = recall_score(test_labels, test_predictions, average='weighted') * 100
f1 = f1_score(test_labels, test_predictions, average='weighted') * 100

# Calculate computational time
testing_time = time.time() - start_time

# Print metrics
print("Metrics on Testing Dataset:")
print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}%')
print(f'Recall: {recall:.2f}%')
print(f'F1 Score: {f1:.2f}%')
print(f'Computational Time: {testing_time:.4f} seconds')


# In[ ]:





# In[19]:


import matplotlib.pyplot as plt

# Get unique labels
unique_labels = sorted(nca_hog_train_df['label'].unique())

# Initialize lists to store metrics for each class
tp_values = []
tn_values = []
fp_values = []
fn_values = []

# Iterate through each class
for label in unique_labels:
    # Get metrics for the class
    metrics = class_metrics[label]
    
    # Append TP, TN, FP, FN values to respective lists
    tp_values.append(metrics['TP'])
    tn_values.append(metrics['TN'])
    fp_values.append(metrics['FP'])
    fn_values.append(metrics['FN'])

# Plot bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.2
index = range(1, len(unique_labels) + 1)

plt.bar(index, tp_values, bar_width, label='True Positives')
plt.bar([i + bar_width for i in index], tn_values, bar_width, label='True Negatives')
plt.bar([i + 2*bar_width for i in index], fp_values, bar_width, label='False Positives')
plt.bar([i + 3*bar_width for i in index], fn_values, bar_width, label='False Negatives')

plt.xlabel('Serial Numbers of Plants')
plt.ylabel('Count')
plt.title('Metrics by Plant Serial Number')
plt.xticks([i + 1.5*bar_width for i in index], index)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[20]:


import matplotlib.pyplot as plt

# Mean metrics
mean_accuracy = sum(svm_metrics['accuracy']) / n_splits
mean_precision = sum(svm_metrics['precision']) / n_splits
mean_recall = sum(svm_metrics['recall']) / n_splits
mean_f1 = sum(svm_metrics['f1']) / n_splits
mean_computation_time = sum(svm_metrics['computation_time']) / n_splits

# Data for line chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Computational Time (seconds)']
metrics_values = [mean_accuracy, mean_precision, mean_recall, mean_f1, mean_computation_time]

# Plot line chart
plt.figure(figsize=(10, 6))
plt.plot(metrics_names, metrics_values, marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Metrics Performance')
plt.xticks(rotation=45)
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()


# In[ ]:





# In[21]:


import numpy as np
from sklearn.model_selection import learning_curve

# Define function to plot learning curves
def plot_learning_curves(estimator, X, y):
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# Plot learning curves
plot_learning_curves(svm_classifier, train_features, train_labels)


# In[ ]:





# In[24]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Compute Precision-Recall and plot curve
precision = dict()
recall = dict()
average_precision = dict()

# Calculate Precision-Recall curve for each class
for i in range(len(unique_labels)):
    precision[i], recall[i], _ = precision_recall_curve(test_labels == i, test_predictions == i)
    average_precision[i] = average_precision_score(test_labels == i, test_predictions == i)

# Plot Precision-Recall curve for each class
plt.figure(figsize=(10, 6))
for i in range(len(unique_labels)):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AP = {average_precision[i]:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.grid(True)
plt.show()


# In[ ]:





# In[96]:


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Calculate ROC curve for each class
for i in range(len(unique_labels)):
    fpr[i], tpr[i], _ = roc_curve(test_labels == i, test_predictions == i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(10, 6))
for i in range(len(unique_labels)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="best")
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# # XGBOOST WITH HOG

# In[ ]:





# In[25]:


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd
import numpy as np

# Load NCA features from CSV for the training set
nca_hog_train_df = pd.read_csv('NCA_hog_train_features.csv')

# Extract labels
labels_train = nca_hog_train_df['label']
features_train = nca_hog_train_df.drop(columns=['label'])

# Initialize LabelEncoder and fit_transform on the labels
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)

# Initialize XGBoost classifier with tuned hyperparameters
xgb_classifier = XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=5,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=len(np.unique(labels_train_encoded)),
    random_state=42
)

# Step 9: Apply cross_val_score and StratifiedKFold to the extracted feature vectors
# Define the number of splits
n_splits = 5

# Initialize StratifiedKFold
stratkf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store metrics for each fold
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
computation_time_list = []

# Loop through each fold
for fold, (train_index, test_index) in enumerate(stratkf.split(features_train, labels_train_encoded), 1):
    # Extract train and test sets for the fold
    X_train_fold, X_test_fold = features_train.iloc[train_index], features_train.iloc[test_index]
    y_train_fold, y_test_fold = labels_train_encoded[train_index], labels_train_encoded[test_index]

    # Train the XGBoost model
    start_time = time.time()
    xgb_classifier.fit(X_train_fold, y_train_fold)
    computation_time = time.time() - start_time

    # Predict on the test set for the current fold
    predictions_fold = xgb_classifier.predict(X_test_fold)

    # Calculate metrics for the fold
    accuracy_fold = accuracy_score(y_test_fold, predictions_fold) * 100
    precision_fold = precision_score(y_test_fold, predictions_fold, average='weighted')
    recall_fold = recall_score(y_test_fold, predictions_fold, average='weighted')
    f1_fold = f1_score(y_test_fold, predictions_fold, average='weighted')

    # Append metrics to lists
    accuracy_list.append(accuracy_fold)
    precision_list.append(precision_fold)
    recall_list.append(recall_fold)
    f1_list.append(f1_fold)
    computation_time_list.append(computation_time)

    # Print metrics for each fold
    print(f"\nFold {fold}:")
    print(f'Accuracy: {accuracy_fold:.2f}%')
    print(f'Precision: {precision_fold:.2f}')
    print(f'Recall: {recall_fold:.2f}')
    print(f'F1 Score: {f1_fold:.2f}')
    print(f'Computation Time: {computation_time:.4f} seconds')

# Step 10: Calculate the mean and standard deviation of metrics
mean_accuracy = np.mean(accuracy_list)
std_accuracy = np.std(accuracy_list)

mean_precision = np.mean(precision_list)
std_precision = np.std(precision_list)

mean_recall = np.mean(recall_list)
std_recall = np.std(recall_list)

mean_f1 = np.mean(f1_list)
std_f1 = np.std(f1_list)

mean_computation_time = np.mean(computation_time_list)

# Print mean and standard deviation of metrics
print("\nMean and Standard Deviation of Metrics:")
print(f'Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}')
print(f'Mean Precision: {mean_precision:.2f} ± {std_precision:.2f}')
print(f'Mean Recall: {mean_recall:.2f} ± {std_recall:.2f}')
print(f'Mean F1 Score: {mean_f1:.2f} ± {std_f1:.2f}')
print(f'Mean Computation Time: {mean_computation_time:.4f} seconds')


# In[ ]:





# # COMPUTATIONAL TIME PER FOLD

# In[26]:


import matplotlib.pyplot as plt

# Assuming you have the SVM metrics data
svm_metrics = {
    'fold_numbers': range(1, n_splits + 1),
    'computation_time': computation_time_list
}

# Plot computational time for each fold
plt.figure(figsize=(10, 6))
plt.plot(svm_metrics['fold_numbers'], svm_metrics['computation_time'], marker='o', linestyle='-', color='b')
plt.title('Computational Time per Fold')
plt.xlabel('Fold')
plt.ylabel('Computation Time (seconds)')
plt.grid(True)
plt.show()


# In[ ]:





# In[27]:


# ... (Previous code)

# Step 12: Box Plot for Metric Distribution
import seaborn as sns

# Combine metrics into a DataFrame for easier plotting
metrics_df = pd.DataFrame({
    'Accuracy': accuracy_list,
    'Precision': precision_list,
    'Recall': recall_list,
    'F1 Score': f1_list,
})

# Plot Box Plot for Metric Distribution
plt.figure(figsize=(12, 8))
sns.boxplot(x='variable', y='value', data=pd.melt(metrics_df))
plt.title('Metric Distribution Across Folds')
plt.ylabel('Metric Value')
plt.xlabel('Metrics')
plt.show()


# In[ ]:





# # FINDING BEST PARAMETERS 

# In[28]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import pandas as pd
import numpy as np

# Load NCA features from CSV for the training set
nca_hog_train_df = pd.read_csv('NCA_hog_train_features.csv')

# Extract labels
labels_train = nca_hog_train_df['label']
features_train = nca_hog_train_df.drop(columns=['label'])

# Initialize LabelEncoder and fit_transform on the labels
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier()

# Define the parameter distributions for randomized search
param_dist = {
    'learning_rate': np.arange(0.01, 0.3, 0.01),
    'max_depth': np.arange(3, 8, 1),
    'min_child_weight': np.arange(1, 6, 1),
    'subsample': np.arange(0.8, 1.1, 0.1),
    'colsample_bytree': np.arange(0.8, 1.1, 0.1),
    'n_estimators': np.arange(100, 301, 50)
}

# Initialize StratifiedKFold
stratkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    xgb_classifier,
    param_distributions=param_dist,
    scoring='accuracy',  # Use accuracy as the scoring metric
    cv=stratkf,  # Use StratifiedKFold for cross-validation
    n_iter=50,  # Number of random combinations to try
    verbose=2,  # Increase verbosity for detailed output
    n_jobs=-1  # Use all available CPU cores
)

# Fit the randomized search to the data
random_search.fit(features_train, labels_train_encoded)

# Print the best parameters and corresponding accuracy
best_params = random_search.best_params_
best_accuracy = random_search.best_score_
print("Best Hyperparameters:")
print(best_params)
print(f"Best Accuracy: {best_accuracy:.2f}")


# In[ ]:





# In[29]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import time

# Load NCA features from CSV for the training set
nca_hog_train_df = pd.read_csv('NCA_hog_train_features.csv')

# Extract labels
labels_train = nca_hog_train_df['label']
features_train = nca_hog_train_df.drop(columns=['label'])

# Initialize LabelEncoder and fit_transform on the labels
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)

# Initialize XGBoost classifier with the best hyperparameters
xgb_classifier = XGBClassifier(
    subsample=0.8,
    n_estimators=250,
    min_child_weight=1,
    max_depth=6,
    learning_rate=0.09,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=len(np.unique(labels_train_encoded)),
    random_state=42
)

# Train the XGBoost model and measure the computational time
start_time = time.time()
xgb_classifier.fit(features_train, labels_train_encoded)
computation_time = time.time() - start_time

# Predict on the training set
predictions_train = xgb_classifier.predict(features_train)

# Calculate metrics
accuracy = accuracy_score(labels_train_encoded, predictions_train) * 100
precision = precision_score(labels_train_encoded, predictions_train, average='weighted') * 100
recall = recall_score(labels_train_encoded, predictions_train, average='weighted') * 100
f1 = f1_score(labels_train_encoded, predictions_train, average='weighted') * 100

# Print the results
print("Results on the whole training dataset:")
print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}%')
print(f'Recall: {recall:.2f}%')
print(f'F1 Score: {f1:.2f}%')
print(f'Computational Time: {computation_time:.4f} seconds')


# In[ ]:





# In[30]:


import joblib

# Save the trained model
joblib.dump(xgb_classifier, 'xgb_model.joblib')

print("Trained model saved successfully as 'xgb_model.joblib'.")


# In[32]:


from sklearn.metrics import classification_report


# # CONFUSION MATRIX & CLASSIFICATION REPORT (XGBOOST + HOG)

# In[39]:


import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved XGBoost model
loaded_model = joblib.load('xgb_model.joblib')

# Load the testing dataset
nca_hog_test_df = pd.read_csv('NCA_hog_test_features.csv')

# Extract labels
labels_test = nca_hog_test_df['label']
features_test = nca_hog_test_df.drop(columns=['label'])

# Use the same LabelEncoder instance to transform the test labels
labels_test_encoded = label_encoder.transform(labels_test)

# Predictions on the testing dataset
predictions_test = loaded_model.predict(features_test)

# Calculate and print classification report
print("Classification Report:")
print(classification_report(labels_test_encoded, predictions_test, target_names=label_encoder.classes_))

# Calculate confusion matrix
conf_matrix = confusion_matrix(labels_test_encoded, predictions_test)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[40]:


# Load the saved XGBoost model
loaded_model = joblib.load('xgb_model.joblib')

# Load the testing dataset
nca_hog_test_df = pd.read_csv('NCA_hog_test_features.csv')

# Extract labels
labels_test = nca_hog_test_df['label']
features_test = nca_hog_test_df.drop(columns=['label'])

# Use the same LabelEncoder instance to transform the test labels
labels_test_encoded = label_encoder.transform(labels_test)

# Predictions on the testing dataset
predictions_test = loaded_model.predict(features_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(labels_test_encoded, predictions_test)

# Initialize dictionaries to store TP, TN, FP, FN for each class
class_metrics = {}

# Iterate through the confusion matrix
for i, class_label in enumerate(label_encoder.classes_):
    # True positives (TP): diagonal element (i, i)
    tp = conf_matrix[i, i]
    
    # True negatives (TN): sum of all elements except row i and column i
    tn = np.sum(np.delete(np.delete(conf_matrix, i, 0), i, 1))
    
    # False positives (FP): sum of column i excluding diagonal element
    fp = np.sum(conf_matrix[:, i]) - tp
    
    # False negatives (FN): sum of row i excluding diagonal element
    fn = np.sum(conf_matrix[i, :]) - tp
    
    # Store the metrics for the class
    class_metrics[class_label] = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

# Print the metrics for each class
for class_label, metrics in class_metrics.items():
    print(f"Metrics for {class_label}:")
    print(f"True Positives (TP): {metrics['TP']}")
    print(f"True Negatives (TN): {metrics['TN']}")
    print(f"False Positives (FP): {metrics['FP']}")
    print(f"False Negatives (FN): {metrics['FN']}")
    print()


# In[ ]:





# In[42]:


import numpy as np
import time

# Initialize lists to store metrics for each plant
accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []

# Start time
start_time = time.time()

# Iterate through the metrics for each plant
for class_label, metrics in class_metrics.items():
    tp = metrics['TP']
    tn = metrics['TN']
    fp = metrics['FP']
    fn = metrics['FN']
    
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    
    # Calculate precision
    precision = tp / (tp + fp) * 100
    
    # Calculate recall
    recall = tp / (tp + fn) * 100
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Append metrics to respective lists
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1_score)

# Calculate average metrics
average_accuracy = np.mean(accuracy_list)
average_precision = np.mean(precision_list)
average_recall = np.mean(recall_list)
average_f1_score = np.mean(f1_score_list)

# End time
end_time = time.time()

# Calculate computational time
computational_time = end_time - start_time

# Print the average metrics and computational time
print(f"Average Accuracy: {average_accuracy:.2f}%")
print(f"Average Precision: {average_precision:.2f}%")
print(f"Average Recall: {average_recall:.2f}%")
print(f"Average F1 Score: {average_f1_score:.2f}%")
print(f"Computational Time: {computational_time:.2f} seconds")


# In[ ]:





# In[43]:


import matplotlib.pyplot as plt

# Define serial numbers of plants based on their order in the confusion matrix
serial_numbers = range(len(label_encoder.classes_))

# Extract TP, TN, FP, FN values for each plant
tp_values = [class_metrics[label]['TP'] for label in label_encoder.classes_]
tn_values = [class_metrics[label]['TN'] for label in label_encoder.classes_]
fp_values = [class_metrics[label]['FP'] for label in label_encoder.classes_]
fn_values = [class_metrics[label]['FN'] for label in label_encoder.classes_]

# Plot bar chart
plt.figure(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(serial_numbers))

plt.bar(index, tp_values, bar_width, label='True Positives')
plt.bar(index + bar_width, tn_values, bar_width, label='True Negatives')
plt.bar(index + 2*bar_width, fp_values, bar_width, label='False Positives')
plt.bar(index + 3*bar_width, fn_values, bar_width, label='False Negatives')

plt.xlabel('Serial Numbers of Plants')
plt.ylabel('Count')
plt.title('Counts of TP, TN, FP, FN for Each Plant')
plt.xticks(index + bar_width, serial_numbers)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[44]:


import matplotlib.pyplot as plt

# Define the x-axis values (e.g., fold numbers or serial numbers of plants)
x_values = range(1, n_splits + 1)  # Adjust according to your data

# Define y-axis values for each metric
accuracy_values = [mean_accuracy] * len(x_values)  # Adjust based on your data
precision_values = [mean_precision] * len(x_values)  # Adjust based on your data
recall_values = [mean_recall] * len(x_values)  # Adjust based on your data
f1_values = [mean_f1] * len(x_values)  # Adjust based on your data
computation_time_values = [mean_computation_time] * len(x_values)  # Adjust based on your data

# Plot the lines for each metric
plt.figure(figsize=(10, 6))

plt.plot(x_values, accuracy_values, label='Accuracy', marker='o')
plt.plot(x_values, precision_values, label='Precision', marker='o')
plt.plot(x_values, recall_values, label='Recall', marker='o')
plt.plot(x_values, f1_values, label='F1-Score', marker='o')
plt.plot(x_values, computation_time_values, label='Computational Time', marker='o')

plt.title('Metrics Over Folds')
plt.xlabel('Fold Number')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# In[45]:


import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(train_sizes, train_scores, val_scores, title='Learning Curves', ylim=None):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Example usage:
train_sizes = [100, 200, 300, 400, 500]  # Adjust based on your data
train_scores = np.random.rand(5, 3)  # Example train scores (5 sizes, 3 repetitions)
val_scores = np.random.rand(5, 3)  # Example validation scores (5 sizes, 3 repetitions)

plot_learning_curves(train_sizes, train_scores, val_scores, title='Learning Curves')
plt.show()


# In[ ]:





# In[46]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

# Example usage:
# Replace y_true and y_scores with your actual data
y_true = [0, 1, 1, 0, 1, 0, 0, 1]
y_scores = [0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.4, 0.6]  # Example scores (probabilities or decision function scores)

plot_precision_recall_curve(y_true, y_scores)


# In[ ]:





# In[47]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Example usage:
# Replace y_true and y_scores with your actual data
y_true = [0, 1, 1, 0, 1, 0, 0, 1]
y_scores = [0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.4, 0.6]  # Example scores (probabilities or decision function scores)

plot_roc_curve(y_true, y_scores)


# In[ ]:





# # GLCM EXTRACTION - XGBOOST WITH GLCM

# In[48]:


import os
import cv2
import mahotas
import pandas as pd
from skimage import io
from tqdm import tqdm

# Step 5: Extract GLCM features and save to CSV
final_splitted_data_path = 'final_splitted_data'
glcm_train_csv_path = 'glcm_train_features.csv'
glcm_test_csv_path = 'glcm_test_features.csv'

# Lists to store GLCM features and labels
glcm_train_features = []
glcm_train_labels = []
glcm_test_features = []
glcm_test_labels = []

# Function to calculate GLCM features for a single image
def calculate_glcm_features(image_path):
    # Read the image using skimage
    image = io.imread(image_path, as_gray=True)

    # Convert image to unsigned 8-bit integer (required by mahotas)
    image = (image * 255).astype('uint8')

    # Calculate GLCM
    glcm = mahotas.features.haralick(image)

    # Flatten the GLCM matrix to a 1D array
    glcm_features = glcm.flatten()

    return glcm_features

# Extract GLCM features for 'res_train_data' folder
for plant_folder in tqdm(os.listdir(final_splitted_data_path), desc="Processing Plants"):
    res_train_folder = os.path.join(final_splitted_data_path, plant_folder, 'res_train_data')

    for image_name in os.listdir(res_train_folder):
        image_path = os.path.join(res_train_folder, image_name)

        # Calculate GLCM features for the current image
        glcm_features = calculate_glcm_features(image_path)

        # Append the features along with the plant label to the list
        glcm_train_features.append([plant_folder] + list(glcm_features))

# Extract GLCM features for 'res_test_data' folder
for plant_folder in tqdm(os.listdir(final_splitted_data_path), desc="Processing Plants"):
    res_test_folder = os.path.join(final_splitted_data_path, plant_folder, 'res_test_data')

    for image_name in os.listdir(res_test_folder):
        image_path = os.path.join(res_test_folder, image_name)

        # Calculate GLCM features for the current image
        glcm_features = calculate_glcm_features(image_path)

        # Append the features along with the plant label to the list
        glcm_test_features.append([plant_folder] + list(glcm_features))

# Create DataFrames from the lists of features
columns = ['Plant'] + [f'GLCM_{i}' for i in range(len(glcm_train_features[0]) - 1)]
glcm_train_df = pd.DataFrame(glcm_train_features, columns=columns)

columns = ['Plant'] + [f'GLCM_{i}' for i in range(len(glcm_test_features[0]) - 1)]
glcm_test_df = pd.DataFrame(glcm_test_features, columns=columns)

# Save GLCM features to CSV
glcm_train_df.to_csv(glcm_train_csv_path, index=False)
glcm_test_df.to_csv(glcm_test_csv_path, index=False)

print("Step 5: GLCM features extracted and saved to CSV successfully.")


# In[ ]:





# # MINMAX NORMALIZATION APPLIED

# In[49]:


from sklearn.preprocessing import MinMaxScaler

# Step 6: Apply Min-Max normalization to GLCM features
glcm_train_csv_path = 'glcm_train_features.csv'
glcm_test_csv_path = 'glcm_test_features.csv'
norm_glcm_train_csv_path = 'norm_glcm_train_features.csv'
norm_glcm_test_csv_path = 'norm_glcm_test_features.csv'

# Load GLCM features from CSV
glcm_train_df = pd.read_csv(glcm_train_csv_path)
glcm_test_df = pd.read_csv(glcm_test_csv_path)

# Extract labels and features from DataFrames
labels_train = glcm_train_df['Plant']
features_train = glcm_train_df.drop(columns=['Plant'])

labels_test = glcm_test_df['Plant']
features_test = glcm_test_df.drop(columns=['Plant'])

# Apply Min-Max normalization
scaler = MinMaxScaler()
norm_features_train = scaler.fit_transform(features_train)
norm_features_test = scaler.transform(features_test)

# Create DataFrames for normalized features
norm_glcm_train_df = pd.DataFrame(norm_features_train, columns=features_train.columns)
norm_glcm_train_df['Plant'] = labels_train

norm_glcm_test_df = pd.DataFrame(norm_features_test, columns=features_test.columns)
norm_glcm_test_df['Plant'] = labels_test

# Save normalized features to CSV
norm_glcm_train_df.to_csv(norm_glcm_train_csv_path, index=False)
norm_glcm_test_df.to_csv(norm_glcm_test_csv_path, index=False)

print("Step 6: Min-Max normalization applied and normalized GLCM features saved to CSV successfully.")


# In[ ]:





# # PCA & NCA Applied

# In[50]:


from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis

# Step 7: Apply PCA and NCA to normalized GLCM features
norm_glcm_train_csv_path = 'norm_glcm_train_features.csv'
norm_glcm_test_csv_path = 'norm_glcm_test_features.csv'
nca_glcm_train_csv_path = 'NCA_glcm_train_features.csv'
nca_glcm_test_csv_path = 'NCA_glcm_test_features.csv'

# Load normalized GLCM features from CSV
norm_glcm_train_df = pd.read_csv(norm_glcm_train_csv_path)
norm_glcm_test_df = pd.read_csv(norm_glcm_test_csv_path)

# Extract labels and features from DataFrames
labels_train = norm_glcm_train_df['Plant']
features_train = norm_glcm_train_df.drop(columns=['Plant'])

labels_test = norm_glcm_test_df['Plant']
features_test = norm_glcm_test_df.drop(columns=['Plant'])

# Apply PCA
pca = PCA(n_components=50)
features_train_pca = pca.fit_transform(features_train)
features_test_pca = pca.transform(features_test)

# Apply NCA to reduced-dimensional data
nca = NeighborhoodComponentsAnalysis()
features_train_nca = nca.fit_transform(features_train_pca, labels_train)
features_test_nca = nca.transform(features_test_pca)

# Create DataFrames for NCA features
nca_glcm_train_df = pd.DataFrame(features_train_nca, columns=[f'NCA_{i}' for i in range(features_train_nca.shape[1])])
nca_glcm_train_df['Plant'] = labels_train

nca_glcm_test_df = pd.DataFrame(features_test_nca, columns=[f'NCA_{i}' for i in range(features_test_nca.shape[1])])
nca_glcm_test_df['Plant'] = labels_test

# Save NCA features to CSV
nca_glcm_train_df.to_csv(nca_glcm_train_csv_path, index=False)
nca_glcm_test_df.to_csv(nca_glcm_test_csv_path, index=False)

print("Step 7: PCA and NCA applied, and NCA features saved to CSV successfully.")


# In[ ]:





# # CALCULATION OF ACCURACY , PRECISION , RECALL AND F1 - SCORE (FOR GLCM)

# In[51]:


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import time

# Load NCA features from CSV
nca_glcm_train_csv_path = 'NCA_glcm_train_features.csv'
nca_glcm_test_csv_path = 'NCA_glcm_test_features.csv'

# Load NCA features from CSV
nca_glcm_train_df = pd.read_csv(nca_glcm_train_csv_path)
nca_glcm_test_df = pd.read_csv(nca_glcm_test_csv_path)

# Extract labels and features from DataFrames
y_train = nca_glcm_train_df['Plant']
X_train = nca_glcm_train_df.drop(columns=['Plant'])

y_test = nca_glcm_test_df['Plant']
X_test = nca_glcm_test_df.drop(columns=['Plant'])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode string labels to numerical labels
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize StratifiedKFold with n_splits=5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
computation_times = []

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train_encoded), 1):
    # Split data into train and test sets for this fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train_encoded[train_index], y_train_encoded[test_index]
    
    # Initialize K-Nearest Neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Train the classifier
    start_time = time.time()
    knn.fit(X_train_fold, y_train_fold)
    end_time = time.time()
    
    # Predict labels on the test set
    predictions_fold = knn.predict(X_test_fold)
    
    # Calculate metrics for the fold
    accuracy_fold = accuracy_score(y_test_fold, predictions_fold) * 100
    precision_fold = precision_score(y_test_fold, predictions_fold, average='macro')
    recall_fold = recall_score(y_test_fold, predictions_fold, average='macro')
    f1_fold = f1_score(y_test_fold, predictions_fold, average='macro')
    computation_time = end_time - start_time
    
    # Append metrics to lists
    accuracy_scores.append(accuracy_fold)
    precision_scores.append(precision_fold)
    recall_scores.append(recall_fold)
    f1_scores.append(f1_fold)
    computation_times.append(computation_time)
    
    # Print metrics for the fold
    print(f"\nFold {fold}:")
    print(f'Accuracy: {accuracy_fold:.2f}%')
    print(f'Precision: {precision_fold:.2f}')
    print(f'Recall: {recall_fold:.2f}')
    print(f'F1 Score: {f1_fold:.2f}')
    print(f'Computation Time: {computation_time:.4f} seconds')

# Calculate mean and standard deviation of metrics
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
std_precision = np.std(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
std_recall = np.std(recall_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
std_f1 = np.std(f1_scores)
mean_computation_time = sum(computation_times) / len(computation_times)

# Print mean and standard deviation of metrics
print("\nMean and Standard Deviation of Metrics:")
print(f'Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}')
print(f'Mean Precision: {mean_precision:.2f} ± {std_precision:.2f}')
print(f'Mean Recall: {mean_recall:.2f} ± {std_recall:.2f}')
print(f'Mean F1 Score: {mean_f1:.2f} ± {std_f1:.2f}')
print(f'Mean Computation Time: {mean_computation_time:.4f} seconds')


# In[ ]:





# In[52]:


import numpy as np

# Calculate mean and standard deviation of metrics
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)
mean_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
mean_computation_time = np.mean(computation_times)

# Print mean and standard deviation of metrics
print("\nMean and Standard Deviation of Metrics Across All Folds:")
print(f'Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}')
print(f'Mean Precision: {mean_precision:.2f} ± {std_precision:.2f}')
print(f'Mean Recall: {mean_recall:.2f} ± {std_recall:.2f}')
print(f'Mean F1 Score: {mean_f1:.2f} ± {std_f1:.2f}')
print(f'Mean Computation Time: {mean_computation_time:.4f} seconds')


# In[ ]:





# In[53]:


import matplotlib.pyplot as plt

# Calculate computational time per fold
computation_times_per_fold = np.array(computation_times)

# Plot computational time per fold
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(computation_times_per_fold) + 1), computation_times_per_fold, marker='o', color='b')
plt.title('Computational Time per Fold')
plt.xlabel('Fold')
plt.ylabel('Computational Time (seconds)')
plt.grid(True)
plt.show()


# In[ ]:





# In[54]:


import matplotlib.pyplot as plt

# Metrics data
metrics_data = {
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores
}

# Create a box plot for each metric
plt.figure(figsize=(10, 6))
plt.boxplot(metrics_data.values(), labels=metrics_data.keys())
plt.title('Distribution of Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.grid(True)
plt.show()


# In[ ]:





# In[55]:


import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

# Load NCA GLCM features from CSV for the training set
nca_glcm_train_df = pd.read_csv('NCA_glcm_train_features.csv')

# Assuming your CSV file contains features and labels columns
features_train = nca_glcm_train_df.drop(columns=['Plant'])  # Update 'label' with the actual column name
labels_train = nca_glcm_train_df['Plant']  # Update 'label' with the actual column name

# Encode labels to integers
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)

# Define the parameter distributions for randomized search
param_dist = {
    'learning_rate': uniform(0.01, 0.2 - 0.01),
    'max_depth': randint(3, 8),
    'min_child_weight': randint(1, 6),
    'subsample': uniform(0.8, 1.0 - 0.8),
    'colsample_bytree': uniform(0.8, 1.0 - 0.8),
    'n_estimators': randint(100, 301)
}

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier()

# Initialize StratifiedKFold
stratkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize RandomizedSearchCV
randomized_search = RandomizedSearchCV(
    xgb_classifier,
    param_distributions=param_dist,
    n_iter=100,  # Adjust the number of iterations as needed
    scoring='accuracy',  # Use accuracy as the scoring metric
    cv=stratkf,  # Use StratifiedKFold for cross-validation
    verbose=2,  # Increase verbosity for detailed output
    n_jobs=-1  # Use all available CPU cores
)

# Fit the randomized search to the data
randomized_search.fit(features_train, labels_train_encoded)

# Print the best parameters and corresponding accuracy
best_params = randomized_search.best_params_
best_accuracy = randomized_search.best_score_
print("Best Hyperparameters:")
print(best_params)
print(f"Best Accuracy: {best_accuracy:.2f}")


# In[ ]:





# In[56]:


import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Load NCA GLCM features from CSV for the training set
nca_glcm_train_df = pd.read_csv('NCA_glcm_train_features.csv')

# Assuming your CSV file contains features and labels columns
features_train = nca_glcm_train_df.drop(columns=['Plant'])  # Update 'label' with the actual column name
labels_train = nca_glcm_train_df['Plant']  # Update 'label' with the actual column name

# Encode labels to integers
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)

# Initialize XGBoost classifier with the best hyperparameters
xgb_classifier = XGBClassifier(
    colsample_bytree=0.8060664215236861,
    learning_rate= 0.15771142658314577,
    max_depth=4,
    min_child_weight=1,
    n_estimators=207,
    subsample=0.9829269087326896
)

# Train the model
start_time = time.time()
xgb_classifier.fit(features_train, labels_train_encoded)
end_time = time.time()

# Calculate predictions
predictions_train = xgb_classifier.predict(features_train)

# Calculate metrics
accuracy = accuracy_score(labels_train_encoded, predictions_train) * 100
precision = precision_score(labels_train_encoded, predictions_train, average='weighted')
recall = recall_score(labels_train_encoded, predictions_train, average='weighted')
f1 = f1_score(labels_train_encoded, predictions_train, average='weighted')
computation_time = end_time - start_time

# Print metrics
print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Computational Time: {computation_time:.4f} seconds')


# In[ ]:





# In[57]:


from joblib import dump

# Save the trained model
dump(xgb_classifier, 'xgb_model2.joblib')


# In[ ]:





# In[66]:


import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load

# Load the saved XGBoost model
xgb_model = load('xgb_model2.joblib')

# Load the testing dataset
test_df = pd.read_csv('NCA_glcm_test_features.csv')

# Extract features and labels
X_test = test_df.drop(columns=['Plant'])  # Assuming 'Plant' is the label column
y_test = test_df['Plant']  # Assuming 'Plant' is the label column

# Encode labels to integers if they are not already encoded
# Assuming label_encoder is the instance used for encoding during training
y_test_encoded = label_encoder.transform(y_test)

# Make predictions on the testing dataset
y_pred = xgb_model.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()


# In[ ]:





# In[67]:


from sklearn.metrics import classification_report

# Assuming you have already loaded the model and the testing dataset
# Replace 'xgb_model' with your loaded model variable and 'X_test' with your testing dataset

# Make predictions
y_pred = xgb_model.predict(X_test)

# Generate classification report
report = classification_report(y_test_encoded, y_pred)

# Print classification report
print("Classification Report:")
print(report)


# In[ ]:





# In[70]:


import pandas as pd

# Initialize lists to store data
plant_names = []
tp_list = []
tn_list = []
fp_list = []
fn_list = []

# Iterate over each class
for class_label in range(len(label_encoder.classes_)):
    plant_name = label_encoder.inverse_transform([class_label])[0]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    # Iterate over each sample
    for true_label, pred_label in zip(y_test_encoded, y_pred):
        # Check if the sample belongs to the current class
        if true_label == class_label:
            # Check if the prediction is correct (TP)
            if pred_label == true_label:
                tp += 1
            else:
                fn += 1  # Incorrect prediction (FN)
        else:
            # Check if the prediction is correct (TN)
            if pred_label != class_label:
                tn += 1
            else:
                fp += 1  # Incorrect prediction (FP)
    
    # Append data to lists
    plant_names.append(plant_name)
    tp_list.append(tp)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)

# Create a DataFrame
results_df = pd.DataFrame({
    'Plant': plant_names,
    'TP': tp_list,
    'TN': tn_list,
    'FP': fp_list,
    'FN': fn_list
})

# Print the DataFrame
print("TP, TN, FP, FN for each plant:")
print(results_df)


# In[ ]:





# In[71]:


# Define functions to calculate metrics
def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def f1_score(prec, rec):
    return 2 * (prec * rec) / (prec + rec)

# Initialize lists to store metrics for each plant
plant_names = results_df['Plant']
acc_list = []
prec_list = []
rec_list = []
f1_list = []

# Calculate metrics for each plant
for index, row in results_df.iterrows():
    tp = row['TP']
    tn = row['TN']
    fp = row['FP']
    fn = row['FN']
    
    acc = accuracy(tp, tn, fp, fn)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)
    
    acc_list.append(acc)
    prec_list.append(prec)
    rec_list.append(rec)
    f1_list.append(f1)

# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame({
    'Plant': plant_names,
    'Accuracy': acc_list,
    'Precision': prec_list,
    'Recall': rec_list,
    'F1 Score': f1_list
})

# Print the DataFrame
print("Metrics for each plant:")
print(metrics_df)


# In[73]:


import numpy as np

# Extract metrics from the DataFrame
accuracy_values = metrics_df['Accuracy']
precision_values = metrics_df['Precision']
recall_values = metrics_df['Recall']
f1_values = metrics_df['F1 Score']

# Calculate average metrics
average_accuracy = np.mean(accuracy_values) * 100  # Convert to percentage
average_precision = np.mean(precision_values) * 100  # Convert to percentage
average_recall = np.mean(recall_values) * 100  # Convert to percentage
average_f1_score = np.mean(f1_values) * 100  # Convert to percentage

# Calculate average computational time (assuming uniform time for each plant calculation)
# Replace 'time_per_plant' with the actual time taken for each plant's calculation
time_per_plant = 10  # Placeholder value (replace with actual time in seconds)
number_of_plants = len(metrics_df)
average_computational_time = time_per_plant / number_of_plants

# Print the results
print("Average Accuracy: {:.2f}%".format(average_accuracy))
print("Average Precision: {:.2f}%".format(average_precision))
print("Average Recall: {:.2f}%".format(average_recall))
print("Average F1 Score: {:.2f}%".format(average_f1_score))
print("Average Computational Time: {:.2f} seconds".format(average_computational_time))


# In[75]:


import matplotlib.pyplot as plt

# Data provided for TP, TN, FP, FN for each plant class
plant_data = [
    {"plant": "Alpinia Galanga (Rasna)", "TP": 65, "TN": 2676, "FP": 7, "FN": 10},
    {"plant": "Amaranthus Viridis (Arive-Dantu)", "TP": 162, "TN": 2540, "FP": 35, "FN": 21},
    {"plant": "Artocarpus Heterophyllus (Jackfruit)", "TP": 74, "TN": 2671, "FP": 3, "FN": 10},
    {"plant": "Azadirachta Indica (Neem)", "TP": 78, "TN": 2659, "FP": 9, "FN": 12},
    {"plant": "Basella Alba (Basale)", "TP": 125, "TN": 2573, "FP": 30, "FN": 30},
    {"plant": "Brassica Juncea (Indian Mustard)", "TP": 37, "TN": 2704, "FP": 3, "FN": 14},
    {"plant": "Carissa Carandas (Karanda)", "TP": 98, "TN": 2634, "FP": 13, "FN": 13},
    {"plant": "Citrus Limon (Lemon)", "TP": 71, "TN": 2655, "FP": 17, "FN": 15},
    {"plant": "Ficus Auriculata (Roxburgh fig)", "TP": 57, "TN": 2661, "FP": 22, "FN": 18},
    {"plant": "Ficus Religiosa (Peepal Tree)", "TP": 91, "TN": 2644, "FP": 19, "FN": 4},
    {"plant": "Hibiscus Rosa-sinensis", "TP": 48, "TN": 2682, "FP": 11, "FN": 17},
    {"plant": "Jasminum (Jasmine)", "TP": 96, "TN": 2634, "FP": 17, "FN": 11},
    {"plant": "Mangifera Indica (Mango)", "TP": 71, "TN": 2639, "FP": 26, "FN": 22},
    {"plant": "Mentha (Mint)", "TP": 129, "TN": 2588, "FP": 24, "FN": 17},
    {"plant": "Moringa Oleifera (Drumstick)", "TP": 94, "TN": 2629, "FP": 13, "FN": 22},
    {"plant": "Muntingia Calabura (Jamaica Cherry-Gasagase)", "TP": 74, "TN": 2665, "FP": 9, "FN": 10},
    {"plant": "Murraya Koenigii (Curry)", "TP": 80, "TN": 2656, "FP": 12, "FN": 10},
    {"plant": "Nerium Oleander (Oleander)", "TP": 66, "TN": 2639, "FP": 26, "FN": 27},
    {"plant": "Nyctanthes Arbor-tristis (Parijata)", "TP": 51, "TN": 2693, "FP": 5, "FN": 9},
    {"plant": "Ocimum Tenuiflorum (Tulsi)", "TP": 56, "TN": 2666, "FP": 14, "FN": 22},
    {"plant": "Piper Betle (Betel)", "TP": 55, "TN": 2675, "FP": 11, "FN": 17},
    {"plant": "Plectranthus Amboinicus (Mexican Mint)", "TP": 63, "TN": 2677, "FP": 9, "FN": 9},
    {"plant": "Pongamia Pinnata (Indian Beech)", "TP": 80, "TN": 2652, "FP": 14, "FN": 12},
    {"plant": "Psidium Guajava (Guava)", "TP": 98, "TN": 2656, "FP": 4, "FN": 0},
    {"plant": "Punica Granatum (Pomegranate)", "TP": 95, "TN": 2624, "FP": 15, "FN": 24},
    {"plant": "Santalum Album (Sandalwood)", "TP": 77, "TN": 2652, "FP": 19, "FN": 10},
    {"plant": "Syzygium Cumini (Jamun)", "TP": 55, "TN": 2693, "FP": 6, "FN": 4},
    {"plant": "Syzygium Jambos (Rose Apple)", "TP": 77, "TN": 2667, "FP": 7, "FN": 7},
    {"plant": "Tabernaemontana Divaricata (Crape Jasmine)", "TP": 73, "TN": 2666, "FP": 8, "FN": 11},
    {"plant": "Trigonella Foenum-graecum (Fenugreek)", "TP": 51, "TN": 2701, "FP": 3, "FN": 3}
]

# Extracting data for plotting
plants = [data["plant"] for data in plant_data]
TP_values = [data["TP"] for data in plant_data]
TN_values = [data["TN"] for data in plant_data]
FP_values = [data["FP"] for data in plant_data]
FN_values = [data["FN"] for data in plant_data]

# Bar chart for TP
plt.figure(figsize=(12, 6))
plt.bar(plants, TP_values, color='b', label='TP')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('True Positive (TP) for Each Plant Class')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart for TN
plt.figure(figsize=(12, 6))
plt.bar(plants, TN_values, color='g', label='TN')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('True Negative (TN) for Each Plant Class')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart for FP
plt.figure(figsize=(12, 6))
plt.bar(plants, FP_values, color='r', label='FP')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('False Positive (FP) for Each Plant Class')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart for FN
plt.figure(figsize=(12, 6))
plt.bar(plants, FN_values, color='orange', label='FN')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('False Negative (FN) for Each Plant Class')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[76]:


import matplotlib.pyplot as plt
import numpy as np

# Data provided for TP, TN, FP, FN for each plant class
plant_data = [
    {"plant": "Alpinia Galanga (Rasna)", "TP": 65, "TN": 2676, "FP": 7, "FN": 10},
    {"plant": "Amaranthus Viridis (Arive-Dantu)", "TP": 162, "TN": 2540, "FP": 35, "FN": 21},
    {"plant": "Artocarpus Heterophyllus (Jackfruit)", "TP": 74, "TN": 2671, "FP": 3, "FN": 10},
    {"plant": "Azadirachta Indica (Neem)", "TP": 78, "TN": 2659, "FP": 9, "FN": 12},
    {"plant": "Basella Alba (Basale)", "TP": 125, "TN": 2573, "FP": 30, "FN": 30},
    {"plant": "Brassica Juncea (Indian Mustard)", "TP": 37, "TN": 2704, "FP": 3, "FN": 14},
    {"plant": "Carissa Carandas (Karanda)", "TP": 98, "TN": 2634, "FP": 13, "FN": 13},
    {"plant": "Citrus Limon (Lemon)", "TP": 71, "TN": 2655, "FP": 17, "FN": 15},
    {"plant": "Ficus Auriculata (Roxburgh fig)", "TP": 57, "TN": 2661, "FP": 22, "FN": 18},
    {"plant": "Ficus Religiosa (Peepal Tree)", "TP": 91, "TN": 2644, "FP": 19, "FN": 4},
    {"plant": "Hibiscus Rosa-sinensis", "TP": 48, "TN": 2682, "FP": 11, "FN": 17},
    {"plant": "Jasminum (Jasmine)", "TP": 96, "TN": 2634, "FP": 17, "FN": 11},
    {"plant": "Mangifera Indica (Mango)", "TP": 71, "TN": 2639, "FP": 26, "FN": 22},
    {"plant": "Mentha (Mint)", "TP": 129, "TN": 2588, "FP": 24, "FN": 17},
    {"plant": "Moringa Oleifera (Drumstick)", "TP": 94, "TN": 2629, "FP": 13, "FN": 22},
    {"plant": "Muntingia Calabura (Jamaica Cherry-Gasagase)", "TP": 74, "TN": 2665, "FP": 9, "FN": 10},
    {"plant": "Murraya Koenigii (Curry)", "TP": 80, "TN": 2656, "FP": 12, "FN": 10},
    {"plant": "Nerium Oleander (Oleander)", "TP": 66, "TN": 2639, "FP": 26, "FN": 27},
    {"plant": "Nyctanthes Arbor-tristis (Parijata)", "TP": 51, "TN": 2693, "FP": 5, "FN": 9},
    {"plant": "Ocimum Tenuiflorum (Tulsi)", "TP": 56, "TN": 2666, "FP": 14, "FN": 22},
    {"plant": "Piper Betle (Betel)", "TP": 55, "TN": 2675, "FP": 11, "FN": 17},
    {"plant": "Plectranthus Amboinicus (Mexican Mint)", "TP": 63, "TN": 2677, "FP": 9, "FN": 9},
    {"plant": "Pongamia Pinnata (Indian Beech)", "TP": 80, "TN": 2652, "FP": 14, "FN": 12},
    {"plant": "Psidium Guajava (Guava)", "TP": 98, "TN": 2656, "FP": 4, "FN": 0},
    {"plant": "Punica Granatum (Pomegranate)", "TP": 95, "TN": 2624, "FP": 15, "FN": 24},
    {"plant": "Santalum Album (Sandalwood)", "TP": 77, "TN": 2652, "FP": 19, "FN": 10},
    {"plant": "Syzygium Cumini (Jamun)", "TP": 55, "TN": 2693, "FP": 6, "FN": 4},
    {"plant": "Syzygium Jambos (Rose Apple)", "TP": 77, "TN": 2667, "FP": 7, "FN": 7},
    {"plant": "Tabernaemontana Divaricata (Crape Jasmine)", "TP": 73, "TN": 2666, "FP": 8, "FN": 11},
    {"plant": "Trigonella Foenum-graecum (Fenugreek)", "TP": 51, "TN": 2701, "FP": 3, "FN": 3}
]

# Extracting data for plotting
plants = [i for i in range(1, len(plant_data) + 1)]
TP_values = [data["TP"] for data in plant_data]
TN_values = [data["TN"] for data in plant_data]
FP_values = [data["FP"] for data in plant_data]
FN_values = [data["FN"] for data in plant_data]

# Bar chart for TP, TN, FP, FN for each plant
bar_width = 0.2
index = np.arange(len(plants))

plt.figure(figsize=(14, 8))

plt.bar(index - bar_width, TP_values, bar_width, color='b', label='TP')
plt.bar(index, TN_values, bar_width, color='g', label='TN')
plt.bar(index + bar_width, FP_values, bar_width, color='r', label='FP')
plt.bar(index + 2 * bar_width, FN_values, bar_width, color='orange', label='FN')

plt.xlabel('Plant Serial Number')
plt.ylabel('Count')
plt.title('TP, TN, FP, FN for Each Plant Class')
plt.xticks(index, plants)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[109]:


import matplotlib.pyplot as plt

# Define data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [99.01, 85.69, 84.84, 85.15]  # Replace with your values
computational_time = 0.33  # Replace with your computational time in seconds

# Plot line chart
plt.figure(figsize=(10, 6))

# Plot metrics
plt.plot(metrics, values, marker='o', label='Metrics', color='blue')

# Plot computational time
plt.axhline(y=computational_time, linestyle='--', label='Computational Time', color='red')

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Performance Metrics and Computational Time')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[78]:


import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(train_sizes, train_scores, val_scores, title='Learning Curves', ylim=None):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Example usage:
train_sizes = [100, 200, 300, 400, 500]  # Adjust based on your data
train_scores = np.random.rand(5, 3)  # Example train scores (5 sizes, 3 repetitions)
val_scores = np.random.rand(5, 3)  # Example validation scores (5 sizes, 3 repetitions)

plot_learning_curves(train_sizes, train_scores, val_scores, title='Learning Curves')
plt.show()


# In[ ]:





# In[84]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np

# Data for TP, TN, FP, FN for each plant class
plant_data = [
    {"plant": "Alpinia Galanga (Rasna)", "TP": 65, "TN": 2676, "FP": 7, "FN": 10},
    {"plant": "Amaranthus Viridis (Arive-Dantu)", "TP": 162, "TN": 2540, "FP": 35, "FN": 21},
    {"plant": "Artocarpus Heterophyllus (Jackfruit)", "TP": 74, "TN": 2671, "FP": 3, "FN": 10},
    {"plant": "Azadirachta Indica (Neem)", "TP": 78, "TN": 2659, "FP": 9, "FN": 12},
    {"plant": "Basella Alba (Basale)", "TP": 125, "TN": 2573, "FP": 30, "FN": 30},
    {"plant": "Brassica Juncea (Indian Mustard)", "TP": 37, "TN": 2704, "FP": 3, "FN": 14},
    {"plant": "Carissa Carandas (Karanda)", "TP": 98, "TN": 2634, "FP": 13, "FN": 13},
    {"plant": "Citrus Limon (Lemon)", "TP": 71, "TN": 2655, "FP": 17, "FN": 15},
    {"plant": "Ficus Auriculata (Roxburgh fig)", "TP": 57, "TN": 2661, "FP": 22, "FN": 18},
    {"plant": "Ficus Religiosa (Peepal Tree)", "TP": 91, "TN": 2644, "FP": 19, "FN": 4},
    {"plant": "Hibiscus Rosa-sinensis", "TP": 48, "TN": 2682, "FP": 11, "FN": 17},
    {"plant": "Jasminum (Jasmine)", "TP": 96, "TN": 2634, "FP": 17, "FN": 11},
    {"plant": "Mangifera Indica (Mango)", "TP": 71, "TN": 2639, "FP": 26, "FN": 22},
    {"plant": "Mentha (Mint)", "TP": 129, "TN": 2588, "FP": 24, "FN": 17},
    {"plant": "Moringa Oleifera (Drumstick)", "TP": 94, "TN": 2629, "FP": 13, "FN": 22},
    {"plant": "Muntingia Calabura (Jamaica Cherry-Gasagase)", "TP": 74, "TN": 2665, "FP": 9, "FN": 10},
    {"plant": "Murraya Koenigii (Curry)", "TP": 80, "TN": 2656, "FP": 12, "FN": 10},
    {"plant": "Nerium Oleander (Oleander)", "TP": 66, "TN": 2639, "FP": 26, "FN": 27},
    {"plant": "Nyctanthes Arbor-tristis (Parijata)", "TP": 51, "TN": 2693, "FP": 5, "FN": 9},
    {"plant": "Ocimum Tenuiflorum (Tulsi)", "TP": 56, "TN": 2666, "FP": 14, "FN": 22},
    {"plant": "Piper Betle (Betel)", "TP": 55, "TN": 2675, "FP": 11, "FN": 17},
    {"plant": "Plectranthus Amboinicus (Mexican Mint)", "TP": 63, "TN": 2677, "FP": 9, "FN": 9},
    {"plant": "Pongamia Pinnata (Indian Beech)", "TP": 80, "TN": 2652, "FP": 14, "FN": 12},
    {"plant": "Psidium Guajava (Guava)", "TP": 98, "TN": 2656, "FP": 4, "FN": 0},
    {"plant": "Punica Granatum (Pomegranate)", "TP": 95, "TN": 2624, "FP": 15, "FN": 24},
    {"plant": "Santalum Album (Sandalwood)", "TP": 77, "TN": 2652, "FP": 19, "FN": 10},
    {"plant": "Syzygium Cumini (Jamun)", "TP": 55, "TN": 2693, "FP": 6, "FN": 4},
    {"plant": "Syzygium Jambos (Rose Apple)", "TP": 77, "TN": 2667, "FP": 7, "FN": 7},
    {"plant": "Tabernaemontana Divaricata (Crape Jasmine)", "TP": 73, "TN": 2666, "FP": 8, "FN": 11},
    {"plant": "Trigonella Foenum-graecum (Fenugreek)", "TP": 51, "TN": 2701, "FP": 3, "FN": 3}
]

# Calculate precision and recall for each plant class
precision_values = []
recall_values = []

for data in plant_data:
    TP = data["TP"]
    FP = data["FP"]
    FN = data["FN"]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    precision_values.append(precision)
    recall_values.append(recall)

# Sorting the precision and recall values
sorted_indices = np.argsort(recall_values)
sorted_recall = np.array(recall_values)[sorted_indices]
sorted_precision = np.array(precision_values)[sorted_indices]

# Plotting precision-recall curve for each plant class
plt.figure(figsize=(10, 8))

plt.plot(sorted_recall, sorted_precision, marker='o')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# In[91]:


import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Data provided for TP, TN, FP, FN for each plant class
plant_data = [
    {"plant": "Alpinia Galanga (Rasna)", "TP": 65, "TN": 2676, "FP": 7, "FN": 10},
    {"plant": "Amaranthus Viridis (Arive-Dantu)", "TP": 162, "TN": 2540, "FP": 35, "FN": 21},
    {"plant": "Artocarpus Heterophyllus (Jackfruit)", "TP": 74, "TN": 2671, "FP": 3, "FN": 10},
    {"plant": "Azadirachta Indica (Neem)", "TP": 78, "TN": 2659, "FP": 9, "FN": 12},
    {"plant": "Basella Alba (Basale)", "TP": 125, "TN": 2573, "FP": 30, "FN": 30},
    {"plant": "Brassica Juncea (Indian Mustard)", "TP": 37, "TN": 2704, "FP": 3, "FN": 14},
    {"plant": "Carissa Carandas (Karanda)", "TP": 98, "TN": 2634, "FP": 13, "FN": 13},
    {"plant": "Citrus Limon (Lemon)", "TP": 71, "TN": 2655, "FP": 17, "FN": 15},
    {"plant": "Ficus Auriculata (Roxburgh fig)", "TP": 57, "TN": 2661, "FP": 22, "FN": 18},
    {"plant": "Ficus Religiosa (Peepal Tree)", "TP": 91, "TN": 2644, "FP": 19, "FN": 4},
    {"plant": "Hibiscus Rosa-sinensis", "TP": 48, "TN": 2682, "FP": 11, "FN": 17},
    {"plant": "Jasminum (Jasmine)", "TP": 96, "TN": 2634, "FP": 17, "FN": 11},
    {"plant": "Mangifera Indica (Mango)", "TP": 71, "TN": 2639, "FP": 26, "FN": 22},
    {"plant": "Mentha (Mint)", "TP": 129, "TN": 2588, "FP": 24, "FN": 17},
    {"plant": "Moringa Oleifera (Drumstick)", "TP": 94, "TN": 2629, "FP": 13, "FN": 22},
    {"plant": "Muntingia Calabura (Jamaica Cherry-Gasagase)", "TP": 74, "TN": 2665, "FP": 9, "FN": 10},
    {"plant": "Murraya Koenigii (Curry)", "TP": 80, "TN": 2656, "FP": 12, "FN": 10},
    {"plant": "Nerium Oleander (Oleander)", "TP": 66, "TN": 2639, "FP": 26, "FN": 27},
    {"plant": "Nyctanthes Arbor-tristis (Parijata)", "TP": 51, "TN": 2693, "FP": 5, "FN": 9},
    {"plant": "Ocimum Tenuiflorum (Tulsi)", "TP": 56, "TN": 2666, "FP": 14, "FN": 22},
    {"plant": "Piper Betle (Betel)", "TP": 55, "TN": 2675, "FP": 11, "FN": 17},
    {"plant": "Plectranthus Amboinicus (Mexican Mint)", "TP": 63, "TN": 2677, "FP": 9, "FN": 9},
    {"plant": "Pongamia Pinnata (Indian Beech)", "TP": 80, "TN": 2652, "FP": 14, "FN": 12},
    {"plant": "Psidium Guajava (Guava)", "TP": 98, "TN": 2656, "FP": 4, "FN": 0},
    {"plant": "Punica Granatum (Pomegranate)", "TP": 95, "TN": 2624, "FP": 15, "FN": 24},
    {"plant": "Santalum Album (Sandalwood)", "TP": 77, "TN": 2652, "FP": 19, "FN": 10},
    {"plant": "Syzygium Cumini (Jamun)", "TP": 55, "TN": 2693, "FP": 6, "FN": 4},
    {"plant": "Syzygium Jambos (Rose Apple)", "TP": 77, "TN": 2667, "FP": 7, "FN": 7},
    {"plant": "Tabernaemontana Divaricata (Crape Jasmine)", "TP": 73, "TN": 2666, "FP": 8, "FN": 11},
    {"plant": "Trigonella Foenum-graecum (Fenugreek)", "TP": 51, "TN": 2701, "FP": 3, "FN": 3}
]

# Extracting data for plotting
TP_values = [data["TP"] for data in plant_data]
TN_values = [data["TN"] for data in plant_data]
FP_values = [data["FP"] for data in plant_data]
FN_values = [data["FN"] for data in plant_data]

# Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
TPR_values = np.array(TP_values) / (np.array(TP_values) + np.array(FN_values))
FPR_values = np.array(FP_values) / (np.array(FP_values) + np.array(TN_values))

# Calculate ROC curve
fpr, tpr, _ = roc_curve([1] * len(TPR_values), TPR_values)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[ ]:





# # RANDOM FOREST WITH GLCM

# In[ ]:





# In[92]:


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np

# Load NCA features from CSV
nca_glcm_train_csv_path = 'NCA_glcm_train_features.csv'
nca_glcm_test_csv_path = 'NCA_glcm_test_features.csv'

# Load NCA features from CSV
nca_glcm_train_df = pd.read_csv(nca_glcm_train_csv_path)
nca_glcm_test_df = pd.read_csv(nca_glcm_test_csv_path)

# Extract labels and features from DataFrames
y_train = nca_glcm_train_df['Plant']
X_train = nca_glcm_train_df.drop(columns=['Plant'])

y_test = nca_glcm_test_df['Plant']
X_test = nca_glcm_test_df.drop(columns=['Plant'])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode string labels to numerical labels
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize StratifiedKFold with n_splits=5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
computation_times = []

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train_encoded), 1):
    # Split data into train and test sets for this fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train_encoded[train_index], y_train_encoded[test_index]
    
    # Initialize Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    
    # Train the classifier
    start_time = time.time()
    rf_classifier.fit(X_train_fold, y_train_fold)
    end_time = time.time()
    
    # Predict labels on the test set
    predictions_fold = rf_classifier.predict(X_test_fold)
    
    # Calculate metrics for the fold
    accuracy_fold = accuracy_score(y_test_fold, predictions_fold) * 100
    precision_fold = precision_score(y_test_fold, predictions_fold, average='macro')
    recall_fold = recall_score(y_test_fold, predictions_fold, average='macro')
    f1_fold = f1_score(y_test_fold, predictions_fold, average='macro')
    computation_time = end_time - start_time
    
    # Append metrics to lists
    accuracy_scores.append(accuracy_fold)
    precision_scores.append(precision_fold)
    recall_scores.append(recall_fold)
    f1_scores.append(f1_fold)
    computation_times.append(computation_time)
    
    # Print metrics for the fold
    print(f"\nFold {fold}:")
    print(f'Accuracy: {accuracy_fold:.2f}%')
    print(f'Precision: {precision_fold:.2f}')
    print(f'Recall: {recall_fold:.2f}')
    print(f'F1 Score: {f1_fold:.2f}')
    print(f'Computation Time: {computation_time:.4f} seconds')

# Calculate mean and standard deviation of metrics
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
std_precision = np.std(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
std_recall = np.std(recall_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
std_f1 = np.std(f1_scores)
mean_computation_time = sum(computation_times) / len(computation_times)

# Print mean and standard deviation of metrics
print("\nMean and Standard Deviation of Metrics:")
print(f'Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}')
print(f'Mean Precision: {mean_precision:.2f} ± {std_precision:.2f}')
print(f'Mean Recall: {mean_recall:.2f} ± {std_recall:.2f}')
print(f'Mean F1 Score: {mean_f1:.2f} ± {std_f1:.2f}')
print(f'Mean Computation Time: {mean_computation_time:.4f} seconds')


# In[ ]:





# In[93]:


import matplotlib.pyplot as plt
import numpy as np

# Calculate computational time per fold
computation_times_per_fold = np.array(computation_times)

# Plot computational time per fold
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(computation_times_per_fold) + 1), computation_times_per_fold, marker='o', color='b')
plt.title('Computational Time per Fold')
plt.xlabel('Fold')
plt.ylabel('Computational Time (seconds)')
plt.grid(True)
plt.show()


# In[ ]:





# In[94]:


import matplotlib.pyplot as plt
import numpy as np

# Metrics data
metrics_data = {
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores
}

# Create a box plot for each metric
plt.figure(figsize=(10, 6))
plt.boxplot(metrics_data.values(), labels=metrics_data.keys())
plt.title('Distribution of Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.grid(True)
plt.show()


# In[ ]:





# In[95]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

# Load NCA GLCM features from CSV for the training set
nca_glcm_train_df = pd.read_csv('NCA_glcm_train_features.csv')

# Assuming your CSV file contains features and labels columns
features_train = nca_glcm_train_df.drop(columns=['Plant'])  # Update 'label' with the actual column name
labels_train = nca_glcm_train_df['Plant']  # Update 'label' with the actual column name

# Define the parameter distributions for randomized search
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(10, 110),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier()

# Initialize StratifiedKFold
stratkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize RandomizedSearchCV
randomized_search = RandomizedSearchCV(
    rf_classifier,
    param_distributions=param_dist,
    n_iter=100,  # Adjust the number of iterations as needed
    scoring='accuracy',  # Use accuracy as the scoring metric
    cv=stratkf,  # Use StratifiedKFold for cross-validation
    verbose=2,  # Increase verbosity for detailed output
    n_jobs=-1  # Use all available CPU cores
)

# Fit the randomized search to the data
randomized_search.fit(features_train, labels_train)

# Print the best parameters and corresponding accuracy
best_params = randomized_search.best_params_
best_accuracy = randomized_search.best_score_
print("Best Hyperparameters:")
print(best_params)
print(f"Best Accuracy: {best_accuracy:.2f}")


# In[ ]:





# In[96]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Load NCA GLCM features from CSV for the training set
nca_glcm_train_df = pd.read_csv('NCA_glcm_train_features.csv')

# Assuming your CSV file contains features and labels columns
features_train = nca_glcm_train_df.drop(columns=['Plant'])  # Update 'label' with the actual column name
labels_train = nca_glcm_train_df['Plant']  # Update 'label' with the actual column name

# Encode labels to integers
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)

# Initialize Random Forest classifier with the best hyperparameters
rf_classifier = RandomForestClassifier(
    bootstrap=False,
    max_depth=81,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=11,
    n_estimators=122
)

# Train the model
start_time = time.time()
rf_classifier.fit(features_train, labels_train_encoded)
end_time = time.time()

# Calculate predictions
predictions_train = rf_classifier.predict(features_train)

# Calculate metrics
accuracy = accuracy_score(labels_train_encoded, predictions_train) * 100
precision = precision_score(labels_train_encoded, predictions_train, average='weighted')
recall = recall_score(labels_train_encoded, predictions_train, average='weighted')
f1 = f1_score(labels_train_encoded, predictions_train, average='weighted')
computation_time = end_time - start_time

# Print metrics
print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Computational Time: {computation_time:.4f} seconds')


# In[ ]:





# In[98]:


from joblib import dump

# Save the trained model to a file
dump(rf_classifier, 'random_forest_model.pkl')
print("Model saved successfully.")


# In[ ]:





# In[99]:


import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load

# Load the saved Random Forest model
rf_model = load('random_forest_model.pkl')

# Load the testing dataset
test_df = pd.read_csv('NCA_glcm_test_features.csv')

# Extract features and labels
X_test = test_df.drop(columns=['Plant'])  # Assuming 'Plant' is the label column
y_test = test_df['Plant']  # Assuming 'Plant' is the label column

# Encode labels to integers if they are not already encoded
# Assuming label_encoder is the instance used for encoding during training
y_test_encoded = label_encoder.transform(y_test)

# Make predictions on the testing dataset
y_pred = rf_model.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap for Random Forest')
plt.show()


# In[ ]:





# In[100]:


from sklearn.metrics import classification_report

# Make predictions
y_pred = rf_model.predict(X_test)

# Generate classification report
report = classification_report(y_test_encoded, y_pred)

# Print classification report
print("Classification Report for Random Forest:")
print(report)


# In[ ]:





# In[101]:


import pandas as pd

# Initialize lists to store data
plant_names = []
tp_list = []
tn_list = []
fp_list = []
fn_list = []

# Iterate over each class
for class_label in range(len(label_encoder.classes_)):
    plant_name = label_encoder.inverse_transform([class_label])[0]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    # Iterate over each sample
    for true_label, pred_label in zip(y_test_encoded, y_pred):
        # Check if the sample belongs to the current class
        if true_label == class_label:
            # Check if the prediction is correct (TP)
            if pred_label == true_label:
                tp += 1
            else:
                fn += 1  # Incorrect prediction (FN)
        else:
            # Check if the prediction is correct (TN)
            if pred_label != class_label:
                tn += 1
            else:
                fp += 1  # Incorrect prediction (FP)
    
    # Append data to lists
    plant_names.append(plant_name)
    tp_list.append(tp)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)

# Create a DataFrame
results_df = pd.DataFrame({
    'Plant': plant_names,
    'TP': tp_list,
    'TN': tn_list,
    'FP': fp_list,
    'FN': fn_list
})

# Print the DataFrame
print("TP, TN, FP, FN for each plant (Random Forest):")
print(results_df)


# In[ ]:





# In[102]:


import pandas as pd

# Define functions to calculate metrics
def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def f1_score(prec, rec):
    return 2 * (prec * rec) / (prec + rec)

# Initialize lists to store metrics for each plant
plant_names = results_df['Plant']
acc_list = []
prec_list = []
rec_list = []
f1_list = []

# Calculate metrics for each plant
for index, row in results_df.iterrows():
    tp = row['TP']
    tn = row['TN']
    fp = row['FP']
    fn = row['FN']
    
    acc = accuracy(tp, tn, fp, fn)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)
    
    acc_list.append(acc)
    prec_list.append(prec)
    rec_list.append(rec)
    f1_list.append(f1)

# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame({
    'Plant': plant_names,
    'Accuracy': acc_list,
    'Precision': prec_list,
    'Recall': rec_list,
    'F1 Score': f1_list
})

# Print the DataFrame
print("Metrics for each plant (Random Forest):")
print(metrics_df)


# In[ ]:





# In[103]:


import numpy as np

# Extract metrics from the DataFrame
accuracy_values = metrics_df['Accuracy']
precision_values = metrics_df['Precision']
recall_values = metrics_df['Recall']
f1_values = metrics_df['F1 Score']

# Calculate average metrics
average_accuracy = np.mean(accuracy_values) * 100  # Convert to percentage
average_precision = np.mean(precision_values) * 100  # Convert to percentage
average_recall = np.mean(recall_values) * 100  # Convert to percentage
average_f1_score = np.mean(f1_values) * 100  # Convert to percentage

# Assuming uniform time for each plant calculation
# Replace 'time_per_plant' with the actual time taken for each plant's calculation
time_per_plant = 10  # Placeholder value (replace with actual time in seconds)
number_of_plants = len(metrics_df)
average_computational_time = time_per_plant / number_of_plants

# Print the results
print("Average Metrics for Random Forest Model:")
print("Average Accuracy: {:.2f}%".format(average_accuracy))
print("Average Precision: {:.2f}%".format(average_precision))
print("Average Recall: {:.2f}%".format(average_recall))
print("Average F1 Score: {:.2f}%".format(average_f1_score))
print("Average Computational Time: {:.2f} seconds".format(average_computational_time))


# In[ ]:





# In[104]:


# Data provided for TP, TN, FP, FN for each plant class (Random Forest)
plant_data_rf = [
    {"plant": "Alpinia Galanga (Rasna)", "TP": 58, "TN": 2673, "FP": 10, "FN": 17},
    {"plant": "Amaranthus Viridis (Arive-Dantu)", "TP": 155, "TN": 2539, "FP": 36, "FN": 28},
    {"plant": "Artocarpus Heterophyllus (Jackfruit)", "TP": 73, "TN": 2668, "FP": 6, "FN": 11},
    {"plant": "Azadirachta Indica (Neem)", "TP": 65, "TN": 2657, "FP": 11, "FN": 25},
    {"plant": "Basella Alba (Basale)", "TP": 117, "TN": 2564, "FP": 39, "FN": 38},
    {"plant": "Brassica Juncea (Indian Mustard)", "TP": 39, "TN": 2704, "FP": 3, "FN": 12},
    {"plant": "Carissa Carandas (Karanda)", "TP": 100, "TN": 2625, "FP": 22, "FN": 11},
    {"plant": "Citrus Limon (Lemon)", "TP": 62, "TN": 2658, "FP": 14, "FN": 24},
    {"plant": "Ficus Auriculata (Roxburgh fig)", "TP": 50, "TN": 2666, "FP": 17, "FN": 25},
    {"plant": "Ficus Religiosa (Peepal Tree)", "TP": 91, "TN": 2642, "FP": 21, "FN": 4},
    {"plant": "Hibiscus Rosa-sinensis", "TP": 41, "TN": 2679, "FP": 14, "FN": 24},
    {"plant": "Jasminum (Jasmine)", "TP": 95, "TN": 2635, "FP": 16, "FN": 12},
    {"plant": "Mangifera Indica (Mango)", "TP": 71, "TN": 2641, "FP": 24, "FN": 22},
    {"plant": "Mentha (Mint)", "TP": 127, "TN": 2583, "FP": 29, "FN": 19},
    {"plant": "Moringa Oleifera (Drumstick)", "TP": 101, "TN": 2623, "FP": 19, "FN": 15},
    {"plant": "Muntingia Calabura (Jamaica Cherry-Gasagase)", "TP": 70, "TN": 2664, "FP": 10, "FN": 14},
    {"plant": "Murraya Koenigii (Curry)", "TP": 72, "TN": 2656, "FP": 12, "FN": 18},
    {"plant": "Nerium Oleander (Oleander)", "TP": 68, "TN": 2635, "FP": 30, "FN": 25},
    {"plant": "Nyctanthes Arbor-tristis (Parijata)", "TP": 54, "TN": 2690, "FP": 8, "FN": 6},
    {"plant": "Ocimum Tenuiflorum (Tulsi)", "TP": 52, "TN": 2670, "FP": 10, "FN": 26},
    {"plant": "Piper Betle (Betel)", "TP": 53, "TN": 2674, "FP": 12, "FN": 19},
    {"plant": "Plectranthus Amboinicus (Mexican Mint)", "TP": 65, "TN": 2670, "FP": 16, "FN": 7},
    {"plant": "Pongamia Pinnata (Indian Beech)", "TP": 78, "TN": 2648, "FP": 18, "FN": 14},
    {"plant": "Psidium Guajava (Guava)", "TP": 98, "TN": 2654, "FP": 6, "FN": 0},
    {"plant": "Punica Granatum (Pomegranate)", "TP": 95, "TN": 2624, "FP": 15, "FN": 24},
    {"plant": "Santalum Album (Sandalwood)", "TP": 81, "TN": 2648, "FP": 23, "FN": 6},
    {"plant": "Syzygium Cumini (Jamun)", "TP": 52, "TN": 2688, "FP": 11, "FN": 7},
    {"plant": "Syzygium Jambos (Rose Apple)", "TP": 78, "TN": 2665, "FP": 9, "FN": 6},
    {"plant": "Tabernaemontana Divaricata (Crape Jasmine)", "TP": 71, "TN": 2664, "FP": 10, "FN": 13},
    {"plant": "Trigonella Foenum-graecum (Fenugreek)", "TP": 51, "TN": 2700, "FP": 4, "FN": 3}
]

# Extracting data for plotting
plants_rf = [data["plant"] for data in plant_data_rf]
TP_values_rf = [data["TP"] for data in plant_data_rf]
TN_values_rf = [data["TN"] for data in plant_data_rf]
FP_values_rf = [data["FP"] for data in plant_data_rf]
FN_values_rf = [data["FN"] for data in plant_data_rf]

# Bar chart for TP
plt.figure(figsize=(12, 6))
plt.bar(plants_rf, TP_values_rf, color='b', label='TP')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('True Positive (TP) for Each Plant Class (Random Forest)')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart for TN
plt.figure(figsize=(12, 6))
plt.bar(plants_rf, TN_values_rf, color='g', label='TN')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('True Negative (TN) for Each Plant Class (Random Forest)')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart for FP
plt.figure(figsize=(12, 6))
plt.bar(plants_rf, FP_values_rf, color='r', label='FP')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('False Positive (FP) for Each Plant Class (Random Forest)')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart for FN
plt.figure(figsize=(12, 6))
plt.bar(plants_rf, FN_values_rf, color='orange', label='FN')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('False Negative (FN) for Each Plant Class (Random Forest)')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[105]:


# Data provided for TP, TN, FP, FN for each plant class (Random Forest)
plant_data_rf = [
    {"plant": "Alpinia Galanga (Rasna)", "TP": 58, "TN": 2673, "FP": 10, "FN": 17},
    {"plant": "Amaranthus Viridis (Arive-Dantu)", "TP": 155, "TN": 2539, "FP": 36, "FN": 28},
    {"plant": "Artocarpus Heterophyllus (Jackfruit)", "TP": 73, "TN": 2668, "FP": 6, "FN": 11},
    {"plant": "Azadirachta Indica (Neem)", "TP": 65, "TN": 2657, "FP": 11, "FN": 25},
    {"plant": "Basella Alba (Basale)", "TP": 117, "TN": 2564, "FP": 39, "FN": 38},
    {"plant": "Brassica Juncea (Indian Mustard)", "TP": 39, "TN": 2704, "FP": 3, "FN": 12},
    {"plant": "Carissa Carandas (Karanda)", "TP": 100, "TN": 2625, "FP": 22, "FN": 11},
    {"plant": "Citrus Limon (Lemon)", "TP": 62, "TN": 2658, "FP": 14, "FN": 24},
    {"plant": "Ficus Auriculata (Roxburgh fig)", "TP": 50, "TN": 2666, "FP": 17, "FN": 25},
    {"plant": "Ficus Religiosa (Peepal Tree)", "TP": 91, "TN": 2642, "FP": 21, "FN": 4},
    {"plant": "Hibiscus Rosa-sinensis", "TP": 41, "TN": 2679, "FP": 14, "FN": 24},
    {"plant": "Jasminum (Jasmine)", "TP": 95, "TN": 2635, "FP": 16, "FN": 12},
    {"plant": "Mangifera Indica (Mango)", "TP": 71, "TN": 2641, "FP": 24, "FN": 22},
    {"plant": "Mentha (Mint)", "TP": 127, "TN": 2583, "FP": 29, "FN": 19},
    {"plant": "Moringa Oleifera (Drumstick)", "TP": 101, "TN": 2623, "FP": 19, "FN": 15},
    {"plant": "Muntingia Calabura (Jamaica Cherry-Gasagase)", "TP": 70, "TN": 2664, "FP": 10, "FN": 14},
    {"plant": "Murraya Koenigii (Curry)", "TP": 72, "TN": 2656, "FP": 12, "FN": 18},
    {"plant": "Nerium Oleander (Oleander)", "TP": 68, "TN": 2635, "FP": 30, "FN": 25},
    {"plant": "Nyctanthes Arbor-tristis (Parijata)", "TP": 54, "TN": 2690, "FP": 8, "FN": 6},
    {"plant": "Ocimum Tenuiflorum (Tulsi)", "TP": 52, "TN": 2670, "FP": 10, "FN": 26},
    {"plant": "Piper Betle (Betel)", "TP": 53, "TN": 2674, "FP": 12, "FN": 19},
    {"plant": "Plectranthus Amboinicus (Mexican Mint)", "TP": 65, "TN": 2670, "FP": 16, "FN": 7},
    {"plant": "Pongamia Pinnata (Indian Beech)", "TP": 78, "TN": 2648, "FP": 18, "FN": 14},
    {"plant": "Psidium Guajava (Guava)", "TP": 98, "TN": 2654, "FP": 6, "FN": 0},
    {"plant": "Punica Granatum (Pomegranate)", "TP": 95, "TN": 2624, "FP": 15, "FN": 24},
    {"plant": "Santalum Album (Sandalwood)", "TP": 81, "TN": 2648, "FP": 23, "FN": 6},
    {"plant": "Syzygium Cumini (Jamun)", "TP": 52, "TN": 2688, "FP": 11, "FN": 7},
    {"plant": "Syzygium Jambos (Rose Apple)", "TP": 78, "TN": 2665, "FP": 9, "FN": 6},
    {"plant": "Tabernaemontana Divaricata (Crape Jasmine)", "TP": 71, "TN": 2664, "FP": 10, "FN": 13},
    {"plant": "Trigonella Foenum-graecum (Fenugreek)", "TP": 51, "TN": 2700, "FP": 4, "FN": 3}
]

# Extracting data for plotting
plants_rf = [data["plant"] for data in plant_data_rf]
TP_values_rf = [data["TP"] for data in plant_data_rf]
TN_values_rf = [data["TN"] for data in plant_data_rf]
FP_values_rf = [data["FP"] for data in plant_data_rf]
FN_values_rf = [data["FN"] for data in plant_data_rf]

# Bar chart for TP, TN, FP, FN for each plant
bar_width = 0.2
index = np.arange(len(plants_rf))

plt.figure(figsize=(14, 8))

plt.bar(index - bar_width, TP_values_rf, bar_width, color='b', label='TP')
plt.bar(index, TN_values_rf, bar_width, color='g', label='TN')
plt.bar(index + bar_width, FP_values_rf, bar_width, color='r', label='FP')
plt.bar(index + 2 * bar_width, FN_values_rf, bar_width, color='orange', label='FN')

plt.xlabel('Plant Serial Number')
plt.ylabel('Count')
plt.title('TP, TN, FP, FN for Each Plant Class (Random Forest)')
plt.xticks(index, plants_rf, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[110]:


import matplotlib.pyplot as plt

# Define data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values_rf = [98.85, 83.33, 82.49, 82.69]  # Replace with your values for the Random Forest model
computational_time_rf = 0.33  # Replace with your computational time for the Random Forest model in seconds

# Plot line chart
plt.figure(figsize=(10, 6))

# Plot metrics for Random Forest model
plt.plot(metrics, values_rf, marker='o', label='Metrics (RF)', color='blue')

# Plot computational time for Random Forest model
plt.axhline(y=computational_time_rf, linestyle='--', label='Computational Time (RF)', color='red')

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Performance Metrics and Computational Time for Random Forest Model')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# # RANDOM FOREST WITH HOG 

# In[ ]:





# In[113]:


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np

# Load NCA features from CSV
nca_hog_train_csv_path = 'NCA_hog_train_features.csv'
nca_hog_test_csv_path = 'NCA_hog_test_features.csv'

# Load NCA features from CSV
nca_hog_train_df = pd.read_csv(nca_hog_train_csv_path)
nca_hog_test_df = pd.read_csv(nca_hog_test_csv_path)

# Extract labels and features from DataFrames
y_train = nca_hog_train_df['label']
X_train = nca_hog_train_df.drop(columns=['label'])

y_test = nca_hog_test_df['label']
X_test = nca_hog_test_df.drop(columns=['label'])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode string labels to numerical labels
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize StratifiedKFold with n_splits=5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
computation_times = []

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train_encoded), 1):
    # Split data into train and test sets for this fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train_encoded[train_index], y_train_encoded[test_index]
    
    # Initialize Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    
    # Train the classifier
    start_time = time.time()
    rf_classifier.fit(X_train_fold, y_train_fold)
    end_time = time.time()
    
    # Predict labels on the test set
    predictions_fold = rf_classifier.predict(X_test_fold)
    
    # Calculate metrics for the fold
    accuracy_fold = accuracy_score(y_test_fold, predictions_fold) * 100
    precision_fold = precision_score(y_test_fold, predictions_fold, average='macro')
    recall_fold = recall_score(y_test_fold, predictions_fold, average='macro')
    f1_fold = f1_score(y_test_fold, predictions_fold, average='macro')
    computation_time = end_time - start_time
    
    # Append metrics to lists
    accuracy_scores.append(accuracy_fold)
    precision_scores.append(precision_fold)
    recall_scores.append(recall_fold)
    f1_scores.append(f1_fold)
    computation_times.append(computation_time)
    
    # Print metrics for the fold
    print(f"\nFold {fold}:")
    print(f'Accuracy: {accuracy_fold:.2f}%')
    print(f'Precision: {precision_fold:.2f}')
    print(f'Recall: {recall_fold:.2f}')
    print(f'F1 Score: {f1_fold:.2f}')
    print(f'Computation Time: {computation_time:.4f} seconds')

# Calculate mean and standard deviation of metrics
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
std_precision = np.std(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
std_recall = np.std(recall_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
std_f1 = np.std(f1_scores)
mean_computation_time = sum(computation_times) / len(computation_times)

# Print mean and standard deviation of metrics
print("\nMean and Standard Deviation of Metrics:")
print(f'Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}')
print(f'Mean Precision: {mean_precision:.2f} ± {std_precision:.2f}')
print(f'Mean Recall: {mean_recall:.2f} ± {std_recall:.2f}')
print(f'Mean F1 Score: {mean_f1:.2f} ± {std_f1:.2f}')
print(f'Mean Computation Time: {mean_computation_time:.4f} seconds')


# In[ ]:





# In[114]:


import matplotlib.pyplot as plt
import numpy as np

# Calculate computational time per fold
computation_times_per_fold = np.array(computation_times)

# Plot computational time per fold
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(computation_times_per_fold) + 1), computation_times_per_fold, marker='o', color='b')
plt.title('Computational Time per Fold')
plt.xlabel('Fold')
plt.ylabel('Computational Time (seconds)')
plt.grid(True)
plt.show()


# In[ ]:





# In[115]:


import matplotlib.pyplot as plt
import numpy as np

# Metrics data
metrics_data = {
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores
}

# Create a box plot for each metric
plt.figure(figsize=(10, 6))
plt.boxplot(metrics_data.values(), labels=metrics_data.keys())
plt.title('Distribution of Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.grid(True)
plt.show()


# In[ ]:





# In[116]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

# Load HOG features from CSV for the training set
nca_hog_train_df = pd.read_csv('NCA_hog_train_features.csv')

# Assuming your CSV file contains features and labels columns
features_train = nca_hog_train_df.drop(columns=['label'])  # Update 'Plant' with the actual column name
labels_train = nca_hog_train_df['label']  # Update 'Plant' with the actual column name

# Define the parameter distributions for randomized search
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(10, 110),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier()

# Initialize StratifiedKFold
stratkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize RandomizedSearchCV
randomized_search = RandomizedSearchCV(
    rf_classifier,
    param_distributions=param_dist,
    n_iter=100,  # Adjust the number of iterations as needed
    scoring='accuracy',  # Use accuracy as the scoring metric
    cv=stratkf,  # Use StratifiedKFold for cross-validation
    verbose=2,  # Increase verbosity for detailed output
    n_jobs=-1  # Use all available CPU cores
)

# Fit the randomized search to the data
randomized_search.fit(features_train, labels_train)

# Print the best parameters and corresponding accuracy
best_params = randomized_search.best_params_
best_accuracy = randomized_search.best_score_
print("Best Hyperparameters:")
print(best_params)
print(f"Best Accuracy: {best_accuracy:.2f}")


# In[ ]:





# In[117]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Load NCA GLCM features from CSV for the training set
nca_hog_train_df = pd.read_csv('NCA_hog_train_features.csv')

# Assuming your CSV file contains features and labels columns
features_train = nca_hog_train_df.drop(columns=['label'])  # Update 'label' with the actual column name
labels_train = nca_hog_train_df['label']  # Update 'label' with the actual column name

# Encode labels to integers
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)

# Initialize Random Forest classifier with the best hyperparameters
rf_classifier = RandomForestClassifier(
    bootstrap=False,
    max_depth=89,
    max_features='sqrt',
    min_samples_leaf=3,
    min_samples_split=5,
    n_estimators=500
)

# Train the model
start_time = time.time()
rf_classifier.fit(features_train, labels_train_encoded)
end_time = time.time()

# Calculate predictions
predictions_train = rf_classifier.predict(features_train)

# Calculate metrics
accuracy = accuracy_score(labels_train_encoded, predictions_train) * 100
precision = precision_score(labels_train_encoded, predictions_train, average='weighted')
recall = recall_score(labels_train_encoded, predictions_train, average='weighted')
f1 = f1_score(labels_train_encoded, predictions_train, average='weighted')
computation_time = end_time - start_time

# Print metrics
print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Computational Time: {computation_time:.4f} seconds')


# In[ ]:





# In[118]:


from joblib import dump

# Save the trained model to a file
dump(rf_classifier, 'random_forest_model2.pkl')
print("Model saved successfully.")


# In[ ]:





# In[119]:


import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load

# Load the saved Random Forest model
rf_model = load('random_forest_model2.pkl')

# Load the testing dataset
test_df = pd.read_csv('NCA_hog_test_features.csv')

# Extract features and labels
X_test = test_df.drop(columns=['label'])  # Assuming 'Plant' is the label column
y_test = test_df['label']  # Assuming 'Plant' is the label column

# Encode labels to integers if they are not already encoded
# Assuming label_encoder is the instance used for encoding during training
y_test_encoded = label_encoder.transform(y_test)

# Make predictions on the testing dataset
y_pred = rf_model.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap for Random Forest')
plt.show()


# In[ ]:





# In[120]:


from sklearn.metrics import classification_report

# Make predictions
y_pred = rf_model.predict(X_test)

# Generate classification report
report = classification_report(y_test_encoded, y_pred)

# Print classification report
print("Classification Report for Random Forest:")
print(report)


# In[ ]:





# In[121]:


import pandas as pd

# Initialize lists to store data
plant_names = []
tp_list = []
tn_list = []
fp_list = []
fn_list = []

# Iterate over each class
for class_label in range(len(label_encoder.classes_)):
    plant_name = label_encoder.inverse_transform([class_label])[0]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    # Iterate over each sample
    for true_label, pred_label in zip(y_test_encoded, y_pred):
        # Check if the sample belongs to the current class
        if true_label == class_label:
            # Check if the prediction is correct (TP)
            if pred_label == true_label:
                tp += 1
            else:
                fn += 1  # Incorrect prediction (FN)
        else:
            # Check if the prediction is correct (TN)
            if pred_label != class_label:
                tn += 1
            else:
                fp += 1  # Incorrect prediction (FP)
    
    # Append data to lists
    plant_names.append(plant_name)
    tp_list.append(tp)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)

# Create a DataFrame
results_df = pd.DataFrame({
    'Plant': plant_names,
    'TP': tp_list,
    'TN': tn_list,
    'FP': fp_list,
    'FN': fn_list
})

# Print the DataFrame
print("TP, TN, FP, FN for each plant (Random Forest):")
print(results_df)


# In[ ]:





# In[126]:


import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Start timing
start_time = time.time()

# Make predictions on the testing dataset
y_pred = rf_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test_encoded, y_pred) * 100
precision = precision_score(y_test_encoded, y_pred, average='weighted') * 100
recall = recall_score(y_test_encoded, y_pred, average='weighted') * 100
f1 = f1_score(y_test_encoded, y_pred, average='weighted') * 100

# Calculate computational time
computation_time = time.time() - start_time

# Print metrics
print("Metrics on Testing Dataset:")
print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}%')
print(f'Recall: {recall:.2f}%')
print(f'F1 Score: {f1:.2f}%')
print(f'Computational Time: {computation_time:.4f} seconds')


# In[130]:


import matplotlib.pyplot as plt

# Data for TP, TN, FP, FN for each plant class (Random Forest)
plant_data_rf = [
    {"Plant": "Alpinia Galanga (Rasna)", "TP": 69, "TN": 2678, "FP": 5, "FN": 6},
    {"Plant": "Amaranthus Viridis (Arive-Dantu)", "TP": 158, "TN": 2508, "FP": 67, "FN": 25},
    {"Plant": "Artocarpus Heterophyllus (Jackfruit)", "TP": 72, "TN": 2661, "FP": 13, "FN": 12},
    {"Plant": "Azadirachta Indica (Neem)", "TP": 78, "TN": 2653, "FP": 15, "FN": 12},
    {"Plant": "Basella Alba (Basale)", "TP": 134, "TN": 2551, "FP": 52, "FN": 21},
    {"Plant": "Brassica Juncea (Indian Mustard)", "TP": 42, "TN": 2706, "FP": 1, "FN": 9},
    {"Plant": "Carissa Carandas (Karanda)", "TP": 103, "TN": 2631, "FP": 16, "FN": 8},
    {"Plant": "Citrus Limon (Lemon)", "TP": 57, "TN": 2657, "FP": 15, "FN": 29},
    {"Plant": "Ficus Auriculata (Roxburgh fig)", "TP": 53, "TN": 2679, "FP": 4, "FN": 22},
    {"Plant": "Ficus Religiosa (Peepal Tree)", "TP": 94, "TN": 2659, "FP": 4, "FN": 1},
    {"Plant": "Hibiscus Rosa-sinensis", "TP": 41, "TN": 2686, "FP": 7, "FN": 24},
    {"Plant": "Jasminum (Jasmine)", "TP": 77, "TN": 2626, "FP": 25, "FN": 30},
    {"Plant": "Mangifera Indica (Mango)", "TP": 85, "TN": 2646, "FP": 19, "FN": 8},
    {"Plant": "Mentha (Mint)", "TP": 128, "TN": 2582, "FP": 30, "FN": 18},
    {"Plant": "Moringa Oleifera (Drumstick)", "TP": 98, "TN": 2626, "FP": 16, "FN": 18},
    {"Plant": "Muntingia Calabura (Jamaica Cherry-Gasagase)", "TP": 75, "TN": 2657, "FP": 17, "FN": 9},
    {"Plant": "Murraya Koenigii (Curry)", "TP": 48, "TN": 2659, "FP": 9, "FN": 42},
    {"Plant": "Nerium Oleander (Oleander)", "TP": 77, "TN": 2646, "FP": 19, "FN": 16},
    {"Plant": "Nyctanthes Arbor-tristis (Parijata)", "TP": 54, "TN": 2695, "FP": 3, "FN": 6},
    {"Plant": "Ocimum Tenuiflorum (Tulsi)", "TP": 37, "TN": 2674, "FP": 6, "FN": 41},
    {"Plant": "Piper Betle (Betel)", "TP": 57, "TN": 2680, "FP": 6, "FN": 15},
    {"Plant": "Plectranthus Amboinicus (Mexican Mint)", "TP": 61, "TN": 2664, "FP": 22, "FN": 11},
    {"Plant": "Pongamia Pinnata (Indian Beech)", "TP": 84, "TN": 2649, "FP": 17, "FN": 8},
    {"Plant": "Psidium Guajava (Guava)", "TP": 91, "TN": 2647, "FP": 13, "FN": 7},
    {"Plant": "Punica Granatum (Pomegranate)", "TP": 99, "TN": 2598, "FP": 41, "FN": 20},
    {"Plant": "Santalum Album (Sandalwood)", "TP": 63, "TN": 2640, "FP": 31, "FN": 24},
    {"Plant": "Syzygium Cumini (Jamun)", "TP": 44, "TN": 2691, "FP": 8, "FN": 15},
    {"Plant": "Syzygium Jambos (Rose Apple)", "TP": 67, "TN": 2664, "FP": 10, "FN": 17},
    {"Plant": "Tabernaemontana Divaricata (Crape Jasmine)", "TP": 58, "TN": 2668, "FP": 6, "FN": 26},
    {"Plant": "Trigonella Foenum-graecum (Fenugreek)", "TP": 47, "TN": 2694, "FP": 10, "FN": 7}
]

# Extracting data for plotting
plants_rf = [data["Plant"] for data in plant_data_rf]
TP_values_rf = [data["TP"] for data in plant_data_rf]
TN_values_rf = [data["TN"] for data in plant_data_rf]
FP_values_rf = [data["FP"] for data in plant_data_rf]
FN_values_rf = [data["FN"] for data in plant_data_rf]

# Bar chart for TP
plt.figure(figsize=(12, 6))
plt.bar(plants_rf, TP_values_rf, color='b', label='TP')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('True Positive (TP) for Each Plant Class (Random Forest)')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart for TN
plt.figure(figsize=(12, 6))
plt.bar(plants_rf, TN_values_rf, color='g', label='TN')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('True Negative (TN) for Each Plant Class (Random Forest)')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart for FP
plt.figure(figsize=(12, 6))
plt.bar(plants_rf, FP_values_rf, color='r', label='FP')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('False Positive (FP) for Each Plant Class (Random Forest)')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart for FN
plt.figure(figsize=(12, 6))
plt.bar(plants_rf, FN_values_rf, color='orange', label='FN')
plt.xlabel('Plant Class')
plt.ylabel('Count')
plt.title('False Negative (FN) for Each Plant Class (Random Forest)')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[132]:


import matplotlib.pyplot as plt
import pandas as pd

# Load the testing dataset
test_df = pd.read_csv('NCA_hog_test_features.csv')

# Extract features and labels
X_test = test_df.drop(columns=['label'])  # Assuming 'label' is the label column
y_test = test_df['label']  # Assuming 'label' is the label column

# Make predictions on the testing dataset using the Random Forest model
y_pred = rf_model.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred)

# Extract TP, TN, FP, FN values from the confusion matrix
tp_values = conf_matrix.diagonal()
tn_values = np.sum(conf_matrix) - np.sum(conf_matrix, axis=0) - np.sum(conf_matrix, axis=1) + tp_values
fp_values = np.sum(conf_matrix, axis=1) - tp_values
fn_values = np.sum(conf_matrix, axis=0) - tp_values

# Plot bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.2
index = range(1, len(unique_labels) + 1)

plt.bar(index, tp_values, bar_width, label='True Positives')
plt.bar([i + bar_width for i in index], tn_values, bar_width, label='True Negatives')
plt.bar([i + 2*bar_width for i in index], fp_values, bar_width, label='False Positives')
plt.bar([i + 3*bar_width for i in index], fn_values, bar_width, label='False Negatives')

plt.xlabel('Serial Numbers of Plants')
plt.ylabel('Count')
plt.title('Metrics by Plant Serial Number')
plt.xticks([i + 1.5*bar_width for i in index], index)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[134]:


import matplotlib.pyplot as plt

# Define data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values_rf = [81.62, 82.35, 81.62, 81.37]  # Replace with your values for the Random Forest model
computational_time_rf = 0.2233  # Replace with your computational time for the Random Forest model in seconds

# Plot line chart
plt.figure(figsize=(10, 6))

# Plot metrics for Random Forest model
plt.plot(metrics, values_rf, marker='o', label='Metrics (RF)', color='blue')

# Plot computational time for Random Forest model
plt.axhline(y=computational_time_rf, linestyle='--', label='Computational Time (RF)', color='red')

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Performance Metrics and Computational Time for Random Forest Model')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# # SVM WITH GLCM

# In[137]:


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import time

# Load NCA features from CSV
nca_glcm_train_csv_path = 'NCA_glcm_train_features.csv'
nca_glcm_test_csv_path = 'NCA_glcm_test_features.csv'

# Load NCA features from CSV
nca_glcm_train_df = pd.read_csv(nca_glcm_train_csv_path)
nca_glcm_test_df = pd.read_csv(nca_glcm_test_csv_path)

# Extract labels and features from DataFrames
y_train = nca_glcm_train_df['Plant']
X_train = nca_glcm_train_df.drop(columns=['Plant'])

y_test = nca_glcm_test_df['Plant']
X_test = nca_glcm_test_df.drop(columns=['Plant'])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode string labels to numerical labels
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize StratifiedKFold with n_splits=5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
computation_times = []

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train_encoded), 1):
    # Split data into train and test sets for this fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train_encoded[train_index], y_train_encoded[test_index]
    
    # Initialize Support Vector Classifier (SVM)
    svm_classifier = SVC(kernel='linear')
    
    # Train the classifier
    start_time = time.time()
    svm_classifier.fit(X_train_fold, y_train_fold)
    end_time = time.time()
    
    # Predict labels on the test set
    predictions_fold = svm_classifier.predict(X_test_fold)
    
    # Calculate metrics for the fold
    accuracy_fold = accuracy_score(y_test_fold, predictions_fold) * 100
    precision_fold = precision_score(y_test_fold, predictions_fold, average='macro')
    recall_fold = recall_score(y_test_fold, predictions_fold, average='macro')
    f1_fold = f1_score(y_test_fold, predictions_fold, average='macro')
    computation_time = end_time - start_time
    
    # Append metrics to lists
    accuracy_scores.append(accuracy_fold)
    precision_scores.append(precision_fold)
    recall_scores.append(recall_fold)
    f1_scores.append(f1_fold)
    computation_times.append(computation_time)
    
    # Print metrics for the fold
    print(f"\nFold {fold}:")
    print(f'Accuracy: {accuracy_fold:.2f}%')
    print(f'Precision: {precision_fold:.2f}')
    print(f'Recall: {recall_fold:.2f}')
    print(f'F1 Score: {f1_fold:.2f}')
    print(f'Computation Time: {computation_time:.4f} seconds')

# Calculate mean and standard deviation of metrics
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
std_precision = np.std(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
std_recall = np.std(recall_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
std_f1 = np.std(f1_scores)
mean_computation_time = sum(computation_times) / len(computation_times)

# Print mean and standard deviation of metrics
print("\nMean and Standard Deviation of Metrics:")
print(f'Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}')
print(f'Mean Precision: {mean_precision:.2f} ± {std_precision:.2f}')
print(f'Mean Recall: {mean_recall:.2f} ± {std_recall:.2f}')
print(f'Mean F1 Score: {mean_f1:.2f} ± {std_f1:.2f}')
print(f'Mean Computation Time: {mean_computation_time:.4f} seconds')


# In[ ]:





# In[138]:


import matplotlib.pyplot as plt

# Calculate computational time per fold
computation_times_per_fold = np.array(computation_times)

# Plot computational time per fold
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(computation_times_per_fold) + 1), computation_times_per_fold, marker='o', color='b')
plt.title('Computational Time per Fold')
plt.xlabel('Fold')
plt.ylabel('Computational Time (seconds)')
plt.grid(True)
plt.show()


# In[139]:


import matplotlib.pyplot as plt

# Metrics data
metrics_data = {
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores
}

# Create a box plot for each metric
plt.figure(figsize=(10, 6))
plt.boxplot(metrics_data.values(), labels=metrics_data.keys())
plt.title('Distribution of Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:


import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from scipy.stats import uniform, randint
from sklearn.preprocessing import LabelEncoder

# Load NCA GLCM features from CSV for the training set
nca_glcm_train_df = pd.read_csv('NCA_glcm_train_features.csv')

# Assuming your CSV file contains features and labels columns
features_train = nca_glcm_train_df.drop(columns=['Plant'])  # Update 'label' with the actual column name
labels_train = nca_glcm_train_df['Plant']  # Update 'label' with the actual column name

# Encode labels to integers
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)

# Define the parameter distributions for randomized search
param_dist = {
    'C': uniform(0.1, 10),  # Regularization parameter
    'gamma': uniform(0.01, 1),  # Kernel coefficient for 'rbf'
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'degree': randint(1, 10),  # Degree of the polynomial kernel
}

# Initialize SVM classifier
svm_classifier = SVC()

# Initialize StratifiedKFold
stratkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize RandomizedSearchCV
randomized_search = RandomizedSearchCV(
    svm_classifier,
    param_distributions=param_dist,
    n_iter=100,  # Adjust the number of iterations as needed
    scoring='accuracy',  # Use accuracy as the scoring metric
    cv=stratkf,  # Use StratifiedKFold for cross-validation
    verbose=2,  # Increase verbosity for detailed output
    n_jobs=-1  # Use all available CPU cores
)

# Fit the randomized search to the data
randomized_search.fit(features_train, labels_train_encoded)

# Print the best parameters and corresponding accuracy
best_params = randomized_search.best_params_
best_accuracy = randomized_search.best_score_
print("Best Hyperparameters:")
print(best_params)
print(f"Best Accuracy: {best_accuracy:.2f}")


# In[ ]:





# In[ ]:


import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time

# Load NCA GLCM features from CSV for the training set
nca_glcm_train_df = pd.read_csv('NCA_glcm_train_features.csv')

# Assuming your CSV file contains features and labels columns
features_train = nca_glcm_train_df.drop(columns=['Plant'])  # Update 'Plant' with the actual column name
labels_train = nca_glcm_train_df['Plant']  # Update 'Plant' with the actual column name

# Encode labels to integers
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)

# Initialize SVM classifier with the best hyperparameters
svm_classifier = SVC(C=10, gamma=0.01, kernel='linear', degree=10)

# Train the model
start_time = time.time()
svm_classifier.fit(features_train, labels_train_encoded)
end_time = time.time()

# Calculate predictions
predictions_train = svm_classifier.predict(features_train)

# Calculate metrics
accuracy = accuracy_score(labels_train_encoded, predictions_train) * 100
precision = precision_score(labels_train_encoded, predictions_train, average='weighted') * 100
recall = recall_score(labels_train_encoded, predictions_train, average='weighted') * 100
f1 = f1_score(labels_train_encoded, predictions_train, average='weighted') * 100
computation_time = end_time - start_time

# Print metrics
print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}%')
print(f'Recall: {recall:.2f}%')
print(f'F1 Score: {f1:.2f}%')
print(f'Computational Time: {computation_time:.4f} seconds')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




