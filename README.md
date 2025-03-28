# Fraud Detection using Machine Learning & Deep Learning

## Project Overview
This project aims to detect fraudulent transactions using various machine learning and deep learning models. The dataset used consists of transaction details labeled as fraud or non-fraud. The project involves data preprocessing, exploratory data analysis (EDA), and the implementation of multiple models for fraud detection.

## Dataset
- The dataset is sourced from Kaggle and includes transactional records.
- The main dataset files are:
  - `fraudTrain.csv` (Training dataset)
  - `fraudTest.csv` (Testing dataset)

## Technologies Used
- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras

## Steps Involved
### 1. Data Loading
- The dataset is read using Pandas.
- The structure of the dataset is examined, including missing values and duplicates.

### 2. Exploratory Data Analysis (EDA)
- Statistical summaries and visualizations are used to understand data distribution.
- Fraud vs. non-fraud transactions are compared.

### 3. Data Preprocessing
- Unnecessary columns are dropped.
- Categorical columns are encoded using Label Encoding.
- Numerical features are scaled using StandardScaler.
- The dataset is balanced using over-sampling and under-sampling techniques.

### 4. Machine Learning Models Implemented
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **XGBoost**

### 5. Deep Learning Models Implemented
- **Feedforward Neural Network (Sequential Model)**
- **Long Short-Term Memory (LSTM) Network**

### 6. Model Evaluation
- Performance is evaluated using metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC-ROC
- Classification reports are generated for each model.

## How to Run the Project
1. Clone the repository or download the dataset from Kaggle.
2. Install required dependencies using:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras xgboost
   ```
3. Run the Jupyter Notebook or Python script:
   ```bash
   python fraud_detection.py
   ```

## Results
- The deep learning models (especially LSTM) showed significant improvements in detecting fraud transactions.
- XGBoost performed well among the machine learning models.

## Future Improvements
- Experimenting with additional feature engineering techniques.
- Implementing anomaly detection models such as Isolation Forest and Autoencoders.
- Testing with real-world financial transaction datasets.

## Contributors
- **Suyash Thakur**

## References
- Kaggle Dataset: [Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)
