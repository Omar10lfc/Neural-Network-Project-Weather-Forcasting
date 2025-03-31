# Weather Prediction using LSTM and GRU

This project focuses on predicting weather conditions using historical weather data. The workflow follows a structured pipeline that includes data preprocessing, exploratory data analysis (EDA), model building, and evaluation of LSTM and GRU models.

## Workflow Overview

### 1. Importing Necessary Libraries
The notebook begins by importing essential libraries for data manipulation, visualization, deep learning, and evaluation. The PyTorch library is used for building and training neural network models.

### 2. Data Loading and Exploration
- The dataset (`weatherHistory.csv`) is loaded using Pandas.
- Initial exploration includes displaying the first few rows, checking data types, and understanding missing values.
- Missing values are handled using forward fill to maintain temporal consistency.
- Redundant columns (e.g., `Daily Summary`) are removed to avoid duplication of information.

### 3. Data Preprocessing and Feature Engineering
- Features are selected and transformed to prepare the dataset for deep learning models.
- Normalization is performed using `MinMaxScaler` to scale numerical features.
- Categorical variables (e.g., `Precip Type`) are encoded using `OneHotEncoder`.
- The dataset is split into training and testing sets using `train_test_split`.

### 4. Model Preparation and Training
- The dataset is converted into `TensorDataset` and loaded into PyTorch `DataLoader`.
- Two models are defined and implemented:
  - **LSTM Model**: A Long Short-Term Memory network for capturing temporal dependencies.
  - **GRU Model**: A Gated Recurrent Unit network as an alternative to LSTM.
- Both models are trained using the Adam optimizer and Mean Squared Error (MSE) loss function.
- Training involves multiple epochs, and performance is monitored using loss curves.

### 5. Model Evaluation and Comparison
- The trained models are evaluated using standard metrics:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R-squared Score (R²)
- Performance comparison between LSTM and GRU models is conducted to determine the better-performing model.

## Results and Observations
- The notebook provides a visual representation of loss curves for both models.
- Evaluation metrics help in assessing the accuracy and generalization of LSTM and GRU for weather prediction.
- The better model is selected based on its ability to minimize error and maximize predictive performance.

## Requirements
To run this notebook, install the required dependencies:
```bash
pip install pandas numpy matplotlib seaborn torch scikit-learn
```
## Conclusion
This project demonstrates the application of deep learning in time-series forecasting, particularly using LSTM and GRU networks. By preprocessing data effectively and leveraging sequential models, we can make meaningful weather predictions.
