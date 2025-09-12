# Insurance Premium Prediction using Artificial Neural Network

A comprehensive machine learning project that predicts insurance premiums using Artificial Neural Network (ANN) regression, optimized for Root Mean Squared Logarithmic Error (RMSLE) evaluation.

## 🎯 Project Overview

This project aims to predict insurance premiums based on various customer attributes including demographics, financial information, vehicle details, and policy characteristics. The model is specifically designed for competition evaluation using RMSLE as the primary metric.

## 📊 Dataset

The dataset contains insurance customer information with the following key features:

### Features Used:
- **Demographics**: Age, Gender, Marital Status, Number of Dependents, Education Level, Occupation, Location
- **Financial**: Annual Income, Credit Score
- **Insurance**: Previous Claims, Insurance Duration, Policy Type, Health Score
- **Vehicle**: Vehicle Age
- **Temporal**: Policy Start Date (engineered into year, month, day, day of week, seconds since epoch)

### Target Variable:
- **Premium Amount**: Continuous variable representing the insurance premium cost

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline
1. **Exploratory Data Analysis (EDA)**
   - Distribution analysis of numerical features
   - Correlation heatmap generation
   - Outlier detection using box plots
   - Skewness analysis and identification

2. **Feature Engineering**
   - DateTime feature extraction (year, month, day, day of week, seconds)
   - Log transformation of target variable
   - Box-Cox transformation for skewed numerical features
   - Square root transformation for specific features

3. **Data Cleaning**
   - Missing value imputation (mean, median, mode strategies)
   - Outlier removal based on statistical thresholds:
     - Previous Claims > 2.4
     - Annual Income < 5.5
     - Premium Amount > 3000

4. **Encoding and Scaling**
   - Label encoding for categorical variables using `factorize()`
   - StandardScaler for feature normalization
   - Memory-efficient data type conversion (int32, float32)

### Model Architecture

**Artificial Neural Network (ANN) Specifications:**
```
Input Layer: 256 neurons (ReLU) + BatchNormalization + Dropout(0.3)
Hidden Layer 1: 128 neurons (ReLU) + BatchNormalization + Dropout(0.3)
Hidden Layer 2: 64 neurons (ReLU) + BatchNormalization + Dropout(0.2)
Hidden Layer 3: 32 neurons (ReLU) + Dropout(0.2)
Hidden Layer 4: 16 neurons (ReLU)
Output Layer: 1 neuron (Linear activation)
```

**Model Configuration:**
- **Optimizer**: Adam (learning_rate=0.01)
- **Loss Function**: Custom RMSLE loss
- **Metrics**: Mean Absolute Error (MAE)
- **Total Parameters**: ~70,000 trainable parameters

### Training Configuration
- **Epochs**: Up to 100 (with early stopping)
- **Batch Size**: 32
- **Validation Split**: 20%
- **Train-Test Split**: 80/20

**Callbacks:**
- **Early Stopping**: Patience=15, monitors validation loss
- **Learning Rate Reduction**: Factor=0.2, Patience=10, Min LR=1e-7

## 📈 Model Performance

The model is evaluated using multiple metrics:

### Primary Metric (Competition):
- **RMSLE (Root Mean Squared Logarithmic Error)**: Optimized for competition evaluation

### Additional Metrics:
- **RMSE (Root Mean Squared Error)**: Standard regression metric
- **MAE (Mean Absolute Error)**: Robust to outliers
- **R² Score**: Coefficient of determination

## 🚀 Usage

### Running the Project

1. **Environment Setup**
```bash
# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow scipy
```

2. **Data Preparation**
```python
# Load and preprocess data
python notebook.ipynb  # Execute cells 1-38
```

3. **Model Training**
```python
# Train ANN model
python notebook.ipynb  # Execute cells 39-44
```

4. **Generate Predictions**
```python
# Create submission file
python notebook.ipynb  # Execute cell 45
```

### File Structure
```
├── train.csv              # Training dataset
├── test.csv               # Test dataset  
├── notebook.ipynb         # Main analysis and model notebook
├── submission.csv         # Generated predictions (output)
├── sample_submission.csv  # Sample submission format
└── README.md             # Project documentation
```

## 🔍 Key Features

### Data Visualization
- **Distribution plots** for numerical features
- **Correlation heatmap** for feature relationships
- **Box plots** for outlier detection
- **KDE plots** for skewness analysis
- **Training curves** for model performance monitoring

### Preprocessing Excellence
- **Memory optimization** with efficient data types
- **Comprehensive outlier handling** with statistical methods
- **Feature transformation** for improved model performance
- **Missing value strategies** tailored to feature types

### Model Robustness
- **Regularization techniques** (Dropout, BatchNormalization)
- **Custom loss function** optimized for RMSLE
- **Callback mechanisms** for training optimization
- **Validation monitoring** to prevent overfitting

## 📊 Model Evaluation Visualizations

The notebook generates comprehensive visualizations:
1. **Training/Validation Loss Curves**: Monitor RMSLE optimization
2. **Training/Validation MAE Curves**: Track auxiliary metric performance
3. **Actual vs Predicted Scatter Plots**: Visualize prediction accuracy
4. **Feature Distribution Analysis**: Understand data characteristics

## 🎯 Competition Readiness

- **RMSLE Optimization**: Direct optimization of competition metric
- **Submission File Generation**: Automatic creation of `submission.csv`
- **Memory Efficient**: Handles large datasets without memory issues
- **Robust Pipeline**: Comprehensive preprocessing and validation

## 🔧 Technical Specifications

- **Framework**: TensorFlow/Keras for deep learning
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization**: Matplotlib, Seaborn for plotting
- **Preprocessing**: Scikit-learn for scaling and splitting
- **Statistical Analysis**: SciPy for transformations

## 📝 Results

The model achieves competitive performance with:
- Optimized RMSLE score for insurance premium prediction
- Robust generalization through proper regularization
- Efficient training with callback mechanisms
- Comprehensive evaluation metrics

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements. Key areas for enhancement:
- Hyperparameter tuning
- Feature engineering techniques
- Model architecture optimization
- Ensemble methods integration

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This project demonstrates advanced machine learning techniques for regression problems with specific focus on RMSLE optimization for competition environments.