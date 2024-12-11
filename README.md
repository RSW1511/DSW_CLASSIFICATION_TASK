# DSW_ML_TASK

# Loan Default Prediction Project

## Project Overview
This project focuses on developing a classification model for a trusted Non-Banking Financial Company (NBFC) to predict loan repayment behavior. Specifically, the model identifies potential defaulters and non-defaulters among applicants. By enhancing risk assessment and streamlining the loan approval process, the project aims to empower financial independence and improve financial inclusion.

## Repository Structure
- **`eda.ipynb`**: Contains the Exploratory Data Analysis (EDA) conducted on the dataset. This includes data cleaning, visualizations, and preliminary insights to understand the dataset's structure and potential relationships.
- **`Model_selection_evaluation.ipynb`**: Details the development, evaluation, and selection of machine learning models to predict loan default status. It includes performance metrics and justifications for the chosen model.

## Problem Statement
The NBFC seeks to:
1. Predict whether an applicant will default (1) or not default (0) on a loan.
2. Enhance risk assessment capabilities to improve loan approval processes and reduce financial risks.
3. Create an efficient and scalable model for better decision-making in loan disbursement.

## Data Overview
The datasets used:
1. **Historic Data (`train_data.xlsx`)**: Contains loan application details and repayment statuses from the past two years.
2. **Validation Data (`test_data.xlsx`)**: Includes loan application details for the last three months to test the model's performance.

### Key Features:
- `customer_id`, `transaction_date`, `sub_grade`, `term`, `home_ownership`, `cibil_score`, etc.
- Target variable: `loan_status` (1 for default, 0 for non-default).

## Workflow and Methodology

### Exploratory Data Analysis (EDA)
The `eda.ipynb` file:
- **Objective**: Understand the data distribution, identify relationships, and handle missing/duplicate data.
- **Key Steps**:
  1. Checked for missing values and duplicates.
  2. Analyzed numerical and categorical features using summary statistics and visualizations.
  3. Examined correlations to identify significant predictors of loan default.
  4. Created visualizations, e.g., boxplots for numerical features and count plots for categorical features, to gain insights into the data.
- **Outcome**: Identified relevant features, cleaned the dataset, and prepared it for modeling.

### Model Development and Selection
The `Model_selection_evaluation.ipynb` file:
- **Objective**: Build and evaluate machine learning models for classification.
- **Key Steps**:
  1. **Data Preprocessing**:
     - Encoded categorical variables.
     - Normalized/standardized numerical features.
     - Split the dataset into training and testing subsets.
  2. **Modeling**:
     - Built at least two models: Logistic Regression and Random Forest.
     - Used an object-oriented, class-based approach with methods for data loading, preprocessing, training, testing, and prediction.
  3. **Evaluation**:
     - Evaluated models using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
     - Performed hyperparameter tuning to optimize model performance.
  4. **Model Selection**:
     - Chose the model with the best balance of interpretability and performance (e.g., Random Forest for high AUC-ROC and feature importance insights).
- **Outcome**: Selected a model that effectively predicts loan defaults, justifying its use based on performance metrics.

### Justification for the Approach
1. **Comprehensive EDA**: Ensured a deep understanding of the dataset and addressed potential issues (e.g., missing values, duplicates).
2. **Feature Engineering**: Enhanced model performance by creating meaningful features (e.g., binning interest rates, encoding categorical variables).
3. **Model Diversity**: Compared multiple models to ensure robustness and reliability.
4. **Efficiency and Scalability**: Selected a model that balances computational efficiency with predictive power, suitable for large-scale deployment.
5. **Business Impact**: The chosen model aligns with the NBFC's goal of minimizing financial risk and improving decision-making in loan disbursement.

## How the Approach Meets the Task Summary
1. **Enhanced Risk Assessment**: By identifying potential defaulters with high accuracy, the model supports better risk management.
2. **Improved Loan Approval Process**: Automates the classification process, reducing manual effort and speeding up loan approvals.
3. **Data-Driven Decision-Making**: Provides insights into key factors influencing loan defaults, aiding strategic planning.

## Usage Instructions
1. **Run `eda.ipynb`**:
   - Ensure dependencies (e.g., Pandas, Matplotlib, Seaborn) are installed.
   - Execute the notebook to explore the dataset and prepare it for modeling.
2. **Run `Model_selection_evaluation.ipynb`**:
   - Train and evaluate machine learning models.
   - Review the results and justification for the selected model.

## Dependencies
- Python 3.7+
- Required libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, etc.

## Conclusion
This project successfully addresses the NBFC's need for a reliable classification model to predict loan defaults. By leveraging robust EDA and machine learning techniques, the approach ensures high accuracy, efficiency, and business relevance.

