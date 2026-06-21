# Student Academic Performance Prediction Web Application

This is a beginner-to-intermediate AI/ML semester project that predicts a student's academic performance using Python, Machine Learning, Flask, HTML, CSS, and JavaScript.

The project follows a **regression-based ML approach**. The model predicts a numeric `final_performance_score`, and the app converts that score into a final grade:

| Score Range | Grade |
|---|---|
| 85 - 100 | A |
| 70 - 84 | B |
| 55 - 69 | C |
| 40 - 54 | D |
| 0 - 39 | F |

---

## Project Features

- Synthetic dataset with more than 500 student records
- Realistic academic performance patterns
- Data cleaning
- Missing value handling
- Duplicate removal
- Feature selection
- Data visualization
- Train-test split
- Machine learning model training
- Model comparison
- Flask backend integration
- Modern frontend with HTML, CSS, and JavaScript
- Local web app prediction
- README documentation for academic submission

---

## Technologies Used

- Python
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- HTML
- CSS
- JavaScript

---

## Dataset Description

Dataset file:

```bash
 data/student_performance_dataset.csv
```

The dataset contains these columns:

| Column | Description |
|---|---|
| `study_hours_per_day` | Number of hours a student studies daily |
| `attendance_percentage` | Student attendance percentage |
| `assignments_completed` | Number of assignments completed out of 10 |
| `previous_semester_marks` | Marks obtained in the previous semester |
| `class_participation` | Participation score from 0 to 10 |
| `final_performance_score` | Numeric target score from 0 to 100 |
| `final_performance_grade` | Grade label A/B/C/D/F derived from the score |

The dataset is generated using realistic patterns:

- Higher study hours usually improve performance.
- Higher attendance usually improves performance.
- More completed assignments improve performance.
- Higher previous marks improve performance.
- Better class participation improves performance.

The dataset also intentionally includes a small number of missing values and duplicate rows so that preprocessing steps can be demonstrated.

---

## Machine Learning Workflow

### 1. Data Cleaning

The training script checks:

- Dataset shape
- Column names
- Data types
- Missing values
- Duplicate rows

### 2. Missing Value Handling

Missing numeric values are filled using the median value of each feature column.

### 3. Duplicate Removal

Duplicate records are removed using:

```python
df = df.drop_duplicates()
```

### 4. Feature Selection

Input features:

```python
study_hours_per_day
attendance_percentage
assignments_completed
previous_semester_marks
class_participation
```

Target variable:

```python
final_performance_score
```

### 5. Data Visualization

The training script creates these visualizations:

```bash
reports/score_distribution.png
reports/grade_count.png
reports/correlation_heatmap.png
reports/model_comparison_rmse.png
reports/model_comparison_mse.png
```

### 6. Train-Test Split

The dataset is split into training and testing sets:

```python
test_size=0.2
random_state=42
```

### 7. Model Training

The project trains and compares these models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

XGBoost is optional and not included by default to keep setup simple for university submission.

### 8. Model Evaluation

Regression models are evaluated using:

- Accuracy Score - grade-level accuracy after converting numeric scores into A/B/C/D/F grades
- MAE - Mean Absolute Error
- MSE - Mean Squared Error
- RMSE - Root Mean Squared Error
- R2 Score

Lower MAE, MSE, and RMSE mean better score prediction. Higher Accuracy Score means better grade prediction. Higher R2 Score means the model explains the target variable better.

---

## Current Model Results

After training, the best model is selected mainly using RMSE and MSE. Accuracy Score and R2 Score are also displayed for clear comparison.

Example result:

| Model | Accuracy Score | MAE | MSE | RMSE | R2 Score |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 0.62 | 4.90 | 38.20 | 6.18 | 0.78 |
| Random Forest Regressor | 0.56 | 5.82 | 57.60 | 7.59 | 0.67 |
| Decision Tree Regressor | 0.50 | 7.34 | 90.65 | 9.52 | 0.49 |

Best model:

```bash
Linear Regression
```

Saved model file:

```bash
models/student_performance_model.pkl
```

Saved preprocessing file:

```bash
models/scaler.pkl
```

---

## Project Structure

```text
student_performance_prediction_app/
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── student_performance_dataset.csv
├── models/
│   ├── student_performance_model.pkl
│   └── scaler.pkl
├── reports/
│   ├── evaluation_results.txt
│   ├── score_distribution.png
│   ├── grade_count.png
│   ├── correlation_heatmap.png
│   ├── model_comparison_rmse.png
│   └── model_comparison_mse.png
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

---

## Installation

Open terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

If you use Mac or Linux and `pip` does not work, try:

```bash
python3 -m pip install -r requirements.txt
```

---

## Generate Dataset

```bash
python generate_dataset.py
```

On Mac or Linux:

```bash
python3 generate_dataset.py
```

---

## Train the Model

```bash
python train_model.py
```

On Mac or Linux:

```bash
python3 train_model.py
```

This will:

- Load the dataset
- Clean the data
- Handle missing values
- Remove duplicates
- Select features
- Create visualizations
- Split data into training and testing sets
- Train regression models
- Compare models
- Save the best model

---

## Run the Flask App

```bash
python app.py
```

On Mac or Linux:

```bash
python3 app.py
```

Then open this URL in your browser:

```text
http://127.0.0.1:5000
```

---

## Test Prediction Example

Try these values in the web form:

| Field | Example Value |
|---|---:|
| Study Hours Per Day | 5 |
| Attendance Percentage | 85 |
| Assignments Completed | 8 |
| Previous Semester Marks | 78 |
| Class Participation | 7 |

The app will return:

```text
Predicted score
Predicted final grade
Model used
```

---

## Common Errors and Fixes

### 1. `ModuleNotFoundError`

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. `python is not recognized`

On Windows, install Python and select:

```text
Add Python to PATH
```

Then restart terminal.

### 3. Flask app is not opening

Make sure the app is running:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### 4. Model file not found

Run training again:

```bash
python generate_dataset.py
python train_model.py
```

---

## Future Improvements

- Add XGBoost Regressor
- Add database storage for prediction history
- Add login system for admin and students
- Deploy the app on Render or Railway
- Add downloadable prediction reports
- Improve UI with charts and dashboards
- Use a real academic dataset from Kaggle

---

## Academic Submission Note

This project demonstrates:

- Machine Learning fundamentals
- Regression model training
- Data preprocessing
- Model evaluation
- Full-stack integration
- Flask backend development
- Frontend form handling
- Project documentation
