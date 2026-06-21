import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

DATASET_PATH = "data/student_performance_dataset.csv"
MODEL_PATH = "models/student_performance_model.pkl"
SCALER_PATH = "models/scaler.pkl"
REPORT_PATH = "reports/evaluation_results.txt"

FEATURE_COLUMNS = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignments_completed",
    "previous_semester_marks",
    "class_participation",
]
TARGET_COLUMN = "final_performance_score"
GRADE_COLUMN = "final_performance_grade"


def score_to_grade(score: float) -> str:
    score = max(0, min(100, float(score)))
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in FEATURE_COLUMNS:
        df[column] = df[column].fillna(df[column].median())
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(df[TARGET_COLUMN].median())
    df[GRADE_COLUMN] = df[TARGET_COLUMN].apply(score_to_grade)
    df = df.drop_duplicates()
    return df


def create_visualizations(df: pd.DataFrame, results_df: pd.DataFrame | None = None) -> None:
    os.makedirs("reports", exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    plt.hist(df["attendance_percentage"], bins=15)
    plt.title("Attendance Distribution")
    plt.xlabel("Attendance Percentage")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig("reports/attendance_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df[TARGET_COLUMN], bins=20)
    plt.title("Marks Distribution")
    plt.xlabel("Final Performance Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig("reports/marks_distribution.png", dpi=150)
    plt.savefig("reports/score_distribution.png", dpi=150)
    plt.close()

    grade_order = ["A", "B", "C", "D", "F"]
    grade_counts = df[GRADE_COLUMN].value_counts().reindex(grade_order, fill_value=0)
    plt.figure(figsize=(8, 5))
    plt.bar(grade_counts.index, grade_counts.values)
    plt.title("Performance Categories")
    plt.xlabel("Grade")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig("reports/performance_categories.png", dpi=150)
    plt.savefig("reports/grade_count.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.heatmap(df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("reports/correlation_heatmap.png", dpi=150)
    plt.close()

    if results_df is not None:
        plt.figure(figsize=(8, 5))
        plt.barh(results_df["Model"], results_df["RMSE"])
        plt.title("Model Comparison Based on RMSE")
        plt.xlabel("RMSE - Lower is Better")
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig("reports/model_comparison_rmse.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.barh(results_df["Model"], results_df["MSE"])
        plt.title("Model Comparison Based on MSE")
        plt.xlabel("MSE - Lower is Better")
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig("reports/model_comparison_mse.png", dpi=150)
        plt.close()


def main() -> None:
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    raw_df = pd.read_csv(DATASET_PATH)
    missing_before = raw_df.isnull().sum()
    duplicates_before = raw_df.duplicated().sum()
    df = clean_dataset(raw_df)
    missing_after = df.isnull().sum()
    duplicates_after = df.duplicated().sum()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=6),
        "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=120, max_depth=8),
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        if model_name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        predictions = np.clip(predictions, 0, 100)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        actual_grades = [score_to_grade(score) for score in y_test]
        predicted_grades = [score_to_grade(score) for score in predictions]
        grade_accuracy = accuracy_score(actual_grades, predicted_grades)

        results.append(
            {
                "Model": model_name,
                "Accuracy Score": grade_accuracy,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2 Score": r2,
            }
        )
        trained_models[model_name] = model

    results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    model_package = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "uses_scaler": best_model_name == "Linear Regression",
    }
    joblib.dump(model_package, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    create_visualizations(df, results_df)

    report_lines = [
        "STUDENT ACADEMIC PERFORMANCE PREDICTION - MODEL REPORT",
        "=" * 70,
        f"Original dataset shape: {raw_df.shape}",
        f"Cleaned dataset shape: {df.shape}",
        "",
        "DATA CLEANING",
        "Column names were standardized and numeric fields were converted safely.",
        "",
        "MISSING VALUE HANDLING",
        "Missing values before handling:",
        str(missing_before),
        "",
        "Missing values after handling:",
        str(missing_after),
        "",
        "DUPLICATE REMOVAL",
        f"Duplicate rows before removal: {duplicates_before}",
        f"Duplicate rows after removal: {duplicates_after}",
        "",
        "FEATURE SELECTION",
        f"Selected features: {FEATURE_COLUMNS}",
        f"Target variable: {TARGET_COLUMN}",
        "",
        "DATA VISUALIZATION FILES",
        "reports/attendance_distribution.png",
        "reports/marks_distribution.png",
        "reports/correlation_heatmap.png",
        "reports/performance_categories.png",
        "reports/model_comparison_rmse.png",
        "reports/model_comparison_mse.png",
        "",
        "TRAIN-TEST SPLIT",
        f"Training records: {X_train.shape[0]}",
        f"Testing records: {X_test.shape[0]}",
        "",
        "MODEL EVALUATION",
        "Accuracy Score is grade-level accuracy after converting numeric scores into A/B/C/D/F.",
        "MAE, MSE, and RMSE measure score prediction error.",
        "R2 Score measures how well the model explains target score variation.",
        "",
        results_df.to_string(index=False),
        "",
        f"Best selected model: {best_model_name}",
        f"Saved best model: {MODEL_PATH}",
        f"Saved scaler: {SCALER_PATH}",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    print("Training completed successfully.")
    print("\nMODEL EVALUATION RESULTS")
    print(results_df.to_string(index=False))
    print(f"Best model: {best_model_name}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
