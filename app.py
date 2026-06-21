import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

MODEL_PATH = "models/student_performance_model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATASET_PATH = "data/student_performance_dataset.csv"
REPORTS_FOLDER = "reports"

FEATURE_COLUMNS = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignments_completed",
    "previous_semester_marks",
    "class_participation",
]
TARGET_COLUMN = "final_performance_score"
GRADE_COLUMN = "final_performance_grade"

model_package = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

model = model_package["model"]
model_name = model_package["model_name"]
feature_columns = model_package["feature_columns"]
uses_scaler = model_package["uses_scaler"]


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


def load_clean_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in FEATURE_COLUMNS:
        df[column] = df[column].fillna(df[column].median())
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(df[TARGET_COLUMN].median())
    df[GRADE_COLUMN] = df[TARGET_COLUMN].apply(score_to_grade)
    df = df.drop_duplicates()
    return df


def build_distribution(values: pd.Series, bins: list[int]) -> list[dict]:
    counts, edges = np.histogram(values, bins=bins)
    distribution = []
    for index, count in enumerate(counts):
        label = f"{int(edges[index])}-{int(edges[index + 1])}"
        distribution.append({"label": label, "count": int(count)})
    return distribution


def get_dashboard_payload() -> dict:
    df = load_clean_dataset()
    grade_order = ["A", "B", "C", "D", "F"]
    grade_counts = df[GRADE_COLUMN].value_counts().reindex(grade_order, fill_value=0)
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr().round(2)

    return {
        "insights": {
            "total_records": int(len(df)),
            "average_marks": round(float(df[TARGET_COLUMN].mean()), 2),
            "average_attendance": round(float(df["attendance_percentage"].mean()), 2),
        },
        "attendance_distribution": build_distribution(df["attendance_percentage"], [30, 40, 50, 60, 70, 80, 90, 100]),
        "marks_distribution": build_distribution(df[TARGET_COLUMN], [0, 40, 55, 70, 85, 100]),
        "performance_categories": [
            {"grade": grade, "count": int(grade_counts.loc[grade])}
            for grade in grade_order
        ],
        "correlation_columns": list(corr.columns),
        "correlation_values": corr.values.tolist(),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/reports/<path:filename>")
def report_file(filename: str):
    return send_from_directory(REPORTS_FOLDER, filename)


@app.route("/dashboard-data")
def dashboard_data():
    try:
        return jsonify(get_dashboard_payload())
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "study_hours_per_day": float(request.form.get("study_hours_per_day")),
            "attendance_percentage": float(request.form.get("attendance_percentage")),
            "assignments_completed": float(request.form.get("assignments_completed")),
            "previous_semester_marks": float(request.form.get("previous_semester_marks")),
            "class_participation": float(request.form.get("class_participation")),
        }

        if not 0 <= input_data["study_hours_per_day"] <= 10:
            return jsonify({"error": "Study hours must be between 0 and 10."}), 400
        if not 0 <= input_data["attendance_percentage"] <= 100:
            return jsonify({"error": "Attendance percentage must be between 0 and 100."}), 400
        if not 0 <= input_data["assignments_completed"] <= 10:
            return jsonify({"error": "Assignments completed must be between 0 and 10."}), 400
        if not 0 <= input_data["previous_semester_marks"] <= 100:
            return jsonify({"error": "Previous semester marks must be between 0 and 100."}), 400
        if not 0 <= input_data["class_participation"] <= 10:
            return jsonify({"error": "Class participation must be between 0 and 10."}), 400

        input_df = pd.DataFrame([input_data], columns=feature_columns)
        model_input = scaler.transform(input_df) if uses_scaler else input_df
        predicted_score = float(model.predict(model_input)[0])
        predicted_score = float(np.clip(predicted_score, 0, 100))
        predicted_grade = score_to_grade(predicted_score)

        return jsonify(
            {
                "predicted_score": round(predicted_score, 2),
                "predicted_grade": predicted_grade,
                "model_used": model_name,
                "message": f"Predicted Final Performance Grade: {predicted_grade}",
            }
        )

    except ValueError:
        return jsonify({"error": "Please enter valid numeric values."}), 400
    except Exception as error:
        return jsonify({"error": f"Prediction failed: {str(error)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="127.0.0.1", port=port)
