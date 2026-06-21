import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUM_RECORDS = 650
OUTPUT_PATH = "data/student_performance_dataset.csv"


def score_to_grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def generate_dataset(num_records: int = NUM_RECORDS) -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)

    study_hours_per_day = np.random.normal(loc=4.2, scale=1.9, size=num_records)
    study_hours_per_day = np.clip(study_hours_per_day, 0, 10).round(1)

    attendance_percentage = np.random.normal(loc=76, scale=14, size=num_records)
    attendance_percentage = np.clip(attendance_percentage, 30, 100).round(1)

    assignments_completed = np.random.normal(loc=7.0, scale=2.2, size=num_records)
    assignments_completed = np.clip(assignments_completed, 0, 10).round().astype(int)

    previous_semester_marks = np.random.normal(loc=68, scale=15, size=num_records)
    previous_semester_marks = np.clip(previous_semester_marks, 20, 100).round(1)

    class_participation = np.random.normal(loc=6.2, scale=2.1, size=num_records)
    class_participation = np.clip(class_participation, 0, 10).round(1)

    noise = np.random.normal(loc=0, scale=6, size=num_records)
    final_performance_score = (
        study_hours_per_day * 4.0
        + attendance_percentage * 0.25
        + assignments_completed * 3.0
        + previous_semester_marks * 0.35
        + class_participation * 2.0
        - 15
        + noise
    )
    final_performance_score = np.clip(final_performance_score, 0, 100).round(1)

    final_performance_grade = [score_to_grade(score) for score in final_performance_score]

    df = pd.DataFrame(
        {
            "study_hours_per_day": study_hours_per_day,
            "attendance_percentage": attendance_percentage,
            "assignments_completed": assignments_completed,
            "previous_semester_marks": previous_semester_marks,
            "class_participation": class_participation,
            "final_performance_score": final_performance_score,
            "final_performance_grade": final_performance_grade,
        }
    )

    duplicate_rows = df.sample(10, random_state=RANDOM_SEED)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    missing_indices = df.sample(12, random_state=RANDOM_SEED + 1).index
    df.loc[missing_indices[:4], "study_hours_per_day"] = np.nan
    df.loc[missing_indices[4:8], "attendance_percentage"] = np.nan
    df.loc[missing_indices[8:], "class_participation"] = np.nan

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset generated successfully: {OUTPUT_PATH}")
    print(f"Dataset shape: {dataset.shape}")
    print(dataset.head())
