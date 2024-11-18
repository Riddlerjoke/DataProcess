import pandas as pd
import random

# Generate synthetic dataset based on provided structure
data = {
    "Gender": random.choices(["Male", "Female"], k=100),
    "Age": [random.randint(18, 65) for _ in range(100)],
    "Occupation": random.choices(
        ["Software Engineer", "Doctor", "Sales Representative", "Teacher", "Nurse"], k=100
    ),
    "Sleep Duration": [round(random.uniform(4.0, 8.5), 1) for _ in range(100)],
    "Quality of Sleep": [random.randint(1, 10) for _ in range(100)],
    "Physical Activity Level": [random.randint(1, 10) for _ in range(100)],
    "Stress Level": [random.randint(10, 90) for _ in range(100)],
    "BMI Category": random.choices(
        ["Underweight", "Normal Weight", "Overweight", "Obese"], k=100
    ),
    "Blood Pressure": [random.randint(100, 160) for _ in range(100)],
    "Heart Rate": [random.randint(60, 100) for _ in range(100)],
    "Daily Steps": [f"{random.randint(2000, 15000):,}" for _ in range(100)],
    "Sleep Disorder": random.choices(
        ["Good Sleep", "Sleep Apnea", "Insomnia", "Restless Legs Syndrome"], k=100
    ),
}

# Create DataFrame
synthetic_dataset = pd.DataFrame(data)

file_path = "C:/Users/utilisateur/Desktop"
file_name = "synthetic_sleep_health_lifestyle.xlsx"
complete_path = f"{file_path}/{file_name}"
synthetic_dataset.to_excel(complete_path, index=False)

print(f"File saved at: {complete_path}")