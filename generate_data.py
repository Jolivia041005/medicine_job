import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)
n = 500

data = {
    "patient_id": [f"P{i:04d}" for i in range(1, n+1)],
    "age": np.random.randint(18, 90, n),
    "sex": np.random.choice(["Male", "Female"], n),
    "bmi": np.random.normal(28, 6, n).round(1),
    "diagnoses": np.random.choice(
        ["Diabetes", "Hypertension", "Diabetes+Hypertension", 
         "Diabetes+Chronic Kidney Disease", "None"], n,
        p=[0.3, 0.2, 0.25, 0.15, 0.1]
    ),
    "medications": np.random.choice(
        ["Metformin", "Insulin", "SGLT2 inhibitor", "Metformin+Insulin", "None"], n,
        p=[0.3, 0.2, 0.15, 0.25, 0.1]
    ),
    "readmitted_30d": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    "in_hospital_death": np.random.choice([0, 1], n, p=[0.95, 0.05]),
}

df = pd.DataFrame(data)
df["description"] = df.apply(
    lambda r: f"{r['sex']} age {r['age']} with {r['diagnoses']} taking {r['medications']}",
    axis=1
)

df.to_parquet("data/patient_data.parquet", index=False)
print("✅ 患者数据已生成 (data/patient_data.parquet)")