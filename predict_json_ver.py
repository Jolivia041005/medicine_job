import io
import base64
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 加载模型（程序启动时执行一次）
# ============================================================

model = joblib.load("xgboost_diabetes_model.pkl")

import numpy as np

# 构造一个简单的背景数据集（来自 PIMA 糖尿病数据集的典型样本）
background_data = np.array([
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],
    [1, 85, 66, 29, 0, 26.6, 0.351, 31],
    [8, 183, 64, 0, 0, 23.3, 0.672, 32],
    [1, 89, 66, 23, 94, 28.1, 0.167, 21],
    [0, 137, 40, 35, 168, 43.1, 2.288, 33]
])

# 以概率模式 + 介入式扰动创建解释器
explainer = shap.TreeExplainer(
    model,
    background_data,
    feature_perturbation="interventional",
    model_output="probability"
)


# ============================================================
# 生成Waterfall图（Base64）
# ============================================================

def get_waterfall_base64(shap_values, probability):
    """
    返回Base64编码的Waterfall图片
    """

    # SHAP自动创建Figure
    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    fig = plt.gcf()

    fig.suptitle(
        f"Diabetes Risk Probability = {probability:.2%}",
        fontsize=14,
        y=0.98
    )

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    buffer.seek(0)

    image_base64 = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    return image_base64


# ============================================================
# 返回Top风险因素
# ============================================================

def top_features(importance, top_n=3):

    result = []

    for _, row in importance.head(top_n).iterrows():

        result.append({

            "feature": row["Feature"],

            "effect": "增加风险" if row["SHAP"] > 0 else "降低风险",

            "shap": round(float(row["SHAP"]), 3)

        })

    return result


# ============================================================
# 单个患者预测
# ============================================================

def predict(patient):
    """
    Parameters
    ----------
    patient : dict
        例如：
        {
            "Age":45,
            "BMI":28.3,
            ...
        }

    Returns
    -------
    dict
    """

    # 转DataFrame
    X = pd.DataFrame([patient])

    # 预测类别
    prediction = int(
        model.predict(X)[0]
    )

    # 患病概率
    probability = float(
        model.predict_proba(X)[0][1]
    )

    # SHAP解释
    shap_values = explainer(X)

    # 特征重要性
    importance = pd.DataFrame({

        "Feature": X.columns,

        "SHAP": shap_values.values[0]

    })

    importance = importance.reindex(

        importance["SHAP"].abs()

        .sort_values(ascending=False)

        .index

    ).reset_index(drop=True)

    # Waterfall图(Base64)
    waterfall = get_waterfall_base64(
        shap_values,
        probability
    )

    # 返回JSON可直接序列化的数据
    return {

        "prediction": prediction,

        "probability": round(probability, 4),

        "top_features": top_features(
            importance,
            top_n=3
        ),

        "waterfall": waterfall

    }


# 调用：
# result = predict(patient)

# 其中 patient 是一个字典，例如：

# patient = {
#     "Pregnancies": 2,
#     "Glucose": 150,
#     "BloodPressure": 72,
#     "SkinThickness": 35,
#     "Insulin": 120,
#     "BMI": 33.6,
#     "DiabetesPedigreeFunction": 0.62,
#     "Age": 45
# }

# 返回：

# {
#     "prediction": 1,
#     "probability": 0.87,
#     "top_features": [...],
#     "waterfall": "base64..."
# }