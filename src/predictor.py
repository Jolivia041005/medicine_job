import joblib
import shap
import pandas as pd
import numpy as np
import json

# 加载模型（全局单例）
_model = None
_explainer = None

def load_model():
    global _model, _explainer
    if _model is None:
        _model = joblib.load("xgboost_diabetes_model.pkl")
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer

def predict_with_shap(input_dict):
    """
    输入：字典，键为特征名（Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age）
    输出：包含预测类别、概率、SHAP waterfall base64、特征重要性（DataFrame）的字典
    """
    model, explainer = load_model()
    X = pd.DataFrame([input_dict])
    prob = model.predict_proba(X)[0][1]  # 患病概率
    pred = int(prob >= 0.5)
    shap_values = explainer(X)
    # 生成特征重要性（SHAP值排序）
    importance = pd.DataFrame({
        'Feature': X.columns,
        'SHAP': shap_values.values[0]
    }).sort_values('SHAP', key=lambda x: abs(x), ascending=False)
    # 生成Waterfall图（Base64）
    import io, base64, matplotlib.pyplot as plt
    shap.plots.waterfall(shap_values[0], show=False)
    fig = plt.gcf()
    fig.suptitle(f"Diabetes Risk Probability = {prob:.2%}", fontsize=14, y=0.98)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return {
        'prediction': pred,
        'probability': round(prob, 4),
        'top_features': importance.head(3).to_dict('records'),
        'waterfall': b64,
        'importance_df': importance
    }