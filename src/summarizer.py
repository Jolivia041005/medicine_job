import pandas as pd

def generate_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"error": "无数据可摘要"}
    summary = {
        "总人数": len(df),
        "年龄分布": {
            "均值": round(df["age"].mean(), 1),
            "中位数": int(df["age"].median()),
            "最小值": int(df["age"].min()),
            "最大值": int(df["age"].max())
        },
        "性别分布": df["sex"].value_counts().to_dict(),
        "诊断分布": df["diagnoses"].value_counts().to_dict(),
        "用药分布": df["medications"].value_counts().to_dict(),
        "30天内再入院率": f"{df['readmitted_30d'].mean()*100:.1f}%",
        "住院死亡率": f"{df['in_hospital_death'].mean()*100:.1f}%",
        "平均BMI": round(df["bmi"].mean(), 1)
    }
    return summary

def generate_structured_evidence(df: pd.DataFrame) -> str:
    """生成自然语言的结构化证据摘要，适合医生阅读"""
    if df.empty:
        return "未找到匹配患者。"
    s = generate_summary(df)
    lines = [
        f"共检索到 {s['总人数']} 名患者，",
        f"年龄范围 {s['年龄分布']['最小值']}~{s['年龄分布']['最大值']} 岁（均值 {s['年龄分布']['均值']} 岁），",
        f"性别：男性 {s['性别分布'].get('Male', 0)} 人，女性 {s['性别分布'].get('Female', 0)} 人。",
        f"主要诊断：{', '.join([f'{k}: {v}人' for k,v in s['诊断分布'].items()])}。",
        f"用药情况：{', '.join([f'{k}: {v}人' for k,v in s['用药分布'].items()])}。",
        f"平均BMI：{s['平均BMI']}，30天再入院率 {s['30天内再入院率']}，住院死亡率 {s['住院死亡率']}。"
    ]
    return " ".join(lines)