import pandas as pd
import io

def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def export_report(query: str, df: pd.DataFrame, summary: dict, evidence_text: str, duration: float) -> str:
    """生成 Markdown 格式报告"""
    lines = []
    lines.append("# 临床队列检索报告")
    lines.append(f"**查询词**: {query}")
    lines.append(f"**检索耗时**: {duration:.3f} 秒")
    lines.append(f"**匹配患者数**: {len(df)}")
    lines.append("\n## 结构化证据")
    lines.append(evidence_text)
    lines.append("\n## 详细统计")
    for k, v in summary.items():
        if isinstance(v, dict):
            lines.append(f"- {k}: {', '.join([f'{sk}: {sv}' for sk, sv in v.items()])}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("\n## 患者列表（前20条）")
    lines.append(df.head(20).to_markdown(index=False))
    lines.append("\n---\n报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    return "\n".join(lines)