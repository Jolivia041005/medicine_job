from __future__ import annotations

import html
import math
import re
import sqlite3
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ====================== 路径常量 ======================
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "t2d_islet.sqlite"
COUNTS_PATH = ROOT / "data" / "log2_normalized_counts.tsv.gz"
EXTERNAL_VALIDATION_SUMMARY_PATH = (
    ROOT / "data" / "model_training" / "processed" / "model_performance_summary.csv"
)
EXTERNAL_PREDICTIONS_PATH = (
    ROOT / "data" / "model_training" / "processed" / "model_external_predictions.csv"
)
MODEL_TRAINING_METADATA_PATH = ROOT / "data" / "model_training" / "processed" / "model_training_metadata.csv"
MODEL_TRAINING_EXPRESSION_PATH = ROOT / "data" / "model_training" / "processed" / "model_training_expression_zscore.csv"
MODEL_TRAINING_PREDICTIONS_PATH = ROOT / "data" / "model_training" / "processed" / "model_training_predictions.csv"
MODEL_FEATURE_COEFFICIENTS_PATH = ROOT / "data" / "model_training" / "processed" / "model_feature_coefficients.csv"
MODEL_DATASET_SUMMARY_PATH = ROOT / "data" / "model_training" / "processed" / "training_dataset_summary.csv"
CANDIDATE_EVIDENCE_PATH = ROOT / "data" / "reference" / "candidate_gene_evidence.csv"
MODEL_FIGURES = {
    "sample_counts": ROOT / "results" / "day3" / "model_training_sample_counts.png",
    "roc": ROOT / "results" / "day3" / "model_roc_curves.png",
    "confusion": ROOT / "results" / "day3" / "model_confusion_matrices.png",
    "coefficients": ROOT / "results" / "day3" / "model_feature_coefficients.png",
    "probability": ROOT / "results" / "day3" / "model_probability_by_dataset.png",
}
MAIN_SUBSET = "all_t2d_nd"
MAIN_CONTRAST = "T2D_vs_ND"

STATUS_ORDER = ["ND", "IGT", "IFG", "T3cD", "T2D"]
STATUS_COLORS = {
    "ND": "#2563eb",
    "IGT": "#16a34a",
    "IFG": "#b45309",
    "T3cD": "#7c3aed",
    "T2D": "#dc2626",
}
DEG_COLORS = {
    "T2D 上调高效应": "#dc2626",
    "T2D 下调高效应": "#2563eb",
    "FDR 显著但效应较小": "#d97706",
    "未显著": "#9ca3af",
}
QUERY_LABELS = {
    "all_high_effect_deg": "全部高效应 DEG",
    "up_high_effect_deg": "T2D 上调高效应 DEG",
    "down_high_effect_deg": "T2D 下调高效应 DEG",
    "all_fdr_deg": "全部 FDR 显著 DEG",
}
MODEL_FEATURES = [
    "SLC2A2",
    "ARG2",
    "CLTRN",
    "PCOLCE2",
    "IAPP",
    "PPP1R1A",
    "GABRA2",
    "OPRD1",
    "CHL1",
    "GLRA1",
    "GLP1R",
    "PDX1",
    "NKX6-1",
    "SLC30A8",
]
GENE_ALIASES = {
    "GLUT2": "SLC2A2",
    "SLC2A2": "SLC2A2",
    "TMEM27": "CLTRN",
    "COLLECTRIN": "CLTRN",
    "CLTRN": "CLTRN",
    "AMYLIN": "IAPP",
    "NKX6.1": "NKX6-1",
    "NKX6-1": "NKX6-1",
    "ZNT8": "SLC30A8",
    "ZNT-8": "SLC30A8",
    "TMEM27": "CLTRN",
}

GENE_EVIDENCE = {
    "SLC2A2": {
        "label": "外部验证：人胰岛/T2D 相关",
        "summary": (
            "本队列中 SLC2A2 在 T2D 中下调。外部研究报道 SLC2A2/GLUT2 与人胰岛"
            "胰岛素分泌和糖尿病表型有关，因此该基因适合作为有外部支持的重点候选。"
        ),
        "links": [
            ("Bacos et al., JCI 2023", "https://www.jci.org/articles/view/163612"),
            ("Sansbury et al., Diabetologia 2012", "https://pubmed.ncbi.nlm.nih.gov/22660720/"),
        ],
    },
    "ARG2": {
        "label": "外部验证：人 beta 细胞/T2D 相关",
        "summary": (
            "本队列中 ARG2 在 T2D 中下调。已有研究直接讨论 ARG2-polyamine 轴在"
            "人胰腺 beta 细胞和 T2D 发病中的可能作用。"
        ),
        "links": [
            ("Marselli et al., IJMS 2021", "https://pubmed.ncbi.nlm.nih.gov/34829980/"),
        ],
    },
    "PCOLCE2": {
        "label": "外部验证：人胰岛候选基因/ECM 相关",
        "summary": (
            "本队列中 PCOLCE2 在 T2D 中上调。外部人胰岛研究报道 PCOLCE2 可影响"
            "胰岛素分泌；其胶原/细胞外基质属性也与本项目富集出的 ECM 信号一致。"
        ),
        "links": [
            ("Bacos et al., JCI 2023", "https://www.jci.org/articles/view/163612"),
            ("Human Protein Atlas: PCOLCE2", "https://www.proteinatlas.org/ENSG00000163710-PCOLCE2"),
        ],
    },
    "CLTRN": {
        "label": "外部验证：TMEM27/collectrin 胰岛功能相关",
        "summary": (
            "CLTRN 又称 TMEM27/collectrin。本队列中该基因在 T2D 中下调；既往研究显示"
            "TMEM27 与 beta 细胞增殖、胰岛素分泌和 beta 细胞量标志物有关。"
        ),
        "links": [
            ("Akpinar et al., Cell Metabolism 2005", "https://pubmed.ncbi.nlm.nih.gov/16330324/"),
            ("Altirriba et al., Diabetologia 2010", "https://pubmed.ncbi.nlm.nih.gov/20386877/"),
        ],
    },
    "IAPP": {
        "label": "外部验证：T2D 胰岛淀粉样病变相关",
        "summary": (
            "IAPP/amylin 是 T2D 胰岛淀粉样沉积的经典组成成分。本项目中 IAPP 为 T2D 下调高效应 DEG，"
            "适合作为 beta 细胞压力和胰岛病理背景候选基因。"
        ),
        "links": [
            ("Westermark et al., Physiol Rev 2011", "https://pubmed.ncbi.nlm.nih.gov/21742788/"),
            ("Jurgens et al., Diabetes 2012", "https://pubmed.ncbi.nlm.nih.gov/22847497/"),
        ],
    },
    "PPP1R1A": {
        "label": "外部验证：GLP1R-胰岛素分泌放大通路相关",
        "summary": (
            "PPP1R1A 被报道为 MafA 靶基因，可调控 GLP1R 介导的葡萄糖刺激胰岛素分泌放大。"
            "本项目中 PPP1R1A 为 T2D 下调高效应 DEG。"
        ),
        "links": [
            ("Asplund et al., Diabetes 2021", "https://pubmed.ncbi.nlm.nih.gov/33631146/"),
            ("Human islet expression study", "https://pubmed.ncbi.nlm.nih.gov/25489054/"),
        ],
    },
    "GABRA2": {
        "label": "外部验证：胰岛信号与表观遗传候选",
        "summary": (
            "GABRA2 在本项目中为 T2D 下调高效应 DEG。外部人胰岛研究提示部分表观遗传改变基因会影响"
            "线粒体功能、胰岛素分泌和 T2D，GABRA2 更适合作为 B 级探索候选。"
        ),
        "links": [
            ("Human islet epigenetic alteration study", "https://pubmed.ncbi.nlm.nih.gov/38086799/"),
        ],
    },
    "PDX1": {
        "label": "外部验证：beta 细胞身份背景基因",
        "summary": (
            "PDX1 是 beta 细胞身份和功能关键转录因子。本项目中 PDX1 达到 FDR 显著但效应量不足，"
            "因此适合作为背景参考基因，高效应 DEG 证据不足。"
        ),
        "links": [
            ("Yang et al., Mol Endocrinol 2012", "https://pubmed.ncbi.nlm.nih.gov/22570331/"),
        ],
    },
    "NKX6-1": {
        "label": "外部验证：beta 细胞身份背景基因",
        "summary": (
            "NKX6.1 是维持 beta 细胞功能状态的重要转录因子。本项目中 NKX6-1 为 FDR 显著但效应量不足，"
            "更适合用于解释 beta 细胞身份背景。"
        ),
        "links": [
            ("Taylor et al., Cell Reports 2013", "https://pubmed.ncbi.nlm.nih.gov/24035389/"),
            ("NKX6.1 review", "https://pubmed.ncbi.nlm.nih.gov/33121533/"),
        ],
    },
    "SLC30A8": {
        "label": "外部验证：T2D 遗传与 beta 细胞锌转运相关",
        "summary": (
            "SLC30A8/ZnT8 是 beta 细胞锌转运和 T2D 易感性经典基因。本项目中 SLC30A8 为 FDR 显著但"
            "效应量不足，因此作为背景参考候选更合适。"
        ),
        "links": [
            ("Human islet SLC30A8 functional study", "https://pubmed.ncbi.nlm.nih.gov/20138556/"),
            ("SLC30A8 review", "https://pubmed.ncbi.nlm.nih.gov/26593983/"),
        ],
    },
}

PATHWAY_EVIDENCE = {
    "immune": {
        "label": "外部验证：T2D 胰岛炎症与 beta 细胞功能受损",
        "summary": (
            "免疫反应、炎症反应、白细胞介导免疫等条目与 T2D 胰岛炎症背景一致。"
            "这些证据支持通路方向，但 bulk RNA-seq 不能区分免疫细胞浸润和胰岛细胞内源反应。"
        ),
        "links": [
            ("Donath et al., Physiology 2009", "https://pubmed.ncbi.nlm.nih.gov/19996363/"),
            ("Eguchi & Manabe, Handb Exp Pharmacol 2022", "https://pubmed.ncbi.nlm.nih.gov/35044537/"),
        ],
    },
    "ecm": {
        "label": "外部验证：胰岛细胞外基质/纤维化相关",
        "summary": (
            "extracellular matrix、collagen、external encapsulating structure 等条目与胰岛"
            "结构重塑相符。已有研究报道 T2D 胰腺/胰岛 ECM 异常和 ECM 力学性质会影响胰岛素分泌。"
        ),
        "links": [
            ("Hayden et al., J Cardiometab Syndr 2008", "https://pubmed.ncbi.nlm.nih.gov/19040593/"),
            ("Johansen et al., Matrix Biology Plus 2024", "https://pubmed.ncbi.nlm.nih.gov/38803329/"),
        ],
    },
}

# ====================== 样式 ======================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1440px;}
    h1, h2, h3 {letter-spacing: 0;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
    }
    div[data-testid="stMetric"] label {color: #475569;}
    .note {
        border-left: 4px solid #0f766e;
        background: #f8fafc;
        padding: 0.75rem 0.95rem;
        border-radius: 0 8px 8px 0;
        color: #334155;
        line-height: 1.72;
    }
    .warn {
        border-left: 4px solid #d97706;
        background: #fffbeb;
        padding: 0.75rem 0.95rem;
        border-radius: 0 8px 8px 0;
        color: #3f3f46;
        line-height: 1.72;
    }
    .badge {
        display: inline-block;
        padding: 0.16rem 0.45rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 650;
        color: #ffffff;
        background: #0f766e;
        margin-right: 0.35rem;
    }
    .muted {color: #64748b; font-size: 0.92rem; line-height: 1.6;}
    .link-list a {margin-right: 0.7rem; white-space: nowrap;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================== 数据库连接与加载 ======================
def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner=False)
def load_table(table_name: str) -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(f"select * from {table_name}", conn)


@st.cache_data(show_spinner=False)
def load_counts() -> pd.DataFrame:
    return pd.read_csv(COUNTS_PATH, sep="\t").set_index("ensembl_id")


@st.cache_data(show_spinner=False)
def load_app_data() -> dict[str, pd.DataFrame]:
    sample = load_table("sample")
    sample_qc = load_table("sample_qc")
    deg = load_table("deg_result")
    enrichment = load_table("enrichment_result")
    annotation = load_table("gene_annotation")

    sample["in_ins_filtered_subset"] = sample["in_ins_filtered_subset"].astype(bool)
    sample_qc = sample_qc.merge(sample, on="sample_id", how="left")

    numeric_deg = ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj", "abs_log2FoldChange"]
    for col in numeric_deg:
        deg[col] = pd.to_numeric(deg[col], errors="coerce")
    for col in ["is_fdr_significant", "is_high_effect_deg", "is_annotated"]:
        deg[col] = deg[col].astype(bool)
    deg = deg[(deg["contrast"] == MAIN_CONTRAST) & (deg["subset"] == MAIN_SUBSET)].copy()

    numeric_enrich = [
        "p_value",
        "query_gene_count",
        "term_size",
        "query_size",
        "intersection_size",
        "precision",
        "recall",
    ]
    for col in numeric_enrich:
        enrichment[col] = pd.to_numeric(enrichment[col], errors="coerce")
    enrichment["significant"] = enrichment["significant"].astype(bool)
    enrichment = enrichment[
        (enrichment["contrast"] == MAIN_CONTRAST) & (enrichment["subset"] == MAIN_SUBSET)
    ].copy()

    return {
        "sample": sample,
        "sample_qc": sample_qc,
        "deg": deg,
        "enrichment": enrichment,
        "annotation": annotation,
    }


# ====================== 辅助函数 ======================
def format_int(value: int | float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{int(value):,}"


def format_p(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if value == 0:
        return "0"
    return f"{value:.3e}" if abs(value) < 0.001 else f"{value:.4f}"


def note(text: str, warning: bool = False) -> None:
    klass = "warn" if warning else "note"
    st.markdown(f"<div class='{klass}'>{html.escape(text)}</div>", unsafe_allow_html=True)


def evidence_links_html(links: list[tuple[str, str]]) -> str:
    if not links:
        return ""
    anchors = [
        f"<a href='{html.escape(url)}' target='_blank'>{html.escape(label)}</a>"
        for label, url in links
    ]
    return "<div class='link-list'>" + " ".join(anchors) + "</div>"


def canonical_symbol(symbol: str | float | None) -> str:
    if symbol is None or pd.isna(symbol):
        return ""
    upper = str(symbol).strip().upper()
    return GENE_ALIASES.get(upper, upper)


def _link_label(url: str) -> str:
    pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if pmid_match:
        return f"PMID {pmid_match.group(1)}"
    if "jci.org" in url:
        return "JCI article"
    return "外部链接"


@st.cache_data(show_spinner=False)
def load_candidate_evidence() -> dict[str, dict[str, object]]:
    if not CANDIDATE_EVIDENCE_PATH.exists():
        return {}
    evidence_df = pd.read_csv(CANDIDATE_EVIDENCE_PATH)
    evidence: dict[str, dict[str, object]] = {}
    for row in evidence_df.to_dict(orient="records"):
        symbol = canonical_symbol(row.get("gene_symbol"))
        links = []
        for url in str(row.get("primary_links", "")).split(";"):
            url = url.strip()
            if url:
                links.append((_link_label(url), url))
        tier = str(row.get("evidence_tier", "")).strip()
        used = str(row.get("used_in_model", "")).strip().lower() == "yes"
        model_note = str(row.get("model_note", "")).strip()
        project_result = str(row.get("project_result", "")).strip()
        label = f"{tier}级证据" if tier else "文献证据"
        if used:
            label += " / 模型特征"
        evidence[symbol] = {
            "label": label,
            "summary": (
                f"{row.get('evidence_summary', '')} 项目结果：{project_result}。"
                f"模型使用情况：{'已纳入' if used else '未纳入'}"
                f"{'，' + model_note if model_note else ''}。"
            ),
            "links": links,
            "used_in_model": used,
            "evidence_tier": tier,
            "project_result": project_result,
        }
    return evidence


def gene_evidence(symbol: str | float | None) -> dict[str, object] | None:
    return load_candidate_evidence().get(canonical_symbol(symbol))


def gene_pubmed_search_url(symbol: str) -> str:
    query = quote_plus(f"{symbol} type 2 diabetes pancreatic islet")
    return f"https://pubmed.ncbi.nlm.nih.gov/?term={query}"


def gene_evidence_or_search_url(symbol: str) -> str:
    evidence = gene_evidence(symbol)
    if evidence:
        links = evidence.get("links", [])
        if links:
            return str(links[0][1])
    return gene_pubmed_search_url(symbol)


def expression_atlas_url(symbol: str) -> str:
    query = quote_plus(symbol)
    return f"https://www.ebi.ac.uk/gxa/search?geneQuery={query}"


def ncbi_gene_url(entrezgene: float | int | str | None) -> str | None:
    if entrezgene is None or pd.isna(entrezgene):
        return None
    try:
        return f"https://www.ncbi.nlm.nih.gov/gene/{int(float(entrezgene))}"
    except (TypeError, ValueError):
        return None


def term_url(source: str, native: str) -> str | None:
    if not isinstance(native, str) or not native:
        return None
    if native.startswith("GO:"):
        return f"https://amigo.geneontology.org/amigo/term/{native}"
    if source == "REAC":
        reactome_id = native.replace("REAC:", "")
        return f"https://reactome.org/content/detail/{reactome_id}"
    if source == "KEGG":
        kegg_id = native.replace("KEGG:", "")
        return f"https://www.kegg.jp/entry/{kegg_id}"
    return None


def annotate_deg_category(df: pd.DataFrame, fdr_cutoff: float = 0.05, log2fc_cutoff: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    conditions = [
        (out["padj"] <= fdr_cutoff) & (out["log2FoldChange"] >= log2fc_cutoff),
        (out["padj"] <= fdr_cutoff) & (out["log2FoldChange"] <= -log2fc_cutoff),
        (out["padj"] <= fdr_cutoff),
    ]
    choices = ["T2D 上调高效应", "T2D 下调高效应", "FDR 显著但效应较小"]
    out["category"] = np.select(conditions, choices, default="未显著")
    out["display_gene"] = np.where(
        out["gene_symbol"].notna() & (out["gene_symbol"].astype(str) != ""),
        out["gene_symbol"].astype(str),
        out["ensembl_id"].astype(str),
    )
    positive_padj = out.loc[out["padj"] > 0, "padj"]
    min_positive = positive_padj.min() if not positive_padj.empty else 1e-300
    out["padj_for_plot"] = out["padj"].fillna(1).clip(lower=min_positive)
    out["neg_log10_padj"] = -np.log10(out["padj_for_plot"])
    out["evidence"] = out["display_gene"].map(
        lambda s: "外部验证" if gene_evidence(s) else "探索性"
    )
    out["pubmed_search"] = out["display_gene"].map(gene_evidence_or_search_url)
    return out


def gene_search_table(deg: pd.DataFrame) -> pd.DataFrame:
    table = annotate_deg_category(deg)
    table = table.sort_values(["padj", "abs_log2FoldChange"], ascending=[True, False])
    return table[
        [
            "display_gene",
            "ensembl_id",
            "gene_name",
            "baseMean",
            "log2FoldChange",
            "padj",
            "category",
            "evidence",
            "pubmed_search",
        ]
    ].rename(
        columns={
            "display_gene": "gene_symbol",
            "gene_name": "gene_name",
            "baseMean": "baseMean",
            "log2FoldChange": "log2FC",
            "padj": "FDR",
            "category": "分类",
            "evidence": "证据",
            "pubmed_search": "PubMed",
        }
    )


# ====================== 图表绘制函数 ======================
def plot_status_counts(sample: pd.DataFrame) -> go.Figure:
    counts = (
        sample["diabetes_status"]
        .value_counts()
        .reindex(STATUS_ORDER)
        .dropna()
        .astype(int)
        .rename_axis("diabetes_status")
        .reset_index(name="sample_count")
    )
    fig = px.bar(
        counts,
        x="diabetes_status",
        y="sample_count",
        color="diabetes_status",
        color_discrete_map=STATUS_COLORS,
        text="sample_count",
        labels={"diabetes_status": "疾病状态", "sample_count": "样本数"},
    )
    fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=20, b=10))
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def plot_volcano(plot_df: pd.DataFrame, fdr_cutoff: float, log2fc_cutoff: float) -> go.Figure:
    fig = px.scatter(
        plot_df,
        x="log2FoldChange",
        y="neg_log10_padj",
        color="category",
        color_discrete_map=DEG_COLORS,
        category_orders={"category": list(DEG_COLORS)},
        hover_name="display_gene",
        hover_data={
            "ensembl_id": True,
            "baseMean": ":.2f",
            "log2FoldChange": ":.3f",
            "padj": ":.3e",
            "evidence": True,
            "category": True,
            "neg_log10_padj": False,
        },
        labels={
            "log2FoldChange": "log2 Fold Change (T2D vs ND)",
            "neg_log10_padj": "-log10(FDR)",
            "category": "分类",
        },
        render_mode="webgl",
    )
    fig.add_vline(x=log2fc_cutoff, line_dash="dash", line_color="#64748b")
    fig.add_vline(x=-log2fc_cutoff, line_dash="dash", line_color="#64748b")
    fig.add_hline(y=-math.log10(fdr_cutoff), line_dash="dash", line_color="#64748b")
    label_df = plot_df[
        plot_df["display_gene"].isin(MODEL_FEATURES + ["TXNIP"])
    ].copy()
    fig.add_trace(
        go.Scatter(
            x=label_df["log2FoldChange"],
            y=label_df["neg_log10_padj"],
            mode="text",
            text=label_df["display_gene"],
            textposition="top center",
            textfont=dict(size=11, color="#111827"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_traces(marker=dict(size=6, opacity=0.72))
    fig.update_layout(height=610, margin=dict(l=10, r=10, t=25, b=10), legend_title=None)
    return fig


def plot_gene_expression(expr: pd.DataFrame, gene_label: str) -> go.Figure:
    fig = px.box(
        expr,
        x="diabetes_status",
        y="log2_normalized_count",
        color="diabetes_status",
        points="all",
        category_orders={"diabetes_status": STATUS_ORDER},
        color_discrete_map=STATUS_COLORS,
        hover_data=["sample_id", "gsm_id"],
        labels={
            "diabetes_status": "疾病状态",
            "log2_normalized_count": "log2 normalized counts",
        },
        title=gene_label,
    )
    fig.update_layout(showlegend=False, height=520, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def plot_enrichment(df: pd.DataFrame) -> go.Figure:
    plot_df = df[df["p_value"].notna() & (df["p_value"] > 0)].copy()
    plot_df = plot_df.sort_values("p_value").head(20)
    plot_df["neg_log10_p"] = -np.log10(plot_df["p_value"])
    plot_df["term_label"] = plot_df["name"].str.slice(0, 72)
    fig = px.bar(
        plot_df.sort_values("neg_log10_p"),
        x="neg_log10_p",
        y="term_label",
        color="source",
        orientation="h",
        hover_data={
            "native": True,
            "name": True,
            "p_value": ":.3e",
            "intersection_size": True,
            "query_gene_count": True,
            "term_label": False,
            "neg_log10_p": ":.2f",
        },
        labels={"neg_log10_p": "-log10(p value)", "term_label": "富集条目"},
        color_discrete_map={
            "GO:BP": "#0f766e",
            "GO:CC": "#2563eb",
            "GO:MF": "#7c3aed",
            "KEGG": "#b45309",
            "REAC": "#dc2626",
        },
    )
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=25, b=10), legend_title=None)
    return fig


def pathway_evidence(name: str, source: str) -> tuple[dict[str, object] | None, bool]:
    lower = name.lower()
    immune_terms = ["immune", "inflamm", "leukocyte", "lymphocyte", "humoral", "antigen", "cytokine"]
    ecm_terms = ["extracellular matrix", "collagen", "external encapsulating", "basement membrane"]
    if any(term in lower for term in ecm_terms):
        return PATHWAY_EVIDENCE["ecm"], False
    if any(term in lower for term in immune_terms):
        return PATHWAY_EVIDENCE["immune"], False
    if source == "KEGG" and "infection" in lower:
        return {
            "label": "谨慎解释：感染命名通常来自共享免疫/补体基因",
            "summary": (
                "感染相关 KEGG term 不应解释为样本存在感染。它更可能反映候选基因与免疫、补体"
                "或炎症模块有统计重叠。"
            ),
            "links": [],
        }, True
    return None, False


def render_evidence_box(evidence: dict[str, object] | None, warning: bool = False) -> None:
    if evidence is None:
        note("暂未为该条目配置直接外部验证。它仍可作为本队列中的探索性结果，但需要独立数据或实验继续确认。", warning=True)
        return
    klass = "warn" if warning else "note"
    label = html.escape(str(evidence["label"]))
    summary = html.escape(str(evidence["summary"]))
    links = evidence_links_html(evidence.get("links", []))
    st.markdown(
        f"<div class='{klass}'><span class='badge'>{label}</span>{summary}<br>{links}</div>",
        unsafe_allow_html=True,
    )


# ====================== 子页面函数 ======================
def overview_page(data: dict[str, pd.DataFrame]) -> None:
    sample = data["sample"]
    deg = data["deg"]
    enrichment = data["enrichment"]
    high_effect = deg[deg["is_high_effect_deg"]]
    fdr_sig = deg[deg["is_fdr_significant"]]

    st.title("T2D 胰岛转录组科研检索平台")
    metrics = st.columns(5)
    metrics[0].metric("样本", format_int(len(sample)))
    metrics[1].metric("T2D / ND", f"{(sample['diabetes_status'] == 'T2D').sum()} / {(sample['diabetes_status'] == 'ND').sum()}")
    metrics[2].metric("测试基因", format_int(len(deg)))
    metrics[3].metric("高效应 DEG", format_int(len(high_effect)))
    metrics[4].metric("富集条目", format_int(len(enrichment)))

    left, right = st.columns([1.05, 1.35])
    with left:
        st.subheader("队列结构")
        st.plotly_chart(plot_status_counts(sample), width="stretch")
    with right:
        st.subheader("优先候选基因")
        candidate = gene_search_table(deg)
        candidate = candidate[candidate["证据"] == "外部验证"].head(12)
        st.dataframe(
            candidate[["gene_symbol", "gene_name", "log2FC", "FDR", "分类", "PubMed"]],
            width="stretch",
            height=330,
            column_config={
                "PubMed": st.column_config.LinkColumn("PubMed", display_text="检索"),
                "log2FC": st.column_config.NumberColumn("log2FC", format="%.3f"),
                "FDR": st.column_config.NumberColumn("FDR", format="%.3e"),
            },
        )

    note(
        "平台的核心用途是围绕 GSE164416 胰岛 RNA-seq 数据进行候选基因、通路条目和表达风险评分检索。"
        "Day2 中没有形成可靠疾病分离证据的降维图不再放入主界面，避免把负结果包装成检索功能。"
    )

    st.subheader("当前主比较概览")
    col1, col2, col3 = st.columns(3)
    col1.metric("FDR 显著基因", format_int(len(fdr_sig)))
    col2.metric("T2D 上调高效应", format_int(((high_effect["regulation"] == "up")).sum()))
    col3.metric("T2D 下调高效应", format_int(((high_effect["regulation"] == "down")).sum()))

    top_terms = enrichment[
        enrichment["query_name"].isin(["all_high_effect_deg", "up_high_effect_deg"])
    ].sort_values("p_value").head(10)
    term_table = top_terms[["source", "native", "name", "p_value", "intersection_size"]].rename(
        columns={"source": "来源", "native": "ID", "name": "通路/条目", "p_value": "p value", "intersection_size": "交集基因数"}
    )
    st.dataframe(
        term_table,
        width="stretch",
        height=310,
        column_config={"p value": st.column_config.NumberColumn("p value", format="%.3e")},
    )


def candidate_discovery_page(data: dict[str, pd.DataFrame]) -> None:
    deg = data["deg"]
    st.title("候选基因筛选")

    controls = st.columns([0.85, 0.85, 1.1, 1.2])
    with controls[0]:
        fdr_cutoff = st.number_input("FDR", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
    with controls[1]:
        log2fc_cutoff = st.number_input("|log2FC|", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
    plot_df = annotate_deg_category(deg, fdr_cutoff, log2fc_cutoff)
    with controls[2]:
        categories = st.multiselect("分类", list(DEG_COLORS), default=list(DEG_COLORS)[:3])
    with controls[3]:
        search = st.text_input("基因/描述检索", value="")

    filtered = plot_df[plot_df["category"].isin(categories)].copy()
    if search.strip():
        pattern = re.escape(search.strip())
        blob = (
            filtered["display_gene"].fillna("")
            + " "
            + filtered["ensembl_id"].fillna("")
            + " "
            + filtered["gene_name"].fillna("")
        )
        filtered = filtered[blob.str.contains(pattern, case=False, regex=True)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("当前结果", format_int(len(filtered)))
    m2.metric("上调高效应", format_int((plot_df["category"] == "T2D 上调高效应").sum()))
    m3.metric("下调高效应", format_int((plot_df["category"] == "T2D 下调高效应").sum()))
    m4.metric("外部验证基因", format_int((filtered["evidence"] == "外部验证").sum()))

    tab1, tab2 = st.tabs(["全局图谱", "可检索表格"])
    with tab1:
        st.plotly_chart(plot_volcano(plot_df, fdr_cutoff, log2fc_cutoff), width="stretch")
        note(
            "主比较中高效应 DEG 以上调为主，说明 T2D 胰岛样本的主要信号更偏向炎症、细胞外基质和组织重塑相关模块；"
            "同时，SLC2A2、ARG2、CLTRN 等下调基因仍保留了明确的胰岛功能解释价值。"
        )
    with tab2:
        table = gene_search_table(filtered)
        st.dataframe(
            table,
            width="stretch",
            height=560,
            column_config={
                "PubMed": st.column_config.LinkColumn("PubMed", display_text="检索"),
                "baseMean": st.column_config.NumberColumn("baseMean", format="%.2f"),
                "log2FC": st.column_config.NumberColumn("log2FC", format="%.3f"),
                "FDR": st.column_config.NumberColumn("FDR", format="%.3e"),
            },
        )
        st.download_button(
            "下载当前筛选结果",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name="t2d_candidate_genes.csv",
            mime="text/csv",
        )


# 其余子页面函数（gene_query_page, pathway_query_page, risk_prediction_page）保持原样
def gene_query_page(data: dict[str, pd.DataFrame]) -> None:
    sample = data["sample"]
    deg = data["deg"]
    annotation = data["annotation"]
    counts = load_counts()

    st.title("基因检索")
    options_df = gene_options(annotation, deg)
    search_text = st.text_input("Gene symbol、别名、Ensembl ID 或基因描述", value="SLC2A2")
    matches = resolve_gene(search_text, options_df)
    if matches.empty:
        st.warning("没有在本地表达矩阵和注释表中找到匹配基因。")
        return

    match_labels = matches["display_gene"].head(80).tolist()
    selected_display = st.selectbox("匹配结果", match_labels)
    selected = matches[matches["display_gene"] == selected_display].iloc[0]
    ensembl_id = str(selected["ensembl_id"])
    symbol = selected["gene_symbol"] if selected["gene_symbol"] else ensembl_id
    gene_label = f"{symbol} ({ensembl_id})"

    gene_deg = deg[deg["ensembl_id"] == ensembl_id]
    if gene_deg.empty or ensembl_id not in counts.index:
        st.warning("该基因缺少表达矩阵或差异表达统计。")
        return
    row = gene_deg.iloc[0]
    expr = expression_for_gene(ensembl_id, counts, sample)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("log2FC", f"{row['log2FoldChange']:.3f}")
    col2.metric("FDR", format_p(row["padj"]))
    col3.metric("baseMean", f"{row['baseMean']:.2f}")
    col4.metric("方向", "T2D 上调" if row["log2FoldChange"] > 0 else "T2D 下调")

    links = [
        ("PubMed 检索", gene_pubmed_search_url(symbol)),
        ("Expression Atlas", expression_atlas_url(symbol)),
    ]
    ncbi_url = ncbi_gene_url(row.get("entrezgene"))
    if ncbi_url:
        links.insert(0, ("NCBI Gene", ncbi_url))
    st.markdown(evidence_links_html(links), unsafe_allow_html=True)

    render_evidence_box(gene_evidence(symbol))

    st.plotly_chart(plot_gene_expression(expr, gene_label), width="stretch")

    stats = (
        expr.groupby("diabetes_status", observed=True)["log2_normalized_count"]
        .agg(["count", "mean", "median", "std"])
        .reindex(STATUS_ORDER)
        .dropna(how="all")
        .reset_index()
    )
    left, right = st.columns([1.05, 1.25])
    with left:
        st.subheader("分组表达摘要")
        st.dataframe(
            stats,
            width="stretch",
            height=260,
            column_config={
                "mean": st.column_config.NumberColumn("mean", format="%.3f"),
                "median": st.column_config.NumberColumn("median", format="%.3f"),
                "std": st.column_config.NumberColumn("std", format="%.3f"),
            },
        )
    with right:
        st.subheader("同方向候选")
        same_direction = deg[
            (deg["regulation"] == row["regulation"])
            & (deg["is_high_effect_deg"])
            & (deg["ensembl_id"] != ensembl_id)
        ].copy()
        same_direction = gene_search_table(same_direction).head(8)
        st.dataframe(
            same_direction[["gene_symbol", "gene_name", "log2FC", "FDR", "证据", "PubMed"]],
            width="stretch",
            height=260,
            column_config={
                "PubMed": st.column_config.LinkColumn("PubMed", display_text="检索"),
                "log2FC": st.column_config.NumberColumn("log2FC", format="%.3f"),
                "FDR": st.column_config.NumberColumn("FDR", format="%.3e"),
            },
        )

    if bool(row["is_high_effect_deg"]):
        direction = "升高" if row["log2FoldChange"] > 0 else "降低"
        note(
            f"{gene_label} 在 T2D vs ND 主比较中达到高效应差异标准，T2D 中表达{direction}。"
            "箱线图用于检查这种差异是否由多数样本共同贡献，并评估少数离群点对结果的影响。"
        )
    elif bool(row["is_fdr_significant"]):
        note("该基因达到 FDR 显著，但效应量没有达到默认高效应阈值，适合保留为背景信号而非首要候选。", warning=True)
    else:
        note("该基因在主比较中未达到 FDR 显著，当前检索结果只能作为表达分布参考。", warning=True)


def pathway_query_page(data: dict[str, pd.DataFrame]) -> None:
    enrichment = data["enrichment"]
    st.title("通路检索")

    controls = st.columns([1.15, 1.15, 1.5])
    with controls[0]:
        query_name = st.selectbox(
            "候选基因集合",
            [q for q in QUERY_LABELS if q in set(enrichment["query_name"])],
            format_func=lambda q: QUERY_LABELS.get(q, q),
            index=0,
        )
    with controls[1]:
        sources = sorted(enrichment["source"].dropna().unique().tolist())
        selected_sources = st.multiselect("数据库来源", sources, default=sources)
    with controls[2]:
        term_search = st.text_input("通路/GO/KEGG/Reactome 检索", value="immune")

    filtered = enrichment[
        (enrichment["query_name"] == query_name)
        & (enrichment["source"].isin(selected_sources))
        & (enrichment["significant"])
    ].copy()
    if term_search.strip():
        pattern = re.escape(term_search.strip())
        blob = filtered["name"].fillna("") + " " + filtered["native"].fillna("") + " " + filtered["description"].fillna("")
        filtered = filtered[blob.str.contains(pattern, case=False, regex=True)]

    if filtered.empty:
        st.warning("当前筛选条件下没有显著富集条目。")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("匹配条目", format_int(len(filtered)))
    m2.metric("最小 p value", format_p(filtered["p_value"].min()))
    m3.metric("最大交集基因数", format_int(filtered["intersection_size"].max()))

    st.plotly_chart(plot_enrichment(filtered), width="stretch")

    ranked = filtered.sort_values("p_value").copy()
    ranked["label"] = ranked["source"] + " | " + ranked["native"] + " | " + ranked["name"]
    selected_label = st.selectbox("查看条目", ranked["label"].head(80).tolist())
    term_row = ranked[ranked["label"] == selected_label].iloc[0]

    url = term_url(str(term_row["source"]), str(term_row["native"]))
    term_cols = st.columns(4)
    term_cols[0].metric("来源", str(term_row["source"]))
    term_cols[1].metric("p value", format_p(term_row["p_value"]))
    term_cols[2].metric("交集基因数", format_int(term_row["intersection_size"]))
    term_cols[3].metric("term size", format_int(term_row["term_size"]))
    if url:
        st.link_button("打开外部数据库条目", url)

    ev, warn = pathway_evidence(str(term_row["name"]), str(term_row["source"]))
    render_evidence_box(ev, warning=warn)

    table = ranked[["source", "native", "name", "p_value", "query_gene_count", "intersection_size", "term_size"]].rename(
        columns={
            "source": "来源",
            "native": "ID",
            "name": "通路/条目",
            "p_value": "p value",
            "query_gene_count": "输入基因数",
            "intersection_size": "交集基因数",
            "term_size": "term size",
        }
    )
    st.dataframe(
        table,
        width="stretch",
        height=460,
        column_config={"p value": st.column_config.NumberColumn("p value", format="%.3e")},
    )


def risk_prediction_page() -> None:
    st.title("表达风险预测")
    outputs = load_multicohort_model_outputs()
    model = train_risk_model()
    pipe = model["pipeline"]
    feature_stats = model["feature_stats"]
    sample_values = model["sample_values"]

    threshold = model_metrics_panel(outputs)

    note(
        "该模块输出的是转录组表达模式评分，用于科研探索和候选基因解释。"
        "它依赖胰岛或 beta 细胞表达数据，不能直接换算为临床患病概率。",
        warning=True,
    )

    st.subheader("训练效果可视化")
    fig_cols = st.columns(2)
    with fig_cols[0]:
        if MODEL_FIGURES["sample_counts"].exists():
            st.image(str(MODEL_FIGURES["sample_counts"]), caption="多队列训练样本组成")
        if MODEL_FIGURES["confusion"].exists():
            st.image(str(MODEL_FIGURES["confusion"]), caption="训练和外部测试混淆矩阵")
    with fig_cols[1]:
        if MODEL_FIGURES["roc"].exists():
            st.image(str(MODEL_FIGURES["roc"]), caption="5 折、留一队列和外部测试 ROC 曲线")
        if MODEL_FIGURES["coefficients"].exists():
            st.image(str(MODEL_FIGURES["coefficients"]), caption="14 个候选基因模型系数")

    if MODEL_FIGURES["probability"].exists():
        st.image(str(MODEL_FIGURES["probability"]), caption="训练队列和外部测试样本的预测概率分布")

    sample_options = ["使用 ND 中位数", "使用 T2D 中位数", "使用总体中位数"] + (
        sample_values["sample_id"].astype(str)
        + " | "
        + sample_values["external_dataset"].astype(str)
        + " | "
        + sample_values["diabetes_status"].astype(str)
    ).tolist()
    selected_sample = st.selectbox("载入数据库样本作为输入模板", sample_options)
    if selected_sample == "使用 ND 中位数":
        defaults = sample_values[sample_values["diabetes_status"] == "ND"][MODEL_FEATURES].median().to_dict()
    elif selected_sample == "使用 T2D 中位数":
        defaults = sample_values[sample_values["diabetes_status"] == "T2D"][MODEL_FEATURES].median().to_dict()
    elif selected_sample == "使用总体中位数":
        defaults = sample_values[MODEL_FEATURES].median().to_dict()
    else:
        selected_id = selected_sample.split(" | ")[0]
        defaults = sample_values[sample_values["sample_id"] == selected_id].iloc[0][MODEL_FEATURES].to_dict()

    values: dict[str, float] = {}
    slider_cols = st.columns(3)
    stats_by_gene = feature_stats.set_index("gene")
    for idx, gene in enumerate(MODEL_FEATURES):
        stats = stats_by_gene.loc[gene]
        with slider_cols[idx % 3]:
            values[gene] = st.slider(
                gene,
                min_value=float(stats["min"]),
                max_value=float(stats["max"]),
                value=float(defaults[gene]),
                step=0.01,
            )

    input_df = pd.DataFrame([values], columns=MODEL_FEATURES)
    probability = float(pipe.predict_proba(input_df)[0, 1])
    label = "高于训练阈值" if probability >= threshold else "低于训练阈值"
    st.metric("T2D 表达模式评分", f"{probability:.1%}", help="基于多队列标准化表达模型，不代表临床患病概率。")
    note(f"按照当前输入，模型评分为“{label}”。建议结合基因检索页面查看每个特征在 GSE164416 中的表达分布。")

    scaler = pipe.named_steps["standardscaler"]
    clf = pipe.named_steps["logisticregression"]
    z_values = scaler.transform(input_df)[0]
    contributions = pd.DataFrame(
        {
            "gene": MODEL_FEATURES,
            "input": [values[g] for g in MODEL_FEATURES],
            "coefficient": clf.coef_[0],
            "contribution": z_values * clf.coef_[0],
        }
    ).sort_values("contribution")
    fig = px.bar(
        contributions,
        x="contribution",
        y="gene",
        orientation="h",
        color="contribution",
        color_continuous_scale=["#2563eb", "#f8fafc", "#dc2626"],
        labels={"contribution": "对 T2D 评分的贡献", "gene": "基因"},
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        contributions.sort_values("gene"),
        width="stretch",
        height=420,
        column_config={
            "input": st.column_config.NumberColumn("输入 z-score", format="%.3f"),
            "coefficient": st.column_config.NumberColumn("模型系数", format="%.3f"),
            "contribution": st.column_config.NumberColumn("贡献", format="%.3f"),
        },
    )

    coeff = outputs["feature_coefficients"]
    dataset_summary = outputs["dataset_summary"]
    table_cols = st.columns([0.9, 1.1])
    with table_cols[0]:
        st.subheader("训练队列样本数")
        st.dataframe(dataset_summary, width="stretch", height=180)
    with table_cols[1]:
        st.subheader("模型系数表")
        st.dataframe(
            coeff,
            width="stretch",
            height=240,
            column_config={"coefficient": st.column_config.NumberColumn("coefficient", format="%.3f")},
        )

    external_validation_panel(outputs, threshold)


# ====================== 主入口（已去除侧边栏） ======================
def main() -> None:
    data = load_app_data()

    # 使用 radio 替代侧边栏
    page = st.radio(
        "功能模块",
        ["数据库概览", "候选基因筛选", "基因检索", "通路检索", "表达风险预测"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "数据库概览":
        overview_page(data)
    elif page == "候选基因筛选":
        candidate_discovery_page(data)
    elif page == "基因检索":
        gene_query_page(data)
    elif page == "通路检索":
        pathway_query_page(data)
    else:
        risk_prediction_page()


if __name__ == "__main__":
    main()