import io

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from microbiome_src.fiber_response.loader import load_fiber_features, load_fiber_sources
from microbiome_src.fiber_response.scoring import compute_match_summary
from microbiome_src.fiber_response.taxonomy_parser import parse_taxonomy_string
from microbiome_src.reporting.export import export_csv, make_export_filename


def show():
    st.title("膳食纤维响应证据匹配")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "上传与兼容性", "匹配结果", "特征详情", "导出报告", "研究依据"
    ])

    features = load_fiber_features()
    sources = load_fiber_sources()

    with tab1:
        user_taxa, user_abund = _tab_upload_and_compatibility()

    with tab2:
        if user_taxa is not None:
            _tab_match_results(user_taxa, features, user_abund)

    with tab3:
        if user_taxa is not None:
            _tab_feature_details(user_taxa, features, user_abund)

    with tab4:
        if user_taxa is not None:
            _tab_export(user_taxa, features, user_abund)

    with tab5:
        _tab_study_background(features, sources)

    st.caption(
        "⚠️ 免责声明：本模块为论文特征证据匹配工具，不是临床响应预测模型。"
        "taxonomy 层级匹配不能替代原论文 ASV 及 LightGBM 模型。"
        "不应据此单独决定患者饮食方案。"
    )


def _tab_study_background(features, sources):
    st.markdown("""
    ### Song et al. 2025 — 肠道微生物组预测糖尿病前期个体对膳食纤维的响应

    **研究设计**：随机开放标签临床试验，802 名中国东部糖尿病前期受试者，
    接受 6 个月膳食纤维补充干预。

    **预测模型**：
    - 输入：基线 44 个 ASV 丰度
    - 模型：LightGBM
    - 验证：分层 10 折交叉验证
    - AUC：0.81 (95% CI 0.74–0.88)

    **高低响应定义**（6 个月内改善的指标计分）：
    - FPG（空腹血糖）
    - PBG（餐后血糖）
    - HbA1c（糖化血红蛋白）
    - 低响应者：0–1 分改善
    - 高响应者：2–3 分改善

    **MFS（微生物纤维评分）**：基于个体 SHAP 值构建，MFS ≤ 17 倾向低响应，MFS ≥ 25 倾向高响应。

    ⚠️ **重要限制**：本模块不计算 MFS。MFS 依赖原训练模型产生的个体 SHAP 值，
    不能用"某菌高或低"简单替代。
    """)

    with st.expander("论文 44 特征一览"):
        df = pd.DataFrame([
            {"ID": f.feature_id, "Taxonomy": f.taxonomy,
             "方向": f.published_direction, "Importance": f.importance}
            for f in sorted(features, key=lambda x: x.importance, reverse=True)
        ])
        st.dataframe(df, use_container_width=True, height=400)

    st.markdown("### 支持性研究")
    for src in sources:
        with st.expander(f"{src['citation']} ({src['year']})"):
            st.markdown(f"**人群**: {src['population']}")
            st.markdown(f"**干预**: {src['intervention']}")
            st.markdown(f"**用途**: {src['main_use']}")
            st.markdown(f"**限制**: {src['limitations']}")
            links = []
            if src.get("pmid"):
                links.append(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{src['pmid']}/)")
            if src.get("doi"):
                links.append(f"[DOI](https://doi.org/{src['doi']})")
            if links:
                st.markdown(" | ".join(links))


def _tab_upload_and_compatibility():
    st.markdown("### 上传 taxonomy 表")
    st.markdown(
        "taxonomy 表将用户特征 ID 映射到分类注释。"
        "支持 QIIME2 风格（`k__;p__;c__;o__;f__;g__;s__`）或简单 taxonomy 字符串。"
        "若已通过主页面载入数据，可复用丰度表列名。"
    )

    use_existing = False
    if "dataset" in st.session_state and st.session_state.dataset is not None:
        use_existing = st.checkbox("使用主应用已载入数据集的 taxa 列名", value=True)
        if use_existing:
            ds = st.session_state.dataset
            taxa = [str(t) for t in ds.abundance.columns.tolist()]
            st.success(f"已识别 {len(taxa)} 个 taxa")
            return taxa, ds.abundance.mean(axis=0)

    tax_file = st.file_uploader("上传 taxonomy 表 (CSV/TSV)", type=["csv", "tsv", "txt"])
    abund_file = st.file_uploader("上传丰度表 (CSV/TSV，可选)", type=["csv", "tsv", "txt"])

    if tax_file is None:
        st.info("请上传 taxonomy 表，或勾选复用已有数据。")
        return None, None

    tax_df = pd.read_csv(tax_file, sep=None, engine="python")
    st.write(f"taxonomy 表: {tax_df.shape[0]} 行 × {tax_df.shape[1]} 列")

    feature_id_col = st.selectbox("特征 ID 列", tax_df.columns.tolist(), index=0)
    taxonomy_col = st.selectbox("taxonomy 列", tax_df.columns.tolist(), index=min(1, len(tax_df.columns) - 1))

    taxa = tax_df[feature_id_col].astype(str).tolist()
    st.info(f"共识别 {len(taxa)} 个特征")

    abund_series = None
    if abund_file is not None:
        abund_df = pd.read_csv(abund_file, sep=None, engine="python")
        if abund_df.shape[1] >= 2:
            abund_series = abund_df.iloc[:, 1]
            st.success("丰度表已载入")

    return taxa, abund_series


def _tab_match_results(user_taxa, features, user_abund):
    st.markdown("### 匹配结果总览")

    summary = compute_match_summary(user_taxa, features, user_abund)

    cols = st.columns(4)
    cols[0].metric("匹配特征", f"{summary.matched_features}/{summary.total_paper_features}")
    cols[1].metric("重要性加权覆盖率", f"{summary.importance_weighted_coverage:.1%}")
    cols[2].metric("高响应关联", f"{summary.high_response_weight:.3f}")
    cols[3].metric("低响应关联", f"{summary.low_response_weight:.3f}")

    st.markdown("#### 匹配层级分布")
    level_data = {
        "精确物种": summary.exact_species_count,
        "属级匹配": summary.exact_genus_count,
        "命名组": summary.named_group_count,
        "科级匹配": summary.family_count,
    }
    fig = px.bar(x=list(level_data.keys()), y=list(level_data.values()),
                 labels={"x": "匹配层级", "y": "特征数"},
                 color=list(level_data.keys()),
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=10, b=10), height=300)
    st.plotly_chart(fig, use_container_width=True)

    if summary.conflict_count > 0:
        st.warning(f"⚠️ 检测到 {summary.conflict_count} 个用户 taxon 匹配了方向相反的论文 ASV。")

    st.markdown("#### 高/低响应证据权重")
    fig2 = go.Figure(data=[
        go.Bar(name="高响应关联", x=["证据权重"], y=[summary.high_response_weight],
               marker_color="#2ecc71"),
        go.Bar(name="低响应关联", x=["证据权重"], y=[summary.low_response_weight],
               marker_color="#e74c3c"),
    ])
    fig2.update_layout(barmode="group", margin=dict(l=20, r=20, t=10, b=10), height=250,
                       yaxis_title="归一化加权证据")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("这是本平台内部构建的证据展示指标，未经临床验证，不是原论文 MFS。")


def _tab_feature_details(user_taxa, features, user_abund):
    st.markdown("### 匹配特征详情")

    summary = compute_match_summary(user_taxa, features, user_abund)

    if not summary.all_matches:
        st.info("未匹配到任何论文特征。")
        return

    match_levels = list(set(m.match_level for m in summary.all_matches))
    selected_level = st.multiselect(
        "筛选匹配层级", match_levels, default=match_levels,
        format_func=lambda x: {"exact_species": "精确物种", "exact_genus": "属级匹配",
                               "named_group": "命名组", "family": "科级匹配"}.get(x, x)
    )

    filtered = [m for m in summary.all_matches if m.match_level in selected_level]

    rows = []
    for m in filtered:
        rows.append({
            "用户特征": m.user_feature,
            "论文 ASV": m.paper_feature_id,
            "论文分类": m.paper_taxonomy,
            "匹配层级": m.match_level,
            "置信度": m.confidence,
            "Importance": m.importance,
            "论文方向": m.published_direction,
            "限制": m.limitations[:80] + "..." if len(m.limitations) > 80 else m.limitations,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

    st.markdown("---")
    st.markdown("### 单特征详情")

    selected_user_taxon = st.selectbox(
        "选择用户特征查看详情",
        options=[""] + sorted(set(m.user_feature for m in summary.all_matches)),
        format_func=lambda x: "— 请选择 —" if x == "" else x,
    )

    if selected_user_taxon:
        taxon_matches = [m for m in summary.all_matches if m.user_feature == selected_user_taxon]
        for m in taxon_matches:
            with st.container(border=True):
                st.markdown(f"**论文特征**: {m.paper_feature_id}")
                st.markdown(f"**分类**: {m.paper_taxonomy}")
                st.markdown(f"**LightGBM importance**: {m.importance:.2f}")
                st.markdown(f"**匹配层级**: {m.match_level} (置信度 {m.confidence:.0%})")
                st.markdown(f"**论文方向**: {m.published_direction}")
                st.markdown(f"**限制**: {m.limitations}")

                if m.warning:
                    st.warning(m.warning)

                parsed = parse_taxonomy_string(selected_user_taxon)
                st.caption(
                    f"解析: family={parsed.get('family', '?')}, "
                    f"genus={parsed.get('genus', '?')}, "
                    f"species={parsed.get('species', '?')}"
                )
            st.divider()


def _tab_export(user_taxa, features, user_abund):
    st.markdown("### 导出报告")

    summary = compute_match_summary(user_taxa, features, user_abund)

    export_rows = []
    for m in summary.all_matches:
        export_rows.append({
            "user_feature": m.user_feature,
            "paper_feature_id": m.paper_feature_id,
            "paper_taxonomy": m.paper_taxonomy,
            "match_level": m.match_level,
            "match_confidence": m.confidence,
            "lightgbm_importance": m.importance,
            "published_direction": m.published_direction,
            "user_mean_abundance": m.user_mean_abundance,
            "limitations": m.limitations,
        })

    df_export = pd.DataFrame(export_rows)
    csv_data = export_csv(df_export)
    filename = make_export_filename("fiber_feature_matches", "csv")

    st.download_button(
        label="下载匹配结果 (CSV)",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
    )

    st.markdown("### 摘要统计")
    st.json({
        "total_paper_features": summary.total_paper_features,
        "matched_features": summary.matched_features,
        "raw_coverage": summary.raw_coverage,
        "importance_weighted_coverage": summary.importance_weighted_coverage,
        "high_response_weight": summary.high_response_weight,
        "low_response_weight": summary.low_response_weight,
        "exact_species_count": summary.exact_species_count,
        "exact_genus_count": summary.exact_genus_count,
        "named_group_count": summary.named_group_count,
        "family_count": summary.family_count,
        "conflict_count": summary.conflict_count,
    })

    st.caption(
        "导出文件包含匹配结果和摘要统计，不含原始上传数据。"
    )

    st.info(
        "报告中固定包含：本结果为论文特征证据匹配，不是临床响应预测。"
        "taxonomy 层级匹配不能替代原论文 ASV 及模型。"
        "不应据此单独决定患者饮食方案。"
    )


show()
