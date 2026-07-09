import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from microbiome_src.analysis.confounders import run_confounder_audit
from microbiome_src.analysis.differential import filter_candidates, run_mannwhitney_differential
from microbiome_src.analysis.preprocess import calculate_prevalence, filter_by_prevalence, to_relative_abundance
from microbiome_src.config import Config
from microbiome_src.reporting.export import export_csv, make_export_filename


def show():
    st.title("差异菌分析")

    if "dataset" not in st.session_state or st.session_state.dataset is None:
        st.warning("请先上传数据并完成质控，或载入演示数据。")
        st.caption(
            "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
        )
        return

    dataset = st.session_state.dataset
    cfg = Config()

    groups = dataset.metadata[dataset.group_column].unique()
    if len(groups) < 2:
        st.error(f"分组列 '{dataset.group_column}' 中需要至少两个组，当前: {list(groups)}")
        return

    with st.sidebar:
        st.subheader("分组设置")
        case_label = st.selectbox("T2D 组", options=groups, index=0)
        remaining = [g for g in groups if g != case_label]
        control_label = st.selectbox("对照组", options=remaining, index=0)
        st.divider()

        st.subheader("差异分析参数")
        pseudocount = st.number_input(
            "伪计数 (pseudocount)", 1e-10, 1.0, cfg.DEFAULT_PSEUDOCOUNT, format="%.2e"
        )
        st.divider()

        st.subheader("候选菌筛选")
        fdr_cutoff = st.slider("FDR 阈值", 0.01, 0.50, cfg.FDR_THRESHOLD, 0.01)
        effect_cutoff = st.slider("最小效应量", 0.0, 1.0, cfg.MIN_EFFECT_SIZE, 0.05)
        prev_cutoff = st.slider("最低检出率", 0.0, 0.50, cfg.PREVALENCE_THRESHOLD, 0.05)
        top_n = st.slider("Top taxa 限制", 5, 100, cfg.MAX_DIFF_TAXA)

    st.subheader("探索性差异丰度分析")

    st.markdown(
        "本分析使用 Mann–Whitney U 检验和 Benjamini–Hochberg FDR 校正。"
        "这是**探索性关联分析**，不代表因果关系或诊断结论。"
    )

    with st.expander("混杂因素审计", expanded=False):
        audit = run_confounder_audit(
            dataset.metadata, dataset.group_column, case_label, control_label
        )
        risk_color = {"low": "green", "medium": "orange", "high": "red"}
        st.markdown(f"**混杂风险等级**: :{risk_color.get(audit.risk_level, 'gray')}[{audit.risk_level.upper()}]")

        if audit.risk_level == "high":
            st.error("当前差异结果可能同时反映疾病、药物、肥胖与批次效应。")
        elif audit.risk_level == "medium":
            st.warning("部分协变量存在组间差异，解读结果时需注意。")

        for finding in audit.findings:
            icon = {"high": ":red_circle:", "medium": ":orange_circle:", "low": ":white_circle:"}
            st.markdown(f"{icon.get(finding.severity, '')} **{finding.variable}** — {finding.issue}")
            if finding.detail:
                st.caption(f"  {finding.detail}")

    st.markdown("---")

    if dataset.abundance_scale != "relative":
        abund_for_diff = to_relative_abundance(dataset.abundance)
    else:
        abund_for_diff = dataset.abundance.copy()

    prev = calculate_prevalence(abund_for_diff)
    abund_filtered = filter_by_prevalence(abund_for_diff, prev, 0.0)

    if abund_filtered.shape[1] > top_n:
        top_taxa = abund_filtered.mean(axis=0).nlargest(top_n).index
        abund_filtered = abund_filtered[top_taxa]

    with st.spinner(f"正在对 {abund_filtered.shape[1]} 个 taxa 进行差异分析..."):
        results = run_mannwhitney_differential(
            abund_filtered, dataset.metadata,
            group_column=dataset.group_column,
            case_label=case_label,
            control_label=control_label,
            pseudocount=pseudocount,
        )

    if results.empty:
        st.warning("差异分析未产生有效结果，请检查数据。")
        st.caption("⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。")
        return

    st.session_state.diff_results = results
    st.session_state.case_label = case_label
    st.session_state.control_label = control_label

    candidates = filter_candidates(results, fdr_cutoff, effect_cutoff, prev_cutoff)

    st.markdown("### 火山图")
    _volcano_plot(results, fdr_cutoff, effect_cutoff)
    st.caption("x 轴 = rank-biserial 相关系数（效应量），y 轴 = −log₁₀(FDR)。红点 = 候选差异菌。")

    st.markdown("### 候选差异菌")

    col_a, col_b = st.columns(2)
    col_a.metric("分析 taxa 数", len(results))
    col_b.metric("候选菌数", len(candidates))

    display_cols = [
        "taxon", "log2fc", "fdr", "effect_size", "direction",
        "prevalence_case", "prevalence_control",
    ]
    if not candidates.empty:
        st.dataframe(
            candidates[display_cols].style
            .format({"fdr": "{:.4g}", "log2fc": "{:.3f}", "effect_size": "{:.3f}"})
            .background_gradient(subset=["log2fc"], cmap="RdBu_r"),
            use_container_width=True,
            height=400,
        )
    else:
        st.info("当前阈值下无候选差异菌。请尝试放宽 FDR 或效应量阈值。")

    st.markdown("### 单菌箱线图")
    selected_taxon = st.selectbox(
        "选择 taxon 查看分布",
        options=[""] + [str(t) for t in results["taxon"].tolist()],
        format_func=lambda x: "— 请选择 —" if x == "" else x,
    )
    if selected_taxon:
        _taxon_boxplot(abund_for_diff, dataset, selected_taxon,
                       case_label, control_label)

    st.markdown("### 结果导出")
    csv_data = export_csv(results)
    filename = make_export_filename("differential_results", "csv", dataset.source_name)
    st.download_button(
        label="下载差异分析结果 (CSV)",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
    )
    st.caption("导出文件包含所有 taxa 的统计结果，不含原始上传数据。")

    st.caption(
        "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
        "统计显著性不代表因果关系。分析结果仅供研究参考，不得作为医疗决策依据。"
    )


def _volcano_plot(results, fdr_cutoff, effect_cutoff):
    results = results.copy()
    results["-log10(fdr)"] = -np.log10(results["fdr"].clip(lower=1e-300))
    results["significant"] = (results["fdr"] < fdr_cutoff) & (results["effect_size"].abs() >= effect_cutoff)

    fig = px.scatter(
        results,
        x="effect_size",
        y="-log10(fdr)",
        color="significant",
        color_discrete_map={True: "red", False: "gray"},
        hover_data=["taxon", "log2fc", "fdr"],
        labels={"effect_size": "效应量 (rank-biserial r)", "-log10(fdr)": "-log₁₀(FDR)"},
    )
    fig.add_hline(y=-np.log10(fdr_cutoff), line_dash="dash", line_color="orange",
                  annotation_text=f"FDR={fdr_cutoff}")
    fig.add_vline(x=effect_cutoff, line_dash="dash", line_color="orange")
    fig.add_vline(x=-effect_cutoff, line_dash="dash", line_color="orange")
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=400,
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _taxon_boxplot(abundance, dataset, taxon, case_label, control_label):
    meta_col = dataset.group_column
    grp_series = dataset.metadata[meta_col]
    df = pd.DataFrame({
        "Value": abundance[taxon],
        "Group": grp_series,
    }).dropna()

    fig = px.box(df, x="Group", y="Value", color="Group", points="outliers")
    fig.update_layout(
        title=f"{taxon}",
        yaxis_title="相对丰度",
        margin=dict(l=20, r=20, t=40, b=10),
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


show()
