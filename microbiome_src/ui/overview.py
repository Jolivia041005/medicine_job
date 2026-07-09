import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from microbiome_src.analysis.diversity import calculate_shannon
from microbiome_src.analysis.preprocess import (
    calculate_prevalence,
    collapse_other_taxa,
    filter_by_mean_abundance,
    filter_by_prevalence,
    select_top_taxa,
    to_relative_abundance,
)


def show():
    st.title("菌群概览")

    if "dataset" not in st.session_state or st.session_state.dataset is None:
        st.warning("请先上传数据并完成质控，或载入演示数据。")
        st.caption(
            "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
        )
        return

    dataset = st.session_state.dataset

    with st.sidebar:
        st.subheader("参数设置")
        prevalence_threshold = st.slider(
            "最低检出率", 0.0, 1.0, 0.10, 0.05,
            help="taxa 在样本中出现的最低比例"
        )
        min_mean_abundance = st.number_input(
            "最低平均相对丰度", 0.0, 1.0, 0.0001, format="%.6f"
        )
        top_n = st.slider(
            "Top taxa 数量", 5, 30, 20,
            help="图表中最多显示的 taxa 数量"
        )

    st.subheader("数据过滤")

    if dataset.abundance_scale != "relative":
        rel_abund = to_relative_abundance(dataset.abundance)
    else:
        rel_abund = dataset.abundance.copy()

    prevalence = calculate_prevalence(rel_abund)
    filtered = filter_by_prevalence(rel_abund, prevalence, prevalence_threshold)
    filtered = filter_by_mean_abundance(filtered, min_mean_abundance)

    col1, col2, col3 = st.columns(3)
    col1.metric("原始 taxa 数", dataset.abundance.shape[1])
    col2.metric("过滤后 taxa 数", filtered.shape[1])
    col3.metric("样本数", dataset.abundance.shape[0])

    groups = dataset.metadata[dataset.group_column].value_counts()
    group_text = " | ".join(f"{k}: {v}" for k, v in groups.items())
    st.info(f"**分组**: {group_text}")

    st.subheader("样本总丰度分布")
    fig = go.Figure()
    for grp, grp_data in groups.items():
        grp_indices = dataset.metadata[dataset.metadata[dataset.group_column] == grp].index
        grp_indices = grp_indices.intersection(dataset.abundance.index)
        fig.add_trace(go.Box(
            y=dataset.abundance.loc[grp_indices].sum(axis=1),
            name=str(grp),
            boxpoints="outliers",
        ))
    fig.update_layout(
        yaxis_title="总丰度（原始计数）",
        xaxis_title="分组",
        margin=dict(l=20, r=20, t=10, b=10),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("菌群组成（Top taxa）")

    top_df = select_top_taxa(filtered, n=top_n)
    collapsed = collapse_other_taxa(filtered, top_n=top_n)

    col1, col2 = st.columns(2)
    for i, (grp, grp_data) in enumerate(groups.items()):
        grp_indices = dataset.metadata[dataset.metadata[dataset.group_column] == grp].index
        grp_indices = grp_indices.intersection(collapsed.index)
        grp_rel = collapsed.loc[grp_indices]
        mean_comp = grp_rel.mean(axis=0).sort_values(ascending=False)

        fig = go.Figure(data=[
            go.Pie(
                labels=mean_comp.index,
                values=mean_comp.values,
                hole=0.4,
                textinfo="label+percent",
                textposition="outside",
                textfont_size=9,
            )
        ])
        fig.update_layout(
            title=f"{grp} 组平均组成",
            margin=dict(l=10, r=10, t=40, b=10),
            height=540,
            legend=dict(font_size=9),
        )
        if i == 0:
            with col1:
                st.plotly_chart(fig, use_container_width=True)
        else:
            with col2:
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Shannon 多样性")

    shannon = calculate_shannon(rel_abund)
    meta_shannon = dataset.metadata.copy()
    meta_shannon["Shannon"] = shannon
    meta_shannon = meta_shannon.dropna(subset=["Shannon"])

    fig = px.box(
        meta_shannon,
        x=dataset.group_column,
        y="Shannon",
        color=dataset.group_column,
        points="outliers",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
        "分析结果仅供研究参考，不得作为医疗决策依据。"
    )


show()
