import streamlit as st

from microbiome_src.io.demo_data import load_demo_dataset


def show():
    st.title("T2D 肠道微生物组解释平台")

    st.markdown("""
    ### 将微生物丰度表转换为可解释的分析结果

    本工具面向研究人员和临床科研人员，帮助快速分析
    2 型糖尿病相关的肠道菌群数据。
    """)

    st.header("目标用户")
    st.markdown("""
    - 持有 OTU、属或种水平丰度表的研究人员
    - 需要快速检查 T2D 与对照组菌群差异的临床科研人员
    - 课程项目用户
    """)

    st.header("支持的分析流程")
    st.markdown("""
    1. 数据上传与格式自动检测
    2. 样本 ID 对齐与质量检查
    3. 相对丰度转换与低丰度过滤
    4. 菌群组成概览（Top taxa、Shannon 多样性）
    5. 差异菌筛选（Mann-Whitney U + FDR + 效应量）
    6. 混杂因素审计
    7. 文献证据匹配与功能菌群标签
    8. CSV 结果导出
    """)

    st.header("演示数据")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("载入 Qin 2012 示例数据", type="primary", use_container_width=True):
            with st.spinner("正在载入 Qin 2012 演示数据..."):
                try:
                    dataset = load_demo_dataset()
                    st.session_state.dataset = dataset
                    st.session_state._prev_abund_name = "qin2012_demo"
                    st.session_state._prev_meta_name = "qin2012_demo"
                    st.success(
                        f"已载入 Qin 2012 数据集: "
                        f"{dataset.abundance.shape[0]} 个样本, "
                        f"{dataset.abundance.shape[1]} 个 taxa"
                    )
                except Exception as e:
                    st.error(f"载入失败: {e}")
    with col2:
        st.caption(
            "Qin et al. (2012) Nature. "
            "124 名中国受试者（65 T2D + 59 Control），"
            "shotgun 宏基因组测序。"
        )

    if "dataset" in st.session_state and st.session_state.dataset is not None:
        ds = st.session_state.dataset
        st.success(f"当前数据集: {ds.source_name} ({ds.abundance.shape[0]} 样本 × {ds.abundance.shape[1]} taxa)")

    st.header("上传自己的数据")
    st.markdown("""
    支持格式：CSV / TSV
    - 丰度表：样本 × 分类单元 或 分类单元 × 样本
    - Metadata 表：至少包含 SampleID 和分组列
    """)

    st.divider()

    st.caption(
        "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
        "分析结果仅供研究参考，不得作为医疗决策依据。"
        "本工具不提供疾病诊断、个体风险评分或治疗建议。"
    )


show()
