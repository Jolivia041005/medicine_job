import io

import streamlit as st

from microbiome_src.io.adapters import build_dataset
from microbiome_src.io.loaders import detect_separator, detect_orientation, read_table
from microbiome_src.io.validators import check_resource_limits, MAX_FILE_SIZE_MB


def _check_session_dataset():
    if "dataset" not in st.session_state or st.session_state.dataset is None:
        st.warning("请先上传数据并完成质控。")
        st.stop()


def show():
    st.title("数据上传与质控")

    st.markdown("### 上传丰度表")
    abund_file = st.file_uploader(
        "选择丰度表文件（CSV 或 TSV）",
        type=["csv", "tsv", "txt"],
        key="abund_uploader",
    )

    st.markdown("### 上传样本 Metadata")
    meta_file = st.file_uploader(
        "选择 metadata 文件（CSV 或 TSV）",
        type=["csv", "tsv", "txt"],
        key="meta_uploader",
    )

    if abund_file is None or meta_file is None:
        st.info("请上传丰度表和 metadata 两个文件。")
        st.caption(
            "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
            "分析结果仅供研究参考，不得作为医疗决策依据。"
        )
        return

    prev_abund = st.session_state.get("_prev_abund_name")
    prev_meta = st.session_state.get("_prev_meta_name")
    if prev_abund == abund_file.name and prev_meta == meta_file.name:
        _check_session_dataset()
        st.success("数据已加载并通过质控。")
        _show_dataset_info()
        st.caption(
            "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
            "分析结果仅供研究参考，不得作为医疗决策依据。"
        )
        return

    abund_bytes = abund_file.read()
    meta_bytes = meta_file.read()

    abund_size_mb = len(abund_bytes) / (1024 * 1024)
    meta_size_mb = len(meta_bytes) / (1024 * 1024)

    if abund_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"丰度表文件 ({abund_size_mb:.1f} MB) 超过限制 ({MAX_FILE_SIZE_MB} MB)")
        st.stop()
    if meta_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"Metadata 文件 ({meta_size_mb:.1f} MB) 超过限制 ({MAX_FILE_SIZE_MB} MB)")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        abund_sep = detect_separator(io.BytesIO(abund_bytes))
        st.write(f"丰度表分隔符: `{'TAB' if abund_sep == chr(9) else abund_sep}`")

        abund_preview = read_table(io.BytesIO(abund_bytes))
        orientation_hint = detect_orientation(abund_preview)
        orientation = st.radio(
            "丰度表方向",
            options=["taxa_as_rows", "samples_as_rows"],
            format_func=lambda x: "分类单元为行 / 样本为列" if x == "taxa_as_rows" else "样本为行 / 分类单元为列",
            index=0 if orientation_hint == "taxa_as_rows" else 1,
        )

    with col2:
        meta_sep = detect_separator(io.BytesIO(meta_bytes))
        st.write(f"Metadata 分隔符: `{'TAB' if meta_sep == chr(9) else meta_sep}`")

        group_column = st.text_input("分组列名", value="Group")
        case_label = st.text_input("T2D 标签值", value="T2D")
        control_label = st.text_input("Control 标签值", value="Control")

    if st.button("执行质控", type="primary", use_container_width=True):
        with st.spinner("正在验证数据..."):
            dataset, abund_report, meta_report = build_dataset(
                io.BytesIO(abund_bytes),
                io.BytesIO(meta_bytes),
                orientation=orientation,
                group_column=group_column,
                source_name=abund_file.name,
            )

        st.subheader("丰度表验证结果")
        _show_report(abund_report)

        st.subheader("Metadata 验证结果")
        _show_report(meta_report)

        if dataset is not None:
            st.session_state.dataset = dataset
            st.session_state._prev_abund_name = abund_file.name
            st.session_state._prev_meta_name = meta_file.name
            st.success("数据通过质控！请前往其他页面查看分析结果。")
            _show_dataset_info()
        else:
            st.session_state.dataset = None
            st.error("数据未通过质控，请修正后重新上传。")

    st.caption(
        "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
        "分析结果仅供研究参考，不得作为医疗决策依据。"
    )


def _show_report(report):
    for issue in report.issues:
        if issue.level == "error":
            st.error(issue.message)
        elif issue.level == "warning":
            st.warning(issue.message)
        else:
            st.info(issue.message)


def _show_dataset_info():
    ds = st.session_state.dataset
    st.markdown(f"""
    | 属性 | 值 |
    |------|-----|
    | 来源 | {ds.source_name} |
    | 样本数 | {ds.abundance.shape[0]} |
    | 分类单元数 | {ds.abundance.shape[1]} |
    | 丰度类型 | {ds.abundance_scale} |
    | 分组列 | {ds.group_column} |
    """)


show()
