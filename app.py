import streamlit as st
import time
import pandas as pd
import pdfplumber
from io import BytesIO
import re
import os
import glob
import traceback
import json
import numpy as np
import tempfile
import plotly.graph_objects as go
from t2d_islet_web.streamlit_app import main as t2d_main
from pathlib import Path
import plotly.express as px

# 现有模块
from src.retriever import HybridRetriever
from src.summarizer import generate_summary, generate_structured_evidence
from src.history_manager import (
    init_db, add_history, get_history, clear_history,
    add_lit_history, get_lit_history, clear_lit_history,
    add_predict_history, get_predict_history, clear_predict_history
)
from src.export_utils import export_csv, export_report
from src.literature_retriever import LiteratureRetriever
from src.pdf_translator import PDFTranslator
from drug_recommendation import recommend_drugs
from t2d_islet_web.streamlit_app import main as t2d_main
# 引入微生物组核心模块
from microbiome_src.io.loaders import read_table, standardize_to_sample_x_taxon, detect_orientation
from microbiome_src.io.validators import validate_abundance, validate_metadata
from microbiome_src.io.adapters import build_dataset  # 如果有
from microbiome_src.analysis.preprocess import to_relative_abundance, calculate_prevalence, filter_by_prevalence, select_top_taxa
from microbiome_src.analysis.diversity import calculate_shannon
from microbiome_src.analysis.differential import run_mannwhitney_differential, filter_candidates
from microbiome_src.analysis.confounders import run_confounder_audit
from microbiome_src.evidence.repository import load_repository
from microbiome_src.models import MicrobiomeDataset, ValidationReport

# 风险预测模块
from predict_json_ver import predict as predict_risk

# ==================== PDF 文本处理函数（保持不变）====================
def extract_text_from_pdf(file):
    with pdfplumber.open(BytesIO(file.read())) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

def split_into_snippets(text, min_len=30, max_len=600):
    paragraphs = re.split(r'\n\s*\n', text)
    snippets = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_len and len(para) >= min_len:
            snippets.append(para)
            continue
        sentences = re.split(r'(?<=[。？！．\?!.])', para)
        merged = []
        current = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) <= max_len:
                current += sent
            else:
                if current:
                    merged.append(current)
                current = sent
        if current:
            merged.append(current)
        for m in merged:
            if len(m) >= min_len:
                snippets.append(m)
            else:
                if snippets and len(snippets[-1]) + len(m) <= max_len:
                    snippets[-1] += m
                else:
                    snippets.append(m)
    unique = []
    for s in snippets:
        if s not in unique:
            unique.append(s)
    return unique

def clean_text(text):
    return re.sub(r'[\x00-\x1f\x7f]', '', text)

# ==================== 加载 /original 目录 ====================
def load_initial_pdfs_with_progress():
    original_dir = "./original"
    if not os.path.exists(original_dir):
        return None
    pdf_files = glob.glob(os.path.join(original_dir, "*.pdf"))
    if not pdf_files:
        return None
    progress_bar = st.progress(0, text="正在加载初始文献...")
    status_text = st.empty()
    all_snippets = []
    all_sources = []
    total = len(pdf_files)
    for i, pdf_path in enumerate(pdf_files):
        file_name = os.path.basename(pdf_path)
        status_text.text(f"处理 {file_name} ({i+1}/{total})")
        try:
            with open(pdf_path, 'rb') as f:
                full_text = extract_text_from_pdf(BytesIO(f.read()))
            snippets = split_into_snippets(full_text)
            for snippet in snippets:
                all_snippets.append(clean_text(snippet))
                all_sources.append(f"初始文献: {file_name}")
        except Exception as e:
            st.warning(f"加载 {file_name} 失败: {e}")
        progress_bar.progress((i+1)/total)
    progress_bar.empty()
    status_text.empty()
    if all_snippets:
        return pd.DataFrame({'snippet': all_snippets, 'source': all_sources})
    return None

# ==================== 页面配置 ====================
st.set_page_config(page_title="临床智能检索系统", layout="wide")
st.title("🩺 临床智能检索与文献证据系统")

init_db()

# ---------- 初始化队列检索器 ----------
@st.cache_resource
def get_queue_retriever():
    return HybridRetriever(data_path="data/patient_data.parquet")

queue_retriever = get_queue_retriever()

# ---------- 文献检索器 ----------
if 'lit_retriever' not in st.session_state:
    with st.spinner("正在初始化文献知识库..."):
        initial_df = load_initial_pdfs_with_progress()
        if initial_df is not None and not initial_df.empty:
            st.session_state['lit_df'] = initial_df
            st.session_state['lit_retriever'] = LiteratureRetriever(corpus_df=initial_df)
            st.info(f"📚 已从 /original 目录加载 {len(initial_df)} 条初始文献片段。")
        else:
            st.session_state['lit_retriever'] = LiteratureRetriever()
            st.session_state['lit_df'] = st.session_state['lit_retriever'].df.copy()
            st.info("📚 使用内置默认文献语料（未找到 /original 目录或其中无 PDF）。")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 📌 使用说明")
    st.sidebar.caption("• **队列检索**：输入医学描述，从患者库中匹配并生成结构化证据。")
    st.sidebar.caption("• **文献检索**：在指南/文献片段中语义检索，支持PDF扩充。")
    st.sidebar.caption("• **风险预测**：基于XGBoost模型预测糖尿病风险，并提供SHAP解释。")
    st.sidebar.caption("• **胰腺微环境**：单细胞数据分析，评估胰岛内皮细胞健康。")
    st.sidebar.caption("• **联动**：队列证据可跳转文献检索，风险因素可自动检索文献。")
    st.sidebar.caption("• **用药推荐**：基于2024/2025指南的个体化降糖方案推荐。")
    st.sidebar.caption("• **T2D转录组**：基于胰岛RNA-seq的差异基因与通路检索。")
    st.sidebar.caption("基于 QIIME2 分类丰度表进行多样性、差异分析和 T2D 文献证据匹配")
    st.divider()
    st.caption("📋 历史记录已移至对应标签页查看。")

# ==================== 七个标签页 ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "👥 队列检索与证据",
    "📚 文献语义检索",
    "🧠 风险预测与解释",
    "🔬 胰腺微环境评估",
    "💊 糖尿病用药推荐",
    "🧬 T2D 转录组检索",
    "🦠 肠道微生物组分析"          # 新增
])

# ============================================================
# TAB 1: 队列检索（保持不变，只展示完整）
# ============================================================
with tab1:
    st.markdown("### 🔍 医学查询输入（从患者队列中检索）")
    # ==================== 自定义数据上传 ====================
    default_data_path = "data/patient_data.parquet"
    if 'user_data_path' not in st.session_state:
        st.session_state['user_data_path'] = default_data_path
    with st.expander("📁 上传自定义患者队列数据（CSV / Parquet）", expanded=False):
        uploaded_file = st.file_uploader(
            "文件需包含列：patient_id, age, sex, bmi, diagnoses, medications, readmitted_30d, in_hospital_death, description",
            type=["csv", "parquet"],
            key="data_upload"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_up = pd.read_csv(uploaded_file)
                else:
                    df_up = pd.read_parquet(uploaded_file)
                required = [
                    "患者编号", "年龄", "性别", "体重指数", "诊断", "用药", "30天内再入院", "住院死亡", "描述"
                ]
                missing = [col for col in required if col not in df_up.columns]
                if missing:
                    st.error(f"缺少必需列：{missing}")
                else:
                    # 保存到临时文件（避免污染原数据）
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                    df_up.to_parquet(tmp.name, index=False)
                    st.session_state['user_data_path'] = tmp.name
                    st.success(f"已使用自定义数据（{len(df_up)} 条记录）")
            except Exception as e:
                st.error(f"读取文件时出错：{e}")
        else:
            if st.session_state['user_data_path'] != default_data_path:
                st.session_state['user_data_path'] = default_data_path
                st.info("已恢复使用默认模拟数据（data/patient_data.parquet）")
    # ==================== 初始化检索器（动态数据源） ====================
    @st.cache_resource
    def get_queue_retriever(data_path):
        return HybridRetriever(data_path=data_path)
    queue_retriever = get_queue_retriever(st.session_state['user_data_path'])
    # ---------- 后续检索 UI、结果展示、历史记录等（保持原代码不变）----------
    if 'queue_results' in st.session_state and not st.session_state['queue_results'].empty:
        cached_results = st.session_state['queue_results']
        cached_evidence = st.session_state.get('queue_evidence', '')
        cached_summary = st.session_state.get('queue_summary', {})
        cached_elapsed = st.session_state.get('queue_elapsed', 0)
        cached_llm = st.session_state.get('queue_llm_output', '')
        st.success(f"✅ 上次检索找到 {len(cached_results)} 条匹配记录（耗时 {cached_elapsed:.3f} 秒）")
        st.subheader("📊 结构化队列证据")
        st.info(cached_evidence)
        with st.expander("📋 详细统计摘要", expanded=False):
            st.json(cached_summary)
        st.subheader("📋 患者列表")
        st.dataframe(cached_results, use_container_width=True)
        if cached_llm:
            st.subheader("🤖 文献建议（DeepSeek 生成）")
            st.markdown(cached_llm)
        st.subheader("🔗 模块联动")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            if st.button("查找相关文献（基于本队列证据）", key="link_to_lit_cached"):
                st.session_state['lit_query'] = cached_evidence
                st.session_state['lit_auto_search'] = True
                st.rerun()
        with col_l2:
            if not cached_results.empty:
                # 提供按钮，跳转到预测标签并导入首个患者的年龄和BMI
                if st.button("导入首个患者特征进行风险预测", key="import_first_patient"):
                    first_patient = cached_results.iloc[0]
                    st.session_state['pending_import'] = {
                        'Age': int(first_patient.get('age', 30)),
                        'BMI': float(first_patient.get('bmi', 25.0)),
                    }
                    st.rerun()  # 跳转到该标签页并不重要，但我们在tab3中会检测pending_import
        st.divider()
        st.caption("💡 以上为缓存结果，如需重新检索，请修改查询并点击下方按钮。")
    
    col_q, col_f = st.columns([3, 1])
    with col_q:
        query = st.text_input("请输入医学问题", label_visibility="collapsed", 
                              placeholder="例如：糖尿病合并肾病的老年患者")
    with col_f:
        top_k = st.number_input("返回数量", min_value=1, max_value=100, value=10, step=1)
    
    sql_filter = st.text_input("SQL 过滤条件（可选）", placeholder="例如： age > 60 AND bmi > 30")
    
    col_opts = st.columns([1, 1, 1])
    with col_opts[0]:
        use_llm = st.checkbox("启用 LLM 生成文献建议（实验性）", value=False)
    with col_opts[1]:
        st.write("")
    with col_opts[2]:
        if use_llm:
            api_key = st.text_input("DeepSeek API Key（可选）", type="password", 
                                    placeholder="输入 DeepSeek API Key 启用真实生成，否则模拟")
            st.session_state['api_key'] = api_key
        else:
            api_key = None
    
    if st.button("检索", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("请输入查询内容")
        else:
            start_time = time.time()
            with st.spinner("检索中..."):
                results = queue_retriever.search(query, sql_filter, top_k)
            elapsed = time.time() - start_time
            
            evidence_text = ""
            summary = {}
            llm_output = ""
            if not results.empty:
                evidence_text = generate_structured_evidence(results)
                summary = generate_summary(results)
                st.session_state['queue_results'] = results
                st.session_state['queue_evidence'] = evidence_text
                st.session_state['queue_summary'] = summary
                st.session_state['queue_elapsed'] = elapsed
                st.session_state['queue_query'] = query
                st.session_state['queue_sql_filter'] = sql_filter
                st.session_state['queue_top_k'] = top_k
                
                st.success(f"✅ 找到 {len(results)} 条匹配记录（耗时 {elapsed:.3f} 秒）")
                st.subheader("📊 结构化队列证据")
                st.info(evidence_text)
                with st.expander("📋 详细统计摘要", expanded=False):
                    st.json(summary)
                st.subheader("📋 患者列表")
                st.dataframe(results, use_container_width=True)
                
                if use_llm:
                    st.subheader("🤖 文献建议（DeepSeek 生成）")
                    with st.spinner("生成中..."):
                        if api_key:
                            try:
                                import openai
                                client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                                prompt = (
                                    f"基于以下患者队列特征：{evidence_text}\n"
                                    "请为医生推荐相关的临床文献或研究方向。要求：\n"
                                    "1. 每条建议不超过 30 字；\n"
                                    "2. 按重要性排序，最多 5 条；\n"
                                    "3. 包含具体文献类型（如指南、RCT、Meta分析）和年份。"
                                )
                                response = client.chat.completions.create(
                                    model="deepseek-chat",
                                    messages=[{"role": "system", "content": "你是一位专业的医学文献顾问，回复应精炼、结构化。"},
                                              {"role": "user", "content": prompt}],
                                    max_tokens=600, temperature=0.5
                                )
                                llm_output = response.choices[0].message.content
                                if not llm_output.strip():
                                    llm_output = "（模型返回空内容，请检查网络或稍后重试）"
                            except Exception as e:
                                llm_output = f"⚠️ API 调用失败：{e}。\n\n模拟建议：针对老年糖尿病合并肾病，建议查阅2024年ADA指南关于SGLT2抑制剂在CKD患者中的应用，以及最新发表的CREDENCE试验后续分析。"
                        else:
                            llm_output = "（模拟）请提供 DeepSeek API Key 以获取真实文献推荐。\n\n模拟建议：针对该队列，建议关注：\n1. 糖尿病肾病的早期生物标志物（RCT, 2023）\n2. SGLT2抑制剂在老年人群中的疗效（Meta分析, 2024）\n3. 血糖控制与心血管结局的最新指南（ADA, 2025）"
                        st.markdown(llm_output)
                        st.session_state['queue_llm_output'] = llm_output
                else:
                    st.session_state['queue_llm_output'] = ""
                
                st.subheader("🔗 模块联动")
                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    if st.button("查找相关文献（基于本队列证据）", key="link_to_lit"):
                        st.session_state['lit_query'] = evidence_text
                        st.session_state['lit_auto_search'] = True
                        st.rerun()
                with col_l2:
                    if st.button("导入首个患者特征进行风险预测", key="import_first_patient2"):
                        first_patient = results.iloc[0]
                        st.session_state['pending_import'] = {
                            'Age': int(first_patient.get('age', 30)),
                            'BMI': float(first_patient.get('bmi', 25.0)),
                        }
                        st.rerun()
                
                st.subheader("📤 导出")
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    csv_data = export_csv(results)
                    st.download_button(
                        label="📥 下载 CSV",
                        data=csv_data,
                        file_name=f"query_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_exp2:
                    report_md = export_report(query, results, summary, evidence_text, elapsed)
                    st.download_button(
                        label="📄 下载报告 (Markdown)",
                        data=report_md,
                        file_name=f"clinical_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            else:
                st.warning("未找到匹配记录。")
                st.session_state['queue_results'] = pd.DataFrame()
                st.session_state['queue_evidence'] = ""
                st.session_state['queue_summary'] = {}
                st.session_state['queue_llm_output'] = ""
            
            add_history(query, sql_filter, top_k, len(results), elapsed, evidence_text, llm_output)
    
    # ---------- 队列历史 ----------
    with st.expander("📜 队列检索历史（最近10条）", expanded=False):
        history_df = get_history(10)
        if not history_df.empty:
            for _, row in history_df.iterrows():
                with st.container():
                    st.markdown(f"**查询：** {row['query']}")
                    st.caption(f"时间：{row['timestamp'][:16]}  |  结果数：{row['result_count']}  |  耗时：{row['duration']:.2f}s")
                    evidence = row.get('evidence_text', '')
                    if evidence:
                        st.markdown(f"**📊 结构化证据：** {evidence[:200]}{'...' if len(evidence)>200 else ''}")
                    llm_out = row.get('llm_output', '')
                    if llm_out:
                        st.markdown(f"**🤖 文献建议前三：**")
                        lines = [line for line in llm_out.split('\n') if line.strip() and any(x in line for x in ['1.', '2.', '3.'])]
                        for line in lines[:3]:
                            st.markdown(f"- {line}")
                    st.divider()
            if st.button("🗑️ 清空队列历史", key="clear_queue_history"):
                clear_history()
                st.rerun()
        else:
            st.info("暂无队列检索历史。")
# ============================================================
# TAB 2: 文献语义检索（保持不变，只展示完整）
# ============================================================
with tab2:
    st.markdown("### 📚 基于临床指南和文献的语义检索")
    st.caption("从内置指南/文献片段中检索最相关的内容，并附来源引用。您可以上传中文或英文 PDF 文献扩充知识库。")

    # ---------- PDF 上传 ----------
    with st.expander("📤 上传 PDF 文献以增强检索", expanded=False):
        if 'api_key' in st.session_state and st.session_state.get('api_key'):
            trans_api_key = st.session_state['api_key']
            st.success("✅ 已从队列检索模块获取 DeepSeek API Key，英文翻译功能可用。")
        else:
            trans_api_key = st.text_input("请输入 DeepSeek API Key 用于英文文献翻译", type="password", key="trans_api_key")
            if trans_api_key:
                st.session_state['api_key'] = trans_api_key
        lang_type = st.radio("文献语言", ["中文", "英文"], horizontal=True, key="lang_type")
        uploaded_files = st.file_uploader("选择 PDF 文件（支持多个）", type="pdf", accept_multiple_files=True, key="pdf_uploader")
        if uploaded_files:
            st.info(f"已选择 {len(uploaded_files)} 个 PDF 文件（{lang_type}）。")
            if st.button("🚀 处理并更新文献库", use_container_width=True):
                if lang_type == "英文" and not trans_api_key:
                    st.error("请提供有效的 DeepSeek API Key 以进行英文翻译。")
                    st.stop()
                new_snippets = []
                new_sources = []
                progress_bar = st.progress(0, text="正在处理文献...")
                total = len(uploaded_files)
                translator = PDFTranslator(api_key=trans_api_key) if lang_type == "英文" else None
                for i, file in enumerate(uploaded_files):
                    progress_bar.progress((i+1)/total, text=f"正在处理: {file.name}")
                    try:
                        full_text = extract_text_from_pdf(file)
                        if not full_text.strip():
                            st.warning(f"文件 {file.name} 未能提取到有效文本，已跳过。")
                            continue
                        if lang_type == "英文":
                            translated_df = translator.translate_pdf_text(full_text, file.name)
                            for _, row in translated_df.iterrows():
                                snippet_text = row['translated_text'] if row['translated_text'] else row['original_text']
                                original_preview = row['original_text'][:200] + "..." if len(row['original_text']) > 200 else row['original_text']
                                source_info = f"PDF: {file.name} (原文: {original_preview})"
                                new_snippets.append(clean_text(snippet_text))
                                new_sources.append(source_info)
                        else:
                            file_snippets = split_into_snippets(full_text)
                            for snippet in file_snippets:
                                new_snippets.append(clean_text(snippet))
                                new_sources.append(f"PDF: {file.name}")
                    except Exception as e:
                        st.error(f"处理文件 {file.name} 失败: {e}")
                progress_bar.empty()
                if new_snippets:
                    new_df = pd.DataFrame({'snippet': new_snippets, 'source': new_sources})
                    existing_df = st.session_state['lit_df']
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['snippet']).reset_index(drop=True)
                    st.session_state['lit_retriever'] = LiteratureRetriever(corpus_df=combined_df)
                    st.session_state['lit_df'] = combined_df
                    st.success(f"✅ 成功添加 {len(new_snippets)} 条文献片段，当前总片段数：{len(combined_df)}")
                    st.rerun()
                else:
                    st.warning("未生成任何有效片段，请检查文件内容。")
    
    # ---------- 知识库管理 ----------
    with st.expander("📂 管理知识库（查看/删除/导出/导入）", expanded=False):
        st.write("当前知识库中共有 **{}** 个片段。".format(len(st.session_state['lit_df'])))
        source_counts = st.session_state['lit_df']['source'].value_counts()
        if not source_counts.empty:
            df_sources = source_counts.reset_index()
            df_sources.columns = ['来源', '片段数']
            st.dataframe(df_sources, use_container_width=True)
            source_to_delete = st.selectbox("选择要删除的文献来源", df_sources['来源'].tolist())
            if st.button("🗑️ 删除该来源的所有片段", use_container_width=True):
                remaining_df = st.session_state['lit_df'][st.session_state['lit_df']['source'] != source_to_delete]
                st.session_state['lit_df'] = remaining_df.reset_index(drop=True)
                st.session_state['lit_retriever'] = LiteratureRetriever(corpus_df=remaining_df)
                st.success(f"已删除来源 '{source_to_delete}' 的所有片段。当前总片段数：{len(remaining_df)}")
                st.rerun()
        else:
            st.info("知识库为空。")
        st.divider()
        col_export, col_import = st.columns(2)
        with col_export:
            if st.button("📤 导出知识库 (Parquet)", use_container_width=True):
                parquet_data = st.session_state['lit_df'].to_parquet(index=False)
                st.download_button(label="📥 点击下载 knowledge_base.parquet", data=parquet_data,
                                   file_name="knowledge_base.parquet", mime="application/octet-stream", use_container_width=True)
        with col_import:
            uploaded_kb = st.file_uploader("📥 导入知识库 (Parquet)", type=["parquet"], key="kb_import")
            if uploaded_kb is not None:
                try:
                    imported_df = pd.read_parquet(BytesIO(uploaded_kb.read()))
                    if 'snippet' in imported_df.columns and 'source' in imported_df.columns:
                        combined_df = pd.concat([st.session_state['lit_df'], imported_df], ignore_index=True)
                        combined_df = combined_df.drop_duplicates(subset=['snippet']).reset_index(drop=True)
                        st.session_state['lit_retriever'] = LiteratureRetriever(corpus_df=combined_df)
                        st.session_state['lit_df'] = combined_df
                        st.success(f"导入成功，合并后共 {len(combined_df)} 个片段。")
                        st.rerun()
                    else:
                        st.error("导入文件格式错误：缺少 'snippet' 或 'source' 列。")
                except Exception as e:
                    st.error(f"导入失败：{e}")

    # ---------- 文献历史 ----------
    with st.expander("📜 文献检索历史（最近10条）", expanded=False):
        lit_history_df = get_lit_history(10)
        if not lit_history_df.empty:
            for _, row in lit_history_df.iterrows():
                with st.container():
                    st.markdown(f"**查询：** {row['query']}")
                    st.caption(f"时间：{row['timestamp'][:16]}  |  结果数：{row['result_count']}  |  耗时：{row['duration']:.2f}s")
                    top_res = row.get('top_results', '')
                    if top_res:
                        st.markdown("**📌 检索片段前2：**")
                        snippets = top_res.split('|||')[:2]
                        for snippet in snippets:
                            st.markdown(f"- {snippet[:150]}{'...' if len(snippet)>150 else ''}")
                    summary = row.get('llm_summary', '')
                    if summary:
                        st.markdown(f"**📝 AI 总结：** {summary}")
                    st.divider()
            if st.button("🗑️ 清空文献历史", key="clear_lit_history"):
                clear_lit_history()
                st.rerun()
        else:
            st.info("暂无文献检索历史。")

    # ---------- 联动自动检索 ----------
    auto_query = st.session_state.pop('lit_query', "")
    auto_search = st.session_state.pop('lit_auto_search', False)
    if auto_query:
        default_query = auto_query
        st.session_state['trigger_search'] = True
    else:
        default_query = ""
    
    lit_query = st.text_input("请输入医学问题（如：SGLT2抑制剂在老年患者中的应用）", value=default_query)
    lit_top_k = st.slider("返回文献片段数量", 1, 50, 10)
    
    use_lit_llm = st.checkbox("使用 LLM 对检索结果进行总结 (需提供 DeepSeek API Key)", value=False)
    if use_lit_llm:
        lit_api_key_input = st.text_input(
            "DeepSeek API Key（在此输入将优先使用，若留空则尝试复用队列的 Key）",
            type="password",
            key="lit_api_key_input",
            placeholder="请输入您的 DeepSeek API Key"
        )
        if lit_api_key_input:
            st.session_state['lit_api_key'] = lit_api_key_input
        if st.session_state.get('lit_api_key'):
            st.success("✅ 已使用文献模块独立的 API Key。")
        elif 'api_key' in st.session_state and st.session_state.get('api_key'):
            st.info("ℹ️ 使用队列检索模块的 API Key。")
        else:
            st.warning("⚠️ 未检测到 API Key，将使用模拟响应。")
    
    search_clicked = st.button("检索文献", type="primary", use_container_width=True)
    trigger = search_clicked or st.session_state.pop('trigger_search', False)
    
    if trigger:
        if not lit_query.strip():
            st.warning("请输入查询内容")
        else:
            start_time = time.time()
            with st.spinner("检索文献库..."):
                lit_retriever = st.session_state['lit_retriever']
                lit_results = lit_retriever.search(lit_query, top_k=lit_top_k)
            elapsed = time.time() - start_time
            top_results_str = ""
            llm_summary_text = ""
            if not lit_results.empty:
                top_results_str = "|||".join([row['snippet'] for _, row in lit_results.head(2).iterrows()])
                if use_lit_llm:
                    lit_api_key = st.session_state.get('lit_api_key') or st.session_state.get('api_key')
                    if lit_api_key:
                        try:
                            import openai
                            client = openai.OpenAI(api_key=lit_api_key, base_url="https://api.deepseek.com")
                            context_parts = []
                            for _, row in lit_results.iterrows():
                                snippet = clean_text(row['snippet'].strip())
                                source = clean_text(row['source'].strip())
                                context_parts.append(f"来源：{source}\n内容：{snippet}")
                            context = "\n\n".join(context_parts)
                            prompt = f"基于以下文献片段，为医生提供精炼结论（≤200字），并标注关键来源：\n\n{context}"
                            response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=400, temperature=0.5
                            )
                            llm_summary_text = response.choices[0].message.content
                            if not llm_summary_text.strip():
                                llm_summary_text = "（空内容）"
                        except Exception as e:
                            llm_summary_text = f"总结失败：{str(e)}"
                    else:
                        llm_summary_text = "（未提供API Key，无法生成总结）"
            add_lit_history(lit_query, lit_top_k, len(lit_results), elapsed, top_results_str, llm_summary_text)
            st.session_state['lit_results'] = lit_results
            st.session_state['lit_query_text'] = lit_query
            st.session_state['lit_llm_summary'] = llm_summary_text if llm_summary_text else None
            st.session_state['lit_elapsed'] = elapsed
            if not lit_results.empty:
                st.rerun()
            else:
                st.warning("未找到相关文献片段。")
    
    if 'lit_results' in st.session_state and not st.session_state['lit_results'].empty:
        lit_results = st.session_state['lit_results']
        elapsed = st.session_state.get('lit_elapsed', 0)
        st.success(f"✅ 找到 {len(lit_results)} 条相关文献片段（耗时 {elapsed:.3f} 秒）")
        for idx, row in lit_results.iterrows():
            with st.container():
                st.markdown(f"**📌 片段 {idx+1}** (相关度: {row['score']:.3f})")
                st.markdown(f"> {row['snippet']}")
                st.caption(f"📖 来源：{row['source']}")
                st.divider()
        st.subheader("📤 导出引文")
        bibtex_entries = []
        for idx, row in lit_results.iterrows():
            key = re.sub(r'[^a-zA-Z0-9_]', '', row['source'][:20]) + str(idx)
            snippet_clean = clean_text(row['snippet'].replace('"', "'"))
            bibtex = f"@misc{{{key},\n  note = {{{snippet_clean[:150]}...}},\n  howpublished = {{{row['source']}}}\n}}"
            bibtex_entries.append(bibtex)
        bibtex_text = "\n\n".join(bibtex_entries)
        st.download_button(
            label="📥 下载 BibTeX (.bib)",
            data=bibtex_text,
            file_name=f"literature_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.bib",
            mime="text/plain",
            use_container_width=True
        )
        if 'lit_llm_summary' in st.session_state and st.session_state['lit_llm_summary']:
            st.subheader("📝 AI 生成文献总结")
            st.markdown(st.session_state['lit_llm_summary'])
        elif use_lit_llm:
            lit_api_key = st.session_state.get('lit_api_key') or st.session_state.get('api_key')
            if lit_api_key:
                st.subheader("📝 AI 生成文献总结")
                with st.spinner("生成总结..."):
                    try:
                        import openai
                        client = openai.OpenAI(api_key=lit_api_key, base_url="https://api.deepseek.com")
                        context_parts = []
                        for _, row in lit_results.iterrows():
                            snippet = clean_text(row['snippet'].strip())
                            source = clean_text(row['source'].strip())
                            context_parts.append(f"来源：{source}\n内容：{snippet}")
                        context = "\n\n".join(context_parts)
                        prompt = f"基于以下文献片段，为医生提供精炼结论（≤200字），并标注关键来源：\n\n{context}"
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=400, temperature=0.5
                        )
                        summary_text = response.choices[0].message.content
                        if not summary_text.strip():
                            summary_text = "（空内容）"
                        st.markdown(summary_text)
                        st.session_state['lit_llm_summary'] = summary_text
                    except Exception as e:
                        st.error(f"总结失败：{str(e)}")
            else:
                st.info("💡 请提供 DeepSeek API Key 以生成总结（可在上方输入）。")
    else:
        if not trigger:
            st.info("暂无文献检索结果，请输入查询并点击检索按钮。")

# ============================================================
# TAB 3: 风险预测与解释（重点完善模块）
# ============================================================
with tab3:
    st.markdown("### 🧠 糖尿病风险预测与个体化解释")
    st.caption("基于XGBoost模型（PIMA数据集）预测患病风险，并使用SHAP生成可视化解释。")

    # ---------- 初始化预测输入默认值 ----------
    # 所有输入控件都绑定到 session_state 的 key 上，确保值可被外部修改（例如从队列导入）
    default_values = {
        'pred_preg': 0,
        'pred_glu': 120,
        'pred_bp': 70,
        'pred_skin': 20,
        'pred_ins': 80,
        'pred_bmi': 25.0,
        'pred_dpf': 0.5,
        'pred_age': 30
    }
    for key, val in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ---------- 从队列导入特征的处理 ----------
    # 检查是否有待导入的数据（来自队列检索的链接点击）
    if 'pending_import' in st.session_state:
        import_data = st.session_state.pop('pending_import')
        st.success("已从队列导入部分特征（年龄、BMI）")
        # 更新对应的 session_state，下次渲染时输入框会显示新值
        st.session_state['pred_age'] = import_data.get('Age', 30)
        st.session_state['pred_bmi'] = import_data.get('BMI', 25.0)
        # 强制刷新一次以应用新值（如果还未刷新）
        st.rerun()

    # ---------- 输入表单 ----------
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("怀孕次数 (Pregnancies)", min_value=0, max_value=20, 
                                       step=1, key="pred_preg")
        glucose = st.number_input("血糖浓度 (Glucose, mg/dL)", min_value=0, max_value=300, 
                                   step=1, key="pred_glu")
        blood_pressure = st.number_input("血压 (BloodPressure, mmHg)", min_value=0, max_value=200, 
                                          step=1, key="pred_bp")
        skin_thickness = st.number_input("皮肤厚度 (SkinThickness, mm)", min_value=0, max_value=100, 
                                          step=1, key="pred_skin")
    with col2:
        insulin = st.number_input("胰岛素 (Insulin, μU/mL)", min_value=0, max_value=900, 
                                   step=1, key="pred_ins")
        bmi = st.number_input("BMI (体重指数)", min_value=0.0, max_value=70.0, 
                              step=0.1, key="pred_bmi")
        dpf = st.number_input("糖尿病家族史 (DiabetesPedigreeFunction)", min_value=0.0, max_value=3.0, 
                              step=0.01, key="pred_dpf")
        age = st.number_input("年龄 (Age)", min_value=0, max_value=120, 
                              step=1, key="pred_age")

    # ---------- 提供手动从队列导入的快捷方式 ----------
    if 'queue_results' in st.session_state and not st.session_state['queue_results'].empty:
        with st.expander("📋 从当前队列结果导入特征", expanded=False):
            queue_df = st.session_state['queue_results']
            # 使用患者列表选择
            patient_id_col = 'patient_id' if 'patient_id' in queue_df.columns else None
            if patient_id_col:
                patient_options = queue_df[patient_id_col].tolist()
            else:
                patient_options = [f"患者 {i+1}" for i in range(len(queue_df))]
            selected_option = st.selectbox("选择患者", patient_options, key="patient_select")
            if st.button("导入该患者的年龄与BMI"):
                if patient_id_col:
                    selected_row = queue_df[queue_df[patient_id_col] == selected_option].iloc[0]
                else:
                    index = int(selected_option.split()[-1]) - 1
                    selected_row = queue_df.iloc[index]
                st.session_state['pred_age'] = int(selected_row.get('age', 30))
                st.session_state['pred_bmi'] = float(selected_row.get('bmi', 25.0))
                st.rerun()

    # ---------- 预测按钮 ----------
    predict_btn = st.button("🔍 预测糖尿病风险", type="primary", use_container_width=True)

    # 执行预测并在下方显示结果
    if predict_btn:
        patient = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        try:
            result = predict_risk(patient)  # 从 predict_json_ver 导入的函数
            st.success(f"预测结果：**{'患病' if result['prediction'] == 1 else '未患病'}**，风险概率：{result['probability']:.2%}")
            
            # 特征重要性表格
            st.subheader("📊 主要风险因素 (Top 3 SHAP)")
            if result.get('top_features'):
                df_importance = pd.DataFrame(result['top_features'])
                st.dataframe(df_importance, use_container_width=True)
            else:
                st.info("未获取到特征重要性数据。")
            
            # SHAP 瀑布图
            if result.get('waterfall'):
                st.subheader("📈 个体化解释 (SHAP Waterfall)")
                st.image(f"data:image/png;base64,{result['waterfall']}", use_container_width=True)
            else:
                st.warning("未能生成 SHAP 瀑布图。")
            
            # 存储预测历史
            add_predict_history(
                patient_features=json.dumps(patient),
                prediction=result['prediction'],
                probability=result['probability'],
                top_features=json.dumps(result.get('top_features', []))
            )
            
            # 联动：自动生成文献检索关键词并跳转
            st.subheader("🔗 联动文献检索")
            # 提取前两个风险因素名称
            top_feature_names = [f['feature'] for f in result.get('top_features', [])[:2]]
            if top_feature_names:
                keywords = " ".join(top_feature_names) + " 糖尿病"
                if st.button(f"以「{keywords}」检索相关文献", key="predict_to_lit"):
                    st.session_state['lit_query'] = keywords
                    st.session_state['lit_auto_search'] = True
                    st.rerun()
            else:
                st.caption("无显著风险因素可用于文献检索。")
                
        except FileNotFoundError:
            st.error("❌ 模型文件 `xgboost_diabetes_model.pkl` 未找到，请确保文件存在于项目根目录。")
        except Exception as e:
            st.error(f"❌ 预测过程中发生错误：{e}")
            st.info("请检查输入值是否合理，或联系开发者。")

    # ---------- 预测历史记录 ----------
    with st.expander("📜 预测历史（最近10条）", expanded=False):
        pred_history = get_predict_history(10)
        if not pred_history.empty:
            for _, row in pred_history.iterrows():
                with st.container():
                    features = json.loads(row['patient_features'])
                    st.markdown(f"**特征概要：** 怀孕{features.get('Pregnancies',0)}次，血糖{features.get('Glucose',0)}，"
                                f"BMI{features.get('BMI',0)}，年龄{features.get('Age',0)}")
                    st.caption(f"预测：{'患病' if row['prediction']==1 else '未患病'}，概率：{row['probability']:.2%}，"
                               f"时间：{row['timestamp'][:16]}")
                    top_feats = json.loads(row['top_features'])
                    if top_feats:
                        st.markdown("**主要风险因素：** " + ", ".join([f"{f['feature']} ({f['effect']})" for f in top_feats]))
                    st.divider()
            if st.button("🗑️ 清空预测历史", key="clear_predict_history"):
                clear_predict_history()
                st.rerun()
        else:
            st.info("暂无预测历史记录。")
# TAB 4: 胰腺微环境评估（新整合模块）
# ============================================================
with tab4:
    st.header("🔬 胰腺微环境评估")
    st.caption("上传健康与疾病单细胞样本，进行质控、聚类、细胞通讯分析（基于 Nat Commun 2025）")
    # ---------- 依赖导入（若全局未导入则在这里单独导入） ----------
    try:
        import scanpy as sc
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap, os, warnings, tempfile, pathlib
        import liana
        warnings.filterwarnings("ignore")
        sc.settings.verbosity = 0
    except ImportError as e:
        st.error(f"缺少单细胞分析库：{e}。请运行 `pip install scanpy liana umap-learn` 后刷新页面。")
        st.stop()
    # ---------- 常量与颜色映射 ----------
    DATA_DIR = "data"   # 统一使用项目 data 目录
    BENCH_PATH = os.path.join(DATA_DIR, "isec_benchmark.csv")
    bm = pd.read_csv(BENCH_PATH) if os.path.exists(BENCH_PATH) else pd.DataFrame()
    ISEC_MARKERS = ["COL13A1","ESM1","PLVAP","UNC5B","LAMA4","KDR","THBS1","BMP4","CXCR4","ACE","PASK","F2RL3"]
    ALL_T2D_GENES = ["KDR", "ESM1", "DLL4", "PLVAP", "CXCL12"]
    PANCREAS_CELL_COLORS = {
        "Beta": "#f1c40f", "Alpha": "#2ecc71", "Delta": "#9b59b6",
        "PP": "#e67e22", "Epsilon": "#d35400",
        "Acinar": "#3498db", "Ductal": "#1abc9c",
        "ISEC": "#e74c3c", "ASEC": "#ff6b6b", "Stellate": "#95a5a6",
        "Macrophage": "#00cec9", "Mast": "#e84393",
        "T cell": "#6c5ce7", "B cell": "#fd79a8",
    }
    PANCREAS_MARKER_MAP = {
        "Beta":     ["INS"],
        "Alpha":    ["GCG"],
        "Delta":    ["SST"],
        "PP":      ["PPY"],
        "Acinar":  ["PRSS1", "AMY2A"],
        "Ductal":  ["KRT19", "CFTR", "KRT7"],
        "ISEC": ISEC_MARKERS,
        "ASEC": ["IGFBP5", "IGFBP3", "COL4A1", "SPARC", "COL1A2", "CLU"],
        "Stellate":  ["LUM", "COL1A1", "DCN"],
        "Mast":    ["TPSAB1", "KIT"],
        "Macrophage": ["CD68", "CD14"],
    }
    # ---------- 辅助函数 ----------
    def annotate_cell_types(adata):
        var_names = list(adata.raw.var_names) if hasattr(adata, "raw") and adata.raw is not None else list(adata.var_names)
        Xr = adata.raw.X if hasattr(adata, "raw") and adata.raw is not None else adata.X
        if hasattr(Xr, "toarray"): Xr = Xr.toarray()
        scores = {}
        for ct, markers in PANCREAS_MARKER_MAP.items():
            found = [v for v in markers if v in var_names]
            if found:
                idx_g = [var_names.index(v) for v in found]
                scores[ct] = np.mean(Xr[:, idx_g], axis=1)
            else:
                scores[ct] = np.zeros(Xr.shape[0] if len(Xr.shape) > 1 else adata.n_obs)
        score_df = pd.DataFrame(scores, index=adata.obs_names)
        adata.obs["cell_type"] = score_df.idxmax(axis=1).values
        return adata
    def get_benchmark_value(gene):
        if bm.empty or gene not in bm["gene"].values:
            return 0.0, 1.0
        r = bm[bm["gene"] == gene].iloc[0]
        return float(r["ISEC_mean"]), float(r["ISEC_std"])
    def compute_t2d_deviation(user_val, bench_mean, bench_std, direction):
        if bench_std < 1e-8:
            return 0.0
        z = (user_val - bench_mean) / bench_std
        if direction == "up":
            return max(0.0, min(1.0, z / 3.0))
        else:
            return max(0.0, min(1.0, -z / 3.0))
    def load_file(f):
        fn = f.name.lower()
        if fn.endswith(".h5ad"):
            return sc.read_h5ad(f)
        elif fn.endswith(".h5"):
            tmp = pathlib.Path(tempfile.mktemp(suffix=".h5"))
            tmp.write_bytes(f.read())
            a = sc.read_10x_h5(str(tmp))
            tmp.unlink(missing_ok=True)
            a.var_names_make_unique()
            return a
        else:
            df = pd.read_csv(f)
            if "Unnamed" in str(df.columns[0]) or df.columns[0] in ["", "gene"]:
                df = df.set_index(df.columns[0])
            else:
                df = df.set_index(df.columns[0])
            isec_check = ["COL13A1","ESM1","PLVAP","KDR","CXCR4"]
            genes_in_idx = sum(1 for g in isec_check if g in df.index[:100])
            genes_in_cols = sum(1 for g in isec_check if g in df.columns[:100])
            if genes_in_idx > genes_in_cols:
                df = df.T
            return sc.AnnData(X=df.values.astype(np.float32),
                              var=pd.DataFrame(index=df.columns),
                              obs=pd.DataFrame(index=df.index))
    def process_sample(adata, mg=200, res=0.8, n_genes=2000, mt_pct=20, min_umi=1000, expr_prop=0.1, n_pcs=20):
        sc.pp.filter_cells(adata, min_genes=mg)
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        adata = adata[adata.obs.pct_counts_mt < mt_pct].copy()
        adata = adata[adata.obs.total_counts > min_umi].copy()
        sc.pp.filter_genes(adata, min_cells=3)
        adata.var_names_make_unique()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)
        adata.raw = adata.copy()
        adata_hvg = adata[:, adata.var.highly_variable].copy()
        sc.pp.pca(adata_hvg, n_comps=n_pcs)
        sc.pp.neighbors(adata_hvg, n_pcs=15, n_neighbors=30)
        sc.tl.leiden(adata_hvg, resolution=res, key_added="leiden")
        adata.obs["leiden"] = adata_hvg.obs["leiden"].values
        sc.tl.umap(adata_hvg, min_dist=0.3, spread=1.0)
        adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
        
        var_names = list(adata.raw.var_names)
        Xr = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else np.array(adata.raw.X)
        adata = annotate_cell_types(adata)
        isec_found = [g for g in ISEC_MARKERS if g in var_names]
        if isec_found:
            isec_s = np.mean(Xr[:, [var_names.index(g) for g in isec_found]], axis=1)
        else:
            isec_s = np.zeros(adata.n_obs)
        adata.obs["isec_score"] = isec_s
        cluster_mean = adata.obs.groupby("leiden")["isec_score"].mean()
        thresh = cluster_mean.quantile(0.75) if len(cluster_mean) > 0 else 0
        isec_clusters = set(cluster_mean[cluster_mean >= thresh].index)
        adata.obs["is_isec"] = adata.obs["leiden"].isin(isec_clusters)
        adata.obs.loc[adata.obs["is_isec"] & (adata.obs["isec_score"] > 0.5), "cell_type"] = "ISEC"
        n_isec = adata.obs["is_isec"].sum()
        isec_mask = adata.obs["is_isec"].values
        isec_X = Xr[isec_mask, :] if isec_mask.sum() > 5 else Xr
        
        gene_expr = {}
        for gene in ALL_T2D_GENES:
            if gene in var_names:
                gidx = var_names.index(gene)
                gene_expr[gene] = float(np.mean(isec_X[:, gidx]))
            else:
                gene_expr[gene] = 0.0
        
        # LIANA 细胞通讯分析
        try:
            if "cell_type" in adata.obs.columns:
                liana.mt.singlecellsignalr(adata, groupby="cell_type", expr_prop=expr_prop, verbose=False, inplace=True, key_added="liana_res")
        except Exception:
            pass
        
        return adata, gene_expr, n_isec
    # ---------- 界面 ----------
    # 文件上传
    c1, c2 = st.columns(2)
    with c1:
        healthy_file = st.file_uploader("健康样本", type=["h5","h5ad","csv"], key="h")
    with c2:
        disease_file = st.file_uploader("疾病样本", type=["h5","h5ad","csv"], key="d")
    if healthy_file is not None and disease_file is not None:
        with st.spinner("Loading files..."):
            adata_h = load_file(healthy_file)
            adata_d = load_file(disease_file)
            common = sorted(set(adata_h.var_names) & set(adata_d.var_names))
            adata_h = adata_h[:, common]
            adata_d = adata_d[:, common]
            st.success(f"Healthy: {adata_h.n_obs:,} cells | Disease: {adata_d.n_obs:,} cells | {len(common):,} genes")
    # 质控参数
    st.markdown("**质控参数**")
    mg = st.slider("nFeature_RNA（最小基因数）", 50, 1000, 200, 50, key="mg")
    min_umi = st.slider("nCount_RNA（最小UMI数）", 200, 5000, 1000, 100, key="min_umi")
    mt_pct = st.slider("percent.mt（最大线粒体比例）", 5, 50, 20, 5, key="mt_pct")
    # 质控分析按钮
    qc_btn = st.button("质量分析", type="primary", key="qc_btn")
    if qc_btn:
        if healthy_file is None or disease_file is None:
            st.warning("请先上传健康与疾病样本")
        else:
            for name_tmp, adata_tmp in [("健康样本", adata_h), ("疾病样本", adata_d)]:
                with st.expander(f"{name_tmp} - {adata_tmp.n_obs:,}细胞", expanded=False):
                    adata_qc = adata_tmp.copy()
                    adata_qc.var["mt"] = adata_qc.var_names.str.startswith("MT-")
                    sc.pp.calculate_qc_metrics(adata_qc, qc_vars=["mt"], inplace=True)
                    st.markdown("Before filtering")
                    fig_qc, axes_qc = plt.subplots(1, 3, figsize=(14, 3.5))
                    for i_tmp, (col_tmp, title_tmp, yl_tmp, th_tmp) in enumerate([
                        ("n_genes_by_counts", "nFeature_RNA", "nFeature_RNA", mg),
                        ("total_counts", "nCount_RNA", "nCount_RNA", min_umi),
                        ("pct_counts_mt", "percent.mt", "percent.mt", mt_pct),
                    ]):
                        vp = axes_qc[i_tmp].violinplot(adata_qc.obs[col_tmp], positions=[0], showmedians=False, showextrema=False, widths=0.6)
                        vp["bodies"][0].set_edgecolor("none"); vp["bodies"][0].set_facecolor("#e74c3c")
                        vp["bodies"][0].set_alpha(0.7)
                        axes_qc[i_tmp].set_xticks([])
                        np.random.seed(42)
                        jitter = np.random.normal(0, 0.04, size=len(adata_qc.obs[col_tmp]))
                        axes_qc[i_tmp].scatter(jitter, adata_qc.obs[col_tmp], s=3, alpha=0.4, color="#333333")
                        axes_qc[i_tmp].set_ylabel(yl_tmp)
                        axes_qc[i_tmp].set_title(title_tmp, fontsize=11)
                        if th_tmp is not None:
                            axes_qc[i_tmp].axhline(th_tmp, color="red", linestyle="--", linewidth=1)
                            axes_qc[i_tmp].text(0.05, th_tmp, f"threshold={th_tmp}", color="red", fontsize=8, transform=axes_qc[i_tmp].get_yaxis_transform())
                    st.pyplot(fig_qc); plt.close()
                    st.markdown("After filtering")
                    adata_filtered = adata_qc[adata_qc.obs.n_genes_by_counts > mg].copy()
                    adata_filtered = adata_filtered[adata_filtered.obs.total_counts > min_umi].copy()
                    adata_filtered = adata_filtered[adata_filtered.obs.pct_counts_mt < mt_pct].copy()
                    fig_qc2, axes_qc2 = plt.subplots(1, 3, figsize=(14, 3.5))
                    for i_tmp, (col_tmp, title_tmp, yl_tmp, _) in enumerate([
                        ("n_genes_by_counts", "nFeature_RNA", "nFeature_RNA", None),
                        ("total_counts", "nCount_RNA", "nCount_RNA", None),
                        ("pct_counts_mt", "percent.mt", "percent.mt", None),
                    ]):
                        vp = axes_qc2[i_tmp].violinplot(adata_filtered.obs[col_tmp], positions=[0],
                                                        showmedians=False, showextrema=False, widths=0.6)
                        vp["bodies"][0].set_edgecolor("none"); vp["bodies"][0].set_facecolor("#2ecc71")
                        vp["bodies"][0].set_alpha(0.7)
                        axes_qc2[i_tmp].set_xticks([])
                        np.random.seed(42)
                        jitter = np.random.normal(0, 0.04, size=len(adata_filtered.obs[col_tmp]))
                        axes_qc2[i_tmp].scatter(jitter, adata_filtered.obs[col_tmp], s=3, alpha=0.4, color="#333333")
                        axes_qc2[i_tmp].set_ylabel(yl_tmp)
                        axes_qc2[i_tmp].set_title(title_tmp, fontsize=11)
                    st.pyplot(fig_qc2); plt.close()
                    st.caption(f"Retained {adata_filtered.n_obs:,}/{adata_qc.n_obs:,} cells after filtering")
    # 分析参数
    st.markdown("**分析参数**")
    res = st.slider("Resolution（聚类分辨率）", 0.1, 2.0, 0.8, 0.1, key="res")
    n_genes_hvg = st.slider("HVGs（高变基因数）", 500, 5000, 2000, 100, key="n_genes_hvg")
    expr_prop = st.slider("expr_prop（表达比例阈值）", 0.01, 0.5, 0.1, 0.01, key="expr_prop")
    n_pcs = st.slider("nPCs（PCA主成分数）", 5, 50, 20, 5, key="n_pcs")
    # 运行分析按钮
    if st.button("运行分析", type="primary", key="run_analysis"):
        if healthy_file is None or disease_file is None:
            st.warning("请先上传健康与疾病样本")
        else:
            with st.spinner("Processing Healthy sample..."):
                adata_h_processed, expr_h, n_h = process_sample(
                    adata_h, mg=mg, res=res, n_genes=n_genes_hvg,
                    mt_pct=mt_pct, min_umi=min_umi, expr_prop=expr_prop, n_pcs=n_pcs
                )
            with st.spinner("Processing Disease sample..."):
                adata_d_processed, expr_d, n_d = process_sample(
                    adata_d, mg=mg, res=res, n_genes=n_genes_hvg,
                    mt_pct=mt_pct, min_umi=min_umi, expr_prop=expr_prop, n_pcs=n_pcs
                )
            
            # 保存到 session_state 便于其它子页面使用
            st.session_state['adata_healthy'] = adata_h_processed
            st.session_state['adata_disease'] = adata_d_processed
            
            # 结果展示
            t1, t2, t3 = st.tabs(["📊 UMAP", "📋 样本概览", "🧪 通讯轴评估"])
            with t1:
                st.subheader("UMAP 细胞分群")
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                for idx, (adata, name) in enumerate([(adata_h_processed, "Healthy"), (adata_d_processed, "Disease")]):
                    ax = axes[idx]
                    if "cell_type" in adata.obs.columns:
                        for ct in adata.obs["cell_type"].value_counts().index.tolist():
                            mask = adata.obs["cell_type"] == ct
                            ax.scatter(adata.obsm["X_umap"][mask, 0], adata.obsm["X_umap"][mask, 1],
                                       c=PANCREAS_CELL_COLORS.get(ct, "#636e72"), label=ct, s=2, alpha=0.4)
                    else:
                        ax.scatter(adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1],
                                   c="#d3d3d3", s=2, alpha=0.4)
                    ax.legend(markerscale=2, fontsize=7, loc="best")
                    ax.set_title(name, fontsize=14)
                    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
                st.pyplot(fig); plt.close()
            with t2:
                st.subheader("样本质控与细胞类型")
                q1, q2 = st.columns(2)
                with q1:
                    st.metric("健康细胞数", f"{adata_h_processed.n_obs:,}")
                    st.metric("健康基因数", f"{adata_h_processed.n_vars:,}")
                with q2:
                    st.metric("疾病细胞数", f"{adata_d_processed.n_obs:,}")
                    st.metric("疾病基因数", f"{adata_d_processed.n_vars:,}")
                ct_h = adata_h_processed.obs["cell_type"].value_counts()
                ct_d = adata_d_processed.obs["cell_type"].value_counts()
                ct_df = pd.DataFrame({
                    "健康 (%)": (ct_h / ct_h.sum() * 100).round(2),
                    "疾病 (%)": (ct_d / ct_d.sum() * 100).round(2)
                }).fillna(0).sort_values("健康 (%)", ascending=False)
                col_a, col_b = st.columns(2)
                with col_a:
                    fig1, ax1 = plt.subplots(figsize=(5, 5))
                    colors_ct = [PANCREAS_CELL_COLORS.get(ct, "#636e72") for ct in ct_df.index]
                    ax1.pie(ct_df["健康 (%)"], labels=None, colors=colors_ct,
                            autopct=lambda p: f"{p:.0f}%" if p >= 10 else "", startangle=90)
                    ax1.set_title("Healthy")
                    st.pyplot(fig1); plt.close()
                with col_b:
                    fig2, ax2 = plt.subplots(figsize=(5, 5))
                    ax2.pie(ct_df["疾病 (%)"], labels=None, colors=colors_ct,
                            autopct=lambda p: f"{p:.0f}%" if p >= 10 else "", startangle=90)
                    ax2.set_title("Disease")
                    st.pyplot(fig2); plt.close()
                st.markdown("**图例**")
                cols_legend = st.columns(4)
                for i, ct in enumerate(ct_df.index):
                    with cols_legend[i % 4]:
                        color = PANCREAS_CELL_COLORS.get(ct, "#636e72")
                        st.markdown(
                            f"<span style='display:inline-block;width:12px;height:12px;background:{color};"
                            f"border-radius:50%;margin-right:4px'></span> {ct}",
                            unsafe_allow_html=True
                        )
                st.dataframe(ct_df, use_container_width=True)
            with t3:
                st.subheader("关键通讯轴对比 (LIANA SingleCellSignalR)")
                st.markdown("分析 3 条 T2D 相关通讯轴：**CXCL12-CXCR4**（血管保护）、**VEGF-A-KDR**（血管生成）、**DLL4-NOTCH1**（内皮成熟）")
                def extract_axis(res_df, ligand, receptor):
                    if res_df is None or len(res_df) == 0: return None
                    mask = res_df["ligand_complex"].apply(lambda x: ligand in x) & res_df["receptor_complex"].apply(lambda x: receptor in x)
                    return res_df[mask] if len(res_df[mask]) > 0 else None
                axes_def = [
                    ("CXCL12-CXCR4", "Vascular Protection", "down",
                     "LUM+ 基质细胞分泌 CXCL12，ISEC 通过 CXCR4 接收血管保护信号，维持胰岛微血管稳态和 β 细胞功能。"),
                    ("VEGF-A-KDR", "Angiogenesis", "up",
                     "β 细胞和基质细胞分泌 VEGF-A，与 ISEC 表面 KDR 结合，引导窗孔形成、维持血管通透性和营养交换。"),
                    ("DLL4-NOTCH1", "EC Maturation", "up",
                     "ISEC 内部的 DLL4-NOTCH1 信号调控内皮细胞成熟、血管新生过程中的尖端细胞-柄细胞分化。"),
                ]
                res_h = adata_h_processed.uns.get("liana_res", None)
                res_d = adata_d_processed.uns.get("liana_res", None)
                for name, label, direction, desc in axes_def:
                    ligand, receptor = name.split("-")
                    sub_h = extract_axis(res_h, ligand, receptor)
                    sub_d = extract_axis(res_d, ligand, receptor)
                    s_h = sub_h["lrscore"].mean() if sub_h is not None else 0
                    s_d = sub_d["lrscore"].mean() if sub_d is not None else 0
                    # 计算变化
                    if s_h > 0 and s_d > 0:
                        ratio = s_d / s_h
                        change = f"{ratio-1:+.0%}"
                    elif s_h > 0 and s_d == 0:
                        change, ratio = "lost", 0
                    elif s_h == 0 and s_d > 0:
                        change, ratio = "new", 999
                    else:
                        change, ratio = "N/A", 1
                    is_t2d = False
                    if direction == "down":
                        is_t2d = (s_h > 0 and s_d == 0) or (s_h > 0 and s_d > 0 and ratio < 0.8)
                    else:
                        is_t2d = (s_h == 0 and s_d > 0) or (s_h > 0 and s_d > 0 and ratio > 1.2)
                    bio_meanings = {
                        "CXCL12-CXCR4": {
                            "up": "T2D 中通讯异常上调，可能反映代偿反应。",
                            "down": "通讯丢失或显著减弱，提示 ISEC 失去血管保护信号，是 T2D 微环境破坏的核心事件。",
                            "normal": "无明显变化，血管保护信号完整。"
                        },
                        "VEGF-A-KDR": {
                            "up": "通讯上调，提示异常血管生成信号，导致窗孔结构紊乱。",
                            "down": "通讯下调，可能导致 ISEC 窗孔形成不足。",
                            "normal": "无明显变化，VEGF-A 信号调控正常。"
                        },
                        "DLL4-NOTCH1": {
                            "up": "通讯上调，提示内皮成熟程序紊乱。",
                            "down": "通讯下调，可能影响内皮分化和血管成熟。",
                            "normal": "无明显变化，内皮成熟信号正常。"
                        }
                    }
                    # 构建文字说明
                    if s_h > 0 and s_d > 0:
                        base = f"健康通讯强度 {s_h:.2f}，疾病通讯强度 {s_d:.2f}，相对变化 {ratio-1:.0%}。"
                        tag = bio_meanings[name]["up"] if direction == "up" and is_t2d else \
                              bio_meanings[name]["down"] if direction == "down" and is_t2d else \
                              bio_meanings[name]["normal"]
                        analysis = base + " " + tag
                    elif s_h > 0 and s_d == 0:
                        analysis = f"健康通讯强度 {s_h:.2f}，疾病中通讯完全丢失。" + bio_meanings[name]["down"]
                    elif s_h == 0 and s_d > 0:
                        analysis = f"健康中未检测到通讯，疾病中出现新通讯（强度 {s_d:.2f}）。" + bio_meanings[name]["up"]
                    else:
                        analysis = "健康与疾病样本中均未检测到该通讯通路活性。"
                    change_color = "#c0392b" if change in ["lost", "new"] or (isinstance(change, str) and change.startswith("+") and change != "+0%") else "#27ae60"
                    html_box = f"""
                    <div style='border:2px solid #888;border-radius:8px;padding:15px;margin-bottom:18px;background:#fafafa;'>
                        <p style='font-size:16px;font-weight:bold;margin:0 0 4px 0;'>{name} ({label})</p>
                        <p style='font-size:13px;color:#666;margin:0 0 10px 0;'>{desc}</p>
                        <table style='width:100%;border:none;'>
                            <tr>
                                <td style='width:35%;vertical-align:top;border:none;padding-right:10px;'>
                                    <p style='font-size:13px;color:#888;margin:0;'>通讯强度</p>
                                    <p style='font-size:20px;font-weight:bold;margin:2px 0;'>{s_h:.2f} → {s_d:.2f}</p>
                                    <p style='font-size:14px;color:{change_color};margin:0;'>{change}</p>
                                </td>
                                <td style='width:65%;vertical-align:top;border:none;padding-left:10px;border-left:1px solid #ddd;'>
                                    <p style='font-size:14px;margin:0;line-height:1.6;'>{analysis}</p>
                                </td>
                            </tr>
                        </table>
                    </div>
                    """
                    st.markdown(html_box, unsafe_allow_html=True)
                    
# TAB 5：糖尿病用药推荐（新整合模块）
# ============================================================
with tab5:
    st.caption("基于《中国糖尿病防治指南（2024版）》及《成人2型糖尿病口服降糖药联合治疗专家共识（2025版）》")

    # ==================== 数据库加载 ====================
    @st.cache_data
    def load_drug_db():
        return pd.read_csv("drugs_db.csv", encoding="utf-8-sig")

    @st.cache_data
    def load_fdc_db():
        try:
            df = pd.read_csv("fdc_db.csv", encoding="utf-8-sig")
            df["components_list"] = df["components"].apply(lambda x: [c.strip() for c in x.split(",")])
            return df
        except FileNotFoundError:
            return pd.DataFrame()

    @st.cache_data
    def load_recommendation_rules():
        return pd.read_csv("drug_recommendation.csv", encoding="utf-8-sig")

    df_drugs = load_drug_db()
    df_fdc = load_fdc_db()
    df_rules = load_recommendation_rules()

    FDC_CLINICAL_VALUE = (
        "FDC在指南中的临床价值（2025联合治疗共识，推荐意见17，IIa类推荐）：\n"
        "1. 单药治疗血糖控制不达标者，可转换为FDC；\n"
        "2. HbA1c较高需起始联合治疗的新诊断患者，可直接起始FDC；\n"
        "3. 正在服用自由联合方案需简化治疗者，可转换为FDC。\n"
        "FDC的优势：减少给药频率和漏服概率，提高依从性，是自由联合方案的合理替代选择。"
    )

    # ==================== 公共辅助函数 ====================
    def find_matching_rule(rule_df, recommendation, patient_features):
        if recommendation.get("triple") and len(recommendation["triple"]) > 0:
            level = "triple"
        elif recommendation.get("dual") and len(recommendation["dual"]) > 0:
            level = "dual"
        else:
            level = "mono"

        if level == "triple":
            candidates = rule_df[rule_df["场景分类"].str.startswith("三联_")]
        elif level == "dual":
            candidates = rule_df[rule_df["场景分类"].str.startswith("联合_")]
        else:
            candidates = rule_df[rule_df["场景分类"].str.startswith("单药_")]

        if candidates.empty:
            return None, None, None, None

        for _, row in candidates.iterrows():
            condition = row["触发条件"]
            if evaluate_condition(condition, patient_features):
                return row["场景分类"], row["证据等级"], row["详细依据"], condition

        if not candidates.empty:
            row = candidates.iloc[0]
            return row["场景分类"], row["证据等级"], row["详细依据"], row["触发条件"]
        return None, None, None, None

    def evaluate_condition(condition_str, features):
        if not condition_str or condition_str == "无":
            return False
        sub_conditions = condition_str.split(" 或 ")
        for cond in sub_conditions:
            cond = cond.strip()
            numbers = re.findall(r'(\d+\.?\d*)', cond)
            if "BMI ≥" in cond:
                if numbers and features.get("bmi", 0) >= float(numbers[0]):
                    return True
            elif "年龄≥" in cond:
                if numbers and features.get("age", 0) >= float(numbers[0]):
                    return True
            elif "合并ASCVD" in cond or "ASCVD" in cond:
                if features.get("has_ascvd", False):
                    return True
            elif "合并HF" in cond or "心衰" in cond:
                if features.get("has_hf", False):
                    return True
            elif "合并CKD" in cond or "CKD" in cond:
                if features.get("has_ckd", False):
                    return True
            elif "低血糖风险高" in cond:
                if features.get("high_hypo_risk", False):
                    return True
            elif "餐后血糖显著升高" in cond:
                if features.get("postprandial_high", False):
                    return True
            elif "明显胰岛素抵抗" in cond:
                if features.get("insulin_resistance", False):
                    return True
            elif "β细胞功能较好" in cond or "β细胞功能好" in cond:
                if features.get("beta_cell_good", False):
                    return True
            elif "二甲双胍不耐受" in cond:
                if features.get("met_intolerance", False):
                    return True
            elif "初诊HbA1c≥7.5%" in cond:
                if features.get("initial_high", False):
                    return True
            elif "单药治疗3个月未达标" in cond:
                if features.get("single_failure", False):
                    return True
            elif "二联治疗3个月未达标" in cond:
                if features.get("dual_failure", False):
                    return True
            elif "二联未达标且胰岛素抵抗明显" in cond:
                if features.get("dual_failure", False) and features.get("insulin_resistance", False):
                    return True
            elif "无特殊并发症且不胖" in cond:
                return True
        return False

    def find_fdc_for_combo(combo_string):
        if df_fdc.empty:
            return []
        combo_lower = combo_string.lower()
        matched = []
        for _, row in df_fdc.iterrows():
            fdc_type = row.get("fdc_type", "").lower()
            if "sglt2" in combo_lower and "sglt2" in fdc_type:
                matched.append(row.to_dict())
            elif "dpp-4" in combo_lower and "dpp-4" in fdc_type:
                matched.append(row.to_dict())
            elif "tzd" in combo_lower and "tzd" in fdc_type:
                matched.append(row.to_dict())
            elif "su" in combo_lower and "su" in fdc_type:
                matched.append(row.to_dict())
        return matched

    def parse_renal_rule(text, egfr):
        if pd.isna(text) or text == "" or text == "可用":
            return "可用"
        text = str(text)
        if "禁用" in text:
            patterns = [
                r"eGFR\s*<\s*(\d+)",
                r"<(\d+)",
                r"(\d+)\s*-\s*(\d+)\s*禁用",
                r"(\d+)\s*以下禁用"
            ]
            for pat in patterns:
                matches = re.findall(pat, text)
                for m in matches:
                    if isinstance(m, tuple):
                        if len(m) == 1:
                            if egfr < float(m[0]):
                                return "禁用"
                        elif len(m) == 2:
                            low, high = float(m[0]), float(m[1])
                            if low <= egfr <= high:
                                return "禁用"
                    else:
                        if egfr < float(m):
                            return "禁用"
            if "eGFR<30" in text and egfr < 30:
                return "禁用"
            if "eGFR<25" in text and egfr < 25:
                return "禁用"
            if "eGFR<20" in text and egfr < 20:
                return "禁用"
            if "eGFR<15" in text and egfr < 15:
                return "禁用"
        if "减量慎用" in text or "减量" in text:
            patterns = [
                r"eGFR\s*(\d+)\s*-\s*(\d+)\s*减量",
                r"(\d+)\s*-\s*(\d+)\s*减量慎用",
                r"eGFR\s*<\s*(\d+)\s*减量",
                r"<(\d+)\s*减量"
            ]
            for pat in patterns:
                matches = re.findall(pat, text)
                for m in matches:
                    if isinstance(m, tuple):
                        if len(m) == 2:
                            low, high = float(m[0]), float(m[1])
                            if low <= egfr <= high:
                                return "减量慎用"
                    else:
                        if egfr < float(m):
                            return "减量慎用"
            if "45-59" in text and 45 <= egfr <= 59:
                return "减量慎用"
            if "30-44" in text and 30 <= egfr <= 44:
                return "减量慎用"
            if "eGFR<45" in text and egfr < 45:
                return "减量慎用"
            if "eGFR<30" in text and egfr < 30:
                return "减量慎用"
            if "eGFR<25" in text and egfr < 25:
                return "减量慎用"
            if "eGFR<20" in text and egfr < 20:
                return "减量慎用"
            if "eGFR<15" in text and egfr < 15:
                return "减量慎用"
        if "慎用" in text and "减量" not in text:
            return "慎用"
        return "可用"

    def parse_hepatic_rule(text, child_pugh):
        if pd.isna(text) or text == "" or text == "可用":
            return "可用"
        text = str(text)
        if "C禁用" in text and child_pugh == "C级（重度损伤）":
            return "禁用"
        if "B级" in text and child_pugh == "B级（中度损伤）":
            if "禁用" in text:
                return "禁用"
            elif "减量" in text:
                return "减量慎用"
            else:
                return "慎用"
        if "A级" in text and child_pugh == "A级（轻度损伤）":
            if "禁用" in text:
                return "禁用"
            elif "减量" in text:
                return "减量慎用"
            else:
                return "可用"
        if "Child-Pugh A/B" in text and child_pugh in ["A级（轻度损伤）", "B级（中度损伤）"]:
            return "可用"
        if "Child-Pugh C" in text and child_pugh == "C级（重度损伤）":
            return "禁用"
        if "ALT" in text or "AST" in text:
            return "需监测肝功能"
        return "可用"

    # ==================== 患者信息输入 ====================
    st.header("患者临床特征")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("年龄（岁）", 18, 100, 55)
        bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, 24.0)
        hba1c_current = st.number_input("当前 HbA1c (%)", 5.0, 15.0, 7.5, step=0.1)

    with col2:
        st.subheader("合并症与高危因素")
        has_ascvd = st.checkbox("合并 ASCVD")
        has_hf = st.checkbox("合并 心衰 (HF)")
        has_ckd = st.checkbox("合并 慢性肾脏病 (CKD)")

    with col3:
        st.subheader("临床特征")
        high_hypo_risk = st.checkbox("低血糖风险高")
        postprandial_high = st.checkbox("餐后血糖显著升高")
        insulin_resistance = st.checkbox("明显胰岛素抵抗")
        beta_cell_good = st.checkbox("β细胞功能较好")
        met_intolerance = st.checkbox("二甲双胍不耐受")

    st.subheader("治疗阶段")
    treatment_stage = st.radio(
        "当前阶段",
        ["未用药/初始治疗", "单药治疗 ≥3个月", "二联治疗 ≥3个月"],
        horizontal=True
    )

    hba1c_after_single = None
    hba1c_after_dual = None
    if treatment_stage == "单药治疗 ≥3个月":
        hba1c_after_single = st.number_input("单药后 HbA1c (%)", 5.0, 15.0, 7.5, step=0.1)
    elif treatment_stage == "二联治疗 ≥3个月":
        hba1c_after_dual = st.number_input("二联后 HbA1c (%)", 5.0, 15.0, 7.8, step=0.1)

    # ==================== 推荐方案生成 ====================
    def generate_recommendation():
        rec = recommend_drugs(
            age=age,
            bmi=bmi,
            hba1c_current=hba1c_current,
            has_ascvd=has_ascvd,
            has_hf=has_hf,
            has_ckd=has_ckd,
            high_hypo_risk=high_hypo_risk,
            postprandial_high=postprandial_high,
            insulin_resistance=insulin_resistance,
            beta_cell_good=beta_cell_good,
            met_intolerance=met_intolerance,
            treatment_stage=treatment_stage,
            hba1c_after_single=hba1c_after_single,
            hba1c_after_dual=hba1c_after_dual
        )
        st.session_state['recommendation'] = rec
        # 同时保存患者特征
        pf = {
            "age": age,
            "bmi": bmi,
            "hba1c_current": hba1c_current,
            "has_ascvd": has_ascvd,
            "has_hf": has_hf,
            "has_ckd": has_ckd,
            "high_hypo_risk": high_hypo_risk,
            "postprandial_high": postprandial_high,
            "insulin_resistance": insulin_resistance,
            "beta_cell_good": beta_cell_good,
            "met_intolerance": met_intolerance,
            "treatment_stage": treatment_stage,
            "single_failure": rec["single_failure"],
            "dual_failure": rec["dual_failure"],
            "initial_high": (treatment_stage == "未用药/初始治疗" and hba1c_current >= 7.5)
        }
        st.session_state['patient_features'] = pf
        st.session_state['hba1c_after_single'] = hba1c_after_single
        st.session_state['hba1c_after_dual'] = hba1c_after_dual

    st.button("生成推荐方案", type="primary", use_container_width=True, on_click=generate_recommendation)

    # ==================== 结果展示（子标签页） ====================
    drug_tab1, drug_tab2, drug_tab3, drug_tab4, drug_tab5 = st.tabs([
        "药物信息浏览",
        "药物对比分析",
        "个体化用药推荐",
        "肝肾功能剂量调整",
        "药物相互作用检查"
    ])

    # ---------- 子tab1: 药物信息浏览 ----------
    with drug_tab1:
        st.subheader("药物信息浏览")
        drug_classes = df_drugs["drug_class"].unique().tolist()
        selected_class = st.selectbox("按药物分类筛选", ["全部"] + drug_classes)
        display_df = df_drugs if selected_class == "全部" else df_drugs[df_drugs["drug_class"] == selected_class]
        st.write(f"共 {len(display_df)} 种药物")

        cols = st.columns(3)
        for idx, (_, row) in enumerate(display_df.iterrows()):
            with cols[idx % 3]:
                with st.expander(f"{row['drug_name_cn']} ({row['drug_class']})"):
                    st.markdown(f"**英文名**：{row['drug_name_en']}")
                    st.markdown(f"**作用机制**：{row['mechanism']}")
                    st.markdown(f"**靶点**：{row['target']}")
                    st.markdown(f"**降糖效力**：{row['hba1c_reduction']}")
                    st.markdown(f"**低血糖风险**：{row['hypoglycemia_risk']}")
                    st.markdown(f"**体重影响**：{row['weight_effect']}")
                    st.markdown(f"**心肾获益**：{row['cv_benefit']}")
                    st.markdown(f"**禁忌症**：{row['contraindications']}")
                    if pd.notna(row['cid']):
                        try:
                            st.image(
                                f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={int(row['cid'])}&t=l",
                                caption="分子结构", width=150
                            )
                        except:
                            pass

        st.markdown("---")
        with st.expander("查看所有固定剂量复方制剂（FDC）"):
            st.markdown(FDC_CLINICAL_VALUE)
            st.markdown("---")
            if not df_fdc.empty:
                st.dataframe(
                    df_fdc[["name", "components", "specification", "drug_class", "advantages", "caution"]],
                    use_container_width=True,
                    hide_index=True
                )
            st.caption("当推荐方案匹配到FDC时，会在「个体化用药推荐」结果中自动提示简化方案")

    # ---------- 子tab2: 药物对比分析 ----------
    with drug_tab2:
        st.subheader("药物对比分析")
        all_drug_names = df_drugs["drug_name_cn"].tolist()
        selected_drugs = st.multiselect("选择2-4种药物进行对比", all_drug_names, default=all_drug_names[:3])

        if len(selected_drugs) >= 2:
            compare_df = df_drugs[df_drugs["drug_name_cn"].isin(selected_drugs)]
            st.write("多维对比表")
            display_cols = ["drug_name_cn", "drug_class", "mechanism", "target", "hba1c_reduction",
                            "hypoglycemia_risk", "weight_effect", "cv_benefit", "contraindications"]
            st.dataframe(compare_df[display_cols], use_container_width=True, hide_index=True)

            st.write("能力雷达图")
            score_map = {
                "低": 3, "较低": 2.5, "中等": 2, "较高": 1, "高": 0,
                "中性或轻度减轻": 2, "轻度减轻": 2, "减轻": 2.5, "显著减轻": 3,
                "中性": 1.5, "增加": 0,
                "获益": 3, "潜在获益": 2, "证据不足": 0,
            }
            fig = go.Figure()
            dimensions = ["降糖效力", "低血糖安全", "体重获益", "心肾获益"]
            for _, row in compare_df.iterrows():
                scores = []
                hba1c = row["hba1c_reduction"]
                if "%~" in str(hba1c):
                    nums = [float(x) for x in str(hba1c).replace("%", "").split("~")]
                    scores.append((nums[0] + nums[1]) / 2)
                else:
                    scores.append(0.5)
                scores.append(score_map.get(row["hypoglycemia_risk"], 1))
                scores.append(score_map.get(row["weight_effect"], 1))
                scores.append(score_map.get(row["cv_benefit"], 1))
                fig.add_trace(go.Scatterpolar(r=scores, theta=dimensions, fill='toself', name=row["drug_name_cn"]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 3.5])), height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("请至少选择2种药物进行对比")

    # ---------- 子tab3: 个体化用药推荐 ----------
    with drug_tab3:
        st.subheader("个体化用药推荐")
        st.caption("基于《中国糖尿病防治指南（2024版）》及《成人2型糖尿病口服降糖药联合治疗专家共识（2025版）》")

        if 'recommendation' in st.session_state:
            recommendation = st.session_state['recommendation']
            patient_features = st.session_state.get('patient_features', {})
            if not patient_features:
                patient_features = {
                    "age": age,
                    "bmi": bmi,
                    "hba1c_current": hba1c_current,
                    "has_ascvd": has_ascvd,
                    "has_hf": has_hf,
                    "has_ckd": has_ckd,
                    "high_hypo_risk": high_hypo_risk,
                    "postprandial_high": postprandial_high,
                    "insulin_resistance": insulin_resistance,
                    "beta_cell_good": beta_cell_good,
                    "met_intolerance": met_intolerance,
                    "treatment_stage": treatment_stage,
                    "single_failure": recommendation["single_failure"],
                    "dual_failure": recommendation["dual_failure"],
                    "initial_high": (treatment_stage == "未用药/初始治疗" and hba1c_current >= 7.5)
                }
                st.session_state['patient_features'] = patient_features

            scenario, evidence_level, evidence_detail, trigger = find_matching_rule(
                df_rules, recommendation, patient_features
            )

            if recommendation["single_failure"]:
                st.warning("单药治疗未达标（HbA1c ≥ 7.0%），建议启动二联治疗")
            if recommendation["dual_failure"]:
                st.warning("二联治疗未达标（HbA1c ≥ 7.0%），建议启动三联治疗")

            st.markdown("### 决策依据")
            if scenario:
                st.markdown(
                    f"""
                    <div style="border:2px solid #4CAF50; border-radius:8px; padding:16px; background-color:#f0fff4; margin-bottom:16px;">
                        <p style="font-size:16px; font-weight:bold; color:#2E7D32;">匹配场景：{scenario}</p>
                        <p><strong>触发条件：</strong>{trigger}</p>
                        <p><strong>证据等级：</strong><span style="background-color:#FFD700; padding:2px 8px; border-radius:4px; font-weight:bold;">{evidence_level}</span></p>
                        <p><strong>详细依据：</strong>{evidence_detail}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("未能从规则库中精确匹配场景，以下推荐基于指南常规路径。")
            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**单药推荐**")
                if recommendation["mono"]:
                    for drug in recommendation["mono"]:
                        st.markdown(
                            f"""<div style="border:1px solid #4CAF50; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#E8F5E9;">
                                <span style="font-size:16px; font-weight:bold; color:#2E7D32;">{drug}</span>
                            </div>""",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("暂不需要单药治疗")

            with col2:
                st.markdown("**二联推荐**")
                if recommendation["dual"]:
                    for combo in recommendation["dual"]:
                        st.markdown(
                            f"""<div style="border:1px solid #2196F3; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#E3F2FD;">
                                <span style="font-size:16px; font-weight:bold; color:#0D47A1;">{combo}</span>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        fdc_matches = find_fdc_for_combo(combo)
                        if fdc_matches:
                            for fdc in fdc_matches:
                                st.markdown(
                                    f"""<div style="border:1px dashed #FF9800; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#FFF3E0; margin-left:12px;">
                                        <span style="font-weight:bold;">简化方案（FDC）：</span>{fdc['name']}
                                        <br><span style="font-size:13px; color:#555;">成分：{fdc['components']} | 规格：{fdc['specification']}</span>
                                        <br><span style="font-size:13px; color:#555;">优势：{fdc['advantages']} | 注意：{fdc['caution']}</span>
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                            st.caption("FDC将两种药物制成单片，每日1次，可显著提高依从性（指南IIa类推荐）")
                else:
                    st.info("暂不需要二联治疗")

            with col3:
                st.markdown("**三联推荐**")
                if recommendation["triple"]:
                    for combo in recommendation["triple"]:
                        st.markdown(
                            f"""<div style="border:1px solid #F44336; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#FFEBEE;">
                                <span style="font-size:16px; font-weight:bold; color:#B71C1C;">{combo}</span>
                            </div>""",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("暂不需要三联治疗")

            # 推荐药物详细指导
            st.markdown("---")
            st.subheader("推荐药物详细指导")
            category_keywords = {"DPP-4i", "SGLT2i", "SU", "GLN", "AGi", "TZD", "GKA", "GLP-1RA", "pan-PPARA", "双胍类"}
            all_recommended_raw = recommendation["mono"] + recommendation["dual"] + recommendation["triple"]
            valid_drugs_set = set()
            invalid_categories_set = set()

            for item in all_recommended_raw:
                parts = re.split(r'[+、，, 或]', item)
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    if p in df_drugs["drug_name_cn"].values:
                        valid_drugs_set.add(p)
                        continue
                    match = re.search(r'（([^）]+)）', p)
                    if match:
                        inner = match.group(1).strip()
                        if inner in df_drugs["drug_name_cn"].values:
                            valid_drugs_set.add(inner)
                            continue
                        found = False
                        for drug in df_drugs["drug_name_cn"].values:
                            if inner in drug or drug in inner:
                                valid_drugs_set.add(drug)
                                found = True
                                break
                        if found:
                            continue
                    clean_p = re.sub(r'[（()）]', '', p).strip()
                    if clean_p in category_keywords:
                        invalid_categories_set.add(clean_p)
                        continue
                    found = False
                    for drug in df_drugs["drug_name_cn"].values:
                        if clean_p in drug or drug in clean_p:
                            valid_drugs_set.add(drug)
                            found = True
                            break
                    if found:
                        continue

            valid_drugs = sorted(list(valid_drugs_set))
            if invalid_categories_set:
                st.info(f"以下推荐为纯药物类别（{', '.join(sorted(invalid_categories_set))}），无法查看单一药物详情，请参照具体药物名称。")

            if valid_drugs:
                guide_drug = st.selectbox(
                    "选择推荐中的具体药物查看详细指导卡片",
                    ["请选择"] + valid_drugs,
                    key="guide_in_tab3"
                )
                if guide_drug and guide_drug != "请选择":
                    matched = df_drugs[df_drugs["drug_name_cn"] == guide_drug]
                    if not matched.empty:
                        row = matched.iloc[0]
                        st.markdown(
                            f"""<div style="border:2px solid #2196F3; border-radius:8px; padding:16px; background-color:#F5F9FF; margin-top:8px;">
                                <h4 style="color:#0D47A1;">{row['drug_name_cn']} ({row['drug_name_en']})</h4>
                                <table style="width:100%; border-collapse:collapse; font-size:14px;">
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">药物分类</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['drug_class']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">作用机制</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['mechanism']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">靶点</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['target']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">降糖效力</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['hba1c_reduction']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">低血糖风险</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['hypoglycemia_risk']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">体重影响</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['weight_effect']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">心肾获益</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['cv_benefit']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">禁忌症</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['contraindications']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">肾功能调整</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['renal_dose_adjustment']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">肝功能使用</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['hepatic_use']}</td></tr>
                                </table>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        if pd.notna(row['cid']):
                            try:
                                st.image(f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={int(row['cid'])}&t=l", caption="分子结构", width=150)
                            except:
                                pass
                    else:
                        st.info(f"未找到 {guide_drug} 的详细信息")
            else:
                st.info("暂无具体的推荐药物可查看详情")
        else:
            st.info("请填写患者特征后，点击「生成推荐方案」")

    # ---------- 子tab4: 肝肾功能剂量调整 ----------
    with drug_tab4:
        st.subheader("肝肾功能剂量调整")
        st.markdown(
            """<div style="border:1px solid #FF9800; border-radius:6px; padding:12px; background-color:#FFF8E1; margin-bottom:16px;">
                <strong>功能说明：</strong>输入患者的肾功能（eGFR）和肝功能分级，系统将自动根据CSV中的剂量调整建议，判断每种降糖药是否需要调整剂量或禁用。
                数据来源于《成人2型糖尿病口服降糖药联合治疗专家共识（2025版）》,  eGFR 数值对应分级基于 KDIGO 指南的 CKD 分期推荐的代表性值。
            </div>""",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=0, max_value=120, value=60, step=5)
            st.caption("参考范围：≥60正常，45-59轻中度下降，30-44中重度下降，15-29重度下降，<15肾衰竭")
        with col2:
            child_pugh = st.selectbox(
                "肝功能分级 (Child-Pugh)",
                ["A级（轻度损伤）", "B级（中度损伤）", "C级（重度损伤）", "不详/未评估"]
            )

        if st.button("评估所有药物的剂量调整建议", type="primary"):
            st.markdown("### 剂量调整评估结果")
            st.caption("根据CSV中的`renal_dose_adjustment`和`hepatic_use`字段进行匹配，颜色标识：绿色=可用，黄色=减量慎用，红色=禁用")
            results = []
            for _, row in df_drugs.iterrows():
                drug_name = row["drug_name_cn"]
                renal_text = row["renal_dose_adjustment"]
                hepatic_text = row["hepatic_use"]
                renal_advice = parse_renal_rule(renal_text, egfr)
                hepatic_advice = parse_hepatic_rule(hepatic_text, child_pugh)

                if "禁用" in renal_advice or "禁用" in hepatic_advice:
                    status = "禁用"
                    color = "#FFEBEE"
                    border = "#F44336"
                elif "需监测肝功能" in hepatic_advice:
                    status = "减量慎用"
                    color = "#FFF8E1"
                    border = "#FF9800"
                elif "减量慎用" in renal_advice or "减量慎用" in hepatic_advice or "慎用" in renal_advice or "慎用" in hepatic_advice:
                    status = "减量慎用"
                    color = "#FFF8E1"
                    border = "#FF9800"
                else:
                    status = "可用"
                    color = "#E8F5E9"
                    border = "#4CAF50"

                results.append({
                    "药物": drug_name,
                    "分类": row["drug_class"],
                    "eGFR建议": renal_advice,
                    "肝功能建议": hepatic_advice,
                    "综合建议": status,
                    "color": color,
                    "border": border
                })

            status_order = {"禁用": 0, "减量慎用": 1, "可用": 2}
            results_sorted = sorted(results, key=lambda x: status_order[x["综合建议"]])
            for r in results_sorted:
                st.markdown(
                    f"""<div style="border-left:6px solid {r['border']}; border-radius:4px; padding:10px 14px; margin-bottom:8px; background-color:{r['color']};">
                        <span style="font-size:16px; font-weight:bold;">{r['药物']}</span>
                        <span style="font-size:13px; color:#555; margin-left:12px;">{r['分类']}</span>
                        <span style="float:right; font-weight:bold; color:{r['border']};">{r['综合建议']}</span>
                        <br><span style="font-size:13px; color:#555;">eGFR建议：{r['eGFR建议']} ｜ 肝功能建议：{r['肝功能建议']}</span>
                    </div>""",
                    unsafe_allow_html=True
                )
            st.caption("提示：具体建议根据CSV字段解析，如有疑问请参考药品说明书。")

    # ---------- 子tab5: 药物相互作用检查 ----------
    with drug_tab5:
        st.subheader("药物相互作用检查")
        st.caption("选择患者当前正在使用的所有药物，系统将检查已知的相互作用")

        all_drugs_for_interaction = df_drugs["drug_name_cn"].tolist() + [
            "ACEI类降压药", "ARB类降压药", "他汀类降脂药", "贝特类降脂药",
            "阿司匹林", "氯吡格雷", "利尿剂", "β受体阻滞剂",
            "胰岛素", "碘化对比剂（造影检查）", "糖皮质激素"
        ]
        current_drugs = st.multiselect("选择患者正在使用的药物", all_drugs_for_interaction)

        if current_drugs:
            st.write(f"**当前用药**：{'、'.join(current_drugs)}")

            interactions = []

            if any("二甲双胍" in d for d in current_drugs) and any("碘化对比剂" in d for d in current_drugs):
                interactions.append({
                    "level": "谨慎",
                    "color": "#FF9800",
                    "bg": "#FFF8E1",
                    "drugs": "二甲双胍 + 碘化对比剂",
                    "advice": "检查当日停用二甲双胍，检查完至少48h后复查肾功能无变化方可继续使用",
                    "evidence": "二甲双胍临床应用专家共识"
                })

            sglt2_drugs = ["恩格列净", "达格列净", "卡格列净", "艾托格列净", "恒格列净", "加格列净"]
            if any(d in sglt2_drugs for d in current_drugs):
                if any("糖皮质激素" in d for d in current_drugs):
                    interactions.append({
                        "level": "谨慎",
                        "color": "#FF9800",
                        "bg": "#FFF8E1",
                        "drugs": "SGLT2i + 糖皮质激素",
                        "advice": "糖皮质激素可升高血糖，需加强血糖监测，可能需要增加降糖药剂量",
                        "evidence": "OAD联合治疗注意事项"
                    })

            if any("沙格列汀" in d for d in current_drugs) and has_hf:
                interactions.append({
                    "level": "谨慎",
                    "color": "#FF9800",
                    "bg": "#FFF8E1",
                    "drugs": "沙格列汀 + 心衰",
                    "advice": "SAVOR研究显示沙格列汀可能增加心衰住院风险，有心衰诱发因素的患者慎用",
                    "evidence": "DPP-4i CVOT研究结果"
                })

            su_drugs = ["格列美脲", "格列吡嗪", "格列齐特", "格列喹酮"]
            if any(d in su_drugs for d in current_drugs):
                if age >= 65:
                    interactions.append({
                        "level": "谨慎",
                        "color": "#FF9800",
                        "bg": "#FFF8E1",
                        "drugs": "SU类 + 高龄(≥65岁)",
                        "advice": "老年患者低血糖感知能力下降，建议减量或换用低血糖风险更小的药物",
                        "evidence": "老年T2DM用药注意事项"
                    })
                if high_hypo_risk:
                    interactions.append({
                        "level": "高警示",
                        "color": "#F44336",
                        "bg": "#FFEBEE",
                        "drugs": "SU类 + 低血糖高危",
                        "advice": "患者有低血糖高危因素，使用SU类药物风险显著增加，建议换用DPP-4i或SGLT2i",
                        "evidence": "OAD联合治疗注意事项"
                    })

            tzd_drugs = ["吡格列酮", "罗格列酮"]
            if any(d in tzd_drugs for d in current_drugs):
                if has_hf:
                    interactions.append({
                        "level": "禁忌",
                        "color": "#F44336",
                        "bg": "#FFEBEE",
                        "drugs": "TZD + 心衰",
                        "advice": "TZD类药物有水钠潴留风险，NYHA II级以上心衰患者禁用",
                        "evidence": "TZD使用注意事项"
                    })
                if age >= 65:
                    interactions.append({
                        "level": "谨慎",
                        "color": "#FF9800",
                        "bg": "#FFF8E1",
                        "drugs": "TZD + 高龄(≥65岁)",
                        "advice": "老年患者骨质疏松和骨折风险增加，TZD慎用",
                        "evidence": "老年T2DM用药注意事项"
                    })

            if interactions:
                st.subheader("相互作用检查结果")
                for inter in interactions:
                    st.markdown(
                        f"""<div style="border-left:6px solid {inter['color']}; border-radius:4px; padding:12px 16px; margin-bottom:12px; background-color:{inter['bg']};">
                            <span style="font-weight:bold; font-size:16px; color:{inter['color']};">{inter['level']}</span>
                            <span style="font-weight:bold; margin-left:12px;">{inter['drugs']}</span>
                            <br><span style="color:#333;">{inter['advice']}</span>
                            <br><span style="font-size:13px; color:#777;">依据：{inter['evidence']}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
            else:
                st.success("未发现已知的药物相互作用，当前用药方案安全。")

            st.info("通用建议：任何用药方案调整后，均应加强血糖监测，特别是联合使用胰岛素促泌剂（SU/格列奈类）或胰岛素时。")
        else:
            st.info("请选择患者正在使用的药物进行相互作用检查")

# TAB 6： T2D 胰岛转录组科研检索平台（新整合模块）
# ============================================================
with tab6:
    t2d_main()
# TAB 7：肠道微生物组分析（新整合模块）
# ============================================================
with tab7:
    st.header("🦠 肠道微生物组分析")
    st.caption("基于 QIIME2 分类丰度表进行多样性、差异分析和 T2D 文献证据匹配")
    # ==================== 内部导航（水平 radio） ====================
    micro_page = st.radio(
        "功能选择",
        ["📤 数据载入", "📊 菌群概览", "🔬 差异分析", "📖 证据解释"],
        horizontal=True,
        key="micro_page"
    )
    # ==================== 数据管理（session_state） ====================
    if "micro_dataset" not in st.session_state:
        st.session_state.micro_dataset = None
    if "micro_demo_loaded" not in st.session_state:
        st.session_state.micro_demo_loaded = False
    # ---------- 辅助函数：加载 Demo 数据 ----------
    def load_demo_dataset():
        # 假设存在预处理好的 Qin 2012 数据（需用户运行 prepare_qin2012.py 生成）
        abundance_path = Path("data/microbiome_processed/qin_abundance.parquet")
        metadata_path = Path("data/microbiome_processed/qin_metadata.parquet")
        if not abundance_path.exists() or not metadata_path.exists():
            st.error("Demo 数据未找到，请先运行 `scripts/prepare_qin2012.py` 生成。")
            return None
        try:
            abundance = pd.read_parquet(abundance_path)
            metadata = pd.read_parquet(metadata_path)
            dataset = MicrobiomeDataset(
                abundance=abundance,
                metadata=metadata.set_index("SampleID"),
                group_column="Group",
                source_name="Qin2012_Demo",
                abundance_scale="relative"
            )
            return dataset
        except Exception as e:
            st.error(f"加载 Demo 数据失败：{e}")
            return None
    # ==================== 页面：数据载入 ====================
    if micro_page == "📤 数据载入":
        st.subheader("📤 数据载入与质控")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**上传数据文件**")
            abundance_file = st.file_uploader("丰度表 (TSV/CSV)", type=["tsv", "csv", "txt"], key="abund")
            metadata_file = st.file_uploader("样本信息表 (TSV/CSV)", type=["tsv", "csv", "txt"], key="meta")
            group_col = st.text_input("分组列名", value="Group")
        with col2:
            st.markdown("**或使用 Demo 数据**")
            if st.button("📥 载入 Qin 2012 Demo", use_container_width=True):
                dataset = load_demo_dataset()
                if dataset:
                    st.session_state.micro_dataset = dataset
                    st.session_state.micro_demo_loaded = True
                    st.success("Demo 数据载入成功！")
        if st.session_state.micro_dataset is not None:
            st.info(f"当前数据集：**{st.session_state.micro_dataset.source_name}**，样本数：{len(st.session_state.micro_dataset.abundance)}")
            if st.button("清除当前数据"):
                st.session_state.micro_dataset = None
                st.session_state.micro_demo_loaded = False
                st.rerun()
        if abundance_file is not None and metadata_file is not None:
            if st.button("🚀 处理并验证数据", type="primary"):
                try:
                    # 读取文件
                    abund_df = read_table(abundance_file)
                    meta_df = read_table(metadata_file)
                    # 标准化方向
                    abund_df = standardize_to_sample_x_taxon(abund_df)
                    # 简单检测分隔符等已在 read_table 内部处理
                    # 验证
                    abund_report = validate_abundance(abund_df)
                    meta_report = validate_metadata(meta_df, abund_df.index.tolist(), group_col)
                    if not abund_report.is_valid:
                        st.error(f"丰度表错误：{abund_report.summary}")
                    if not meta_report.is_valid:
                        st.error(f"样本信息错误：{meta_report.summary}")
                    if abund_report.is_valid and meta_report.is_valid:
                        dataset = MicrobiomeDataset(
                            abundance=abund_df,
                            metadata=meta_df.set_index("SampleID"),
                            group_column=group_col,
                            source_name="UserUpload",
                            abundance_scale="unknown"
                        )
                        st.session_state.micro_dataset = dataset
                        st.success("数据载入成功并通过质控！")
                except Exception as e:
                    st.error(f"处理失败：{e}")
    # ==================== 页面：菌群概览 ====================
    elif micro_page == "📊 菌群概览":
        st.subheader("📊 菌群概览")
        if st.session_state.micro_dataset is None:
            st.warning("请先载入数据")
        else:
            dataset = st.session_state.micro_dataset
            abundance = dataset.abundance
            meta = dataset.metadata
            group = dataset.group_column
            # 相对丰度转换（若需要）
            rel_abund = to_relative_abundance(abundance)
            # 计算 Shannon 多样性
            shannon = calculate_shannon(rel_abund)
            meta_shannon = meta.copy()
            meta_shannon["Shannon"] = shannon.values
            # 展示多样性箱线图
            fig_shannon = px.box(meta_shannon, x=group, y="Shannon", color=group,
                                 title="Shannon 多样性指数", points="all")
            st.plotly_chart(fig_shannon, use_container_width=True)
            # 计算 prevalence 并过滤
            prev_df = calculate_prevalence(rel_abund, min_samples=0.1)
            # 选择 Top 30 菌属作图
            top30 = select_top_taxa(rel_abund, n=30)
            # 将 top30 转为长格式用于饼图展示（按组平均）
            top30_long = top30.T
            top30_long.index.name = "Taxon"
            top30_long = top30_long.reset_index()
            fig_pie = px.pie(top30_long, values=top30_long.columns[1], names="Taxon",
                             title="Top 30 菌属相对丰度（平均）")
            st.plotly_chart(fig_pie, use_container_width=True)
            # 显示样本总丰度
            total_abund = rel_abund.sum(axis=1)
            fig_total = px.box(x=meta[group], y=total_abund, color=meta[group],
                               title="样本总相对丰度")
            st.plotly_chart(fig_total, use_container_width=True)
    # ==================== 页面：差异分析 ====================
    elif micro_page == "🔬 差异分析":
        st.subheader("🔬 差异分析")
        if st.session_state.micro_dataset is None:
            st.warning("请先载入数据")
        else:
            dataset = st.session_state.micro_dataset
            abundance = to_relative_abundance(dataset.abundance)
            meta = dataset.metadata
            group = dataset.group_column
            # 选择分组
            groups = meta[group].unique().tolist()
            col1, col2 = st.columns(2)
            with col1:
                case_label = st.selectbox("Case 组", groups, index=1 if len(groups)>1 else 0)
            with col2:
                control_label = st.selectbox("Control 组", groups, index=0)
            fdr_cutoff = st.slider("FDR 阈值", 0.0, 0.5, 0.05, 0.01)
            effect_cutoff = st.slider("最小效应量 (abs log2FC)", 0.0, 2.0, 0.5, 0.1)
            if st.button("运行差异分析", type="primary"):
                with st.spinner("正在运行 Mann-Whitney U 检验..."):
                    results = run_mannwhitney_differential(
                        abundance, meta, group, case_label, control_label,
                        pseudocount=1e-5
                    )
                    # 过滤候选菌
                    candidates = filter_candidates(results, fdr_threshold=fdr_cutoff,
                                                   min_effect_size=effect_cutoff,
                                                   min_prevalence=0.1)
                st.success(f"分析完成，发现 {len(candidates)} 个差异菌属")
                # 火山图
                results["-log10(FDR)"] = -np.log10(results["fdr"].clip(lower=1e-300))
                fig_volcano = px.scatter(
                    results, x="log2fc", y="-log10(FDR)",
                    color="direction",
                    hover_name="taxon",
                    title="差异丰度火山图"
                )
                fig_volcano.add_hline(y=-np.log10(fdr_cutoff), line_dash="dash")
                st.plotly_chart(fig_volcano, use_container_width=True)
                # 候选菌表
                st.dataframe(candidates)
                # 下载按钮
                csv = candidates.to_csv(index=False).encode('utf-8')
                st.download_button("下载候选菌 CSV", csv, "differential_taxa.csv", "text/csv")
    # ==================== 页面：证据解释 ====================
    elif micro_page == "📖 证据解释":
        st.subheader("📖 T2D 文献证据匹配")
        if st.session_state.micro_dataset is None:
            st.warning("请先载入数据，并在差异分析中获取候选菌列表")
        else:
            # 加载证据库
            repo = load_repository()  # 从种子数据自动加载
            search_taxon = st.text_input("输入菌属名称 (如 Bacteroides)", value="Bacteroides")
            if search_taxon:
                results = repo.search(search_taxon)
                if results:
                    for r in results:
                        with st.expander(f"{r['canonical_taxon']} ({r['reported_direction']})"):
                            st.markdown(f"**研究**: {r['study']} ({r['year']})")
                            st.markdown(f"**PMID**: {r['pmid']} | **DOI**: {r['doi']}")
                            st.markdown(f"**限制**: {r['limitations']}")
                else:
                    st.info("未找到该菌属的证据记录")
                # 功能标签
                guilds = repo.get_functional_guilds(search_taxon)
                if guilds:
                    st.markdown("**功能菌群标签**")
                    st.dataframe(pd.DataFrame(guilds))