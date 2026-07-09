import streamlit as st
st.set_page_config(page_title="胰腺微环境评估", page_icon=":microscope:", layout="wide")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap, os, warnings, tempfile, pathlib
import liana
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

DATA_DIR = "D:/streamlit/data"
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
    "ISEC": ["KDR", "ESM1", "PLVAP", "COL13A1", "UNC5B", "LAMA4", "THBS1", "BMP4", "CXCR4", "ACE", "PASK", "F2RL3"],
    "ASEC": ["IGFBP5", "IGFBP3", "COL4A1", "SPARC", "COL1A2", "CLU"],
    "Stellate":  ["LUM", "COL1A1", "DCN"],
    "Mast":    ["TPSAB1", "KIT"],
    "Macrophage": ["CD68", "CD14"],
}

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

def process_sample(adata, mg=200, res=0.8, mt_pct=20):
    sc.pp.filter_cells(adata, min_genes=mg)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < mt_pct].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata.raw = adata.copy()
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.pca(adata_hvg, n_comps=20)
    sc.pp.neighbors(adata_hvg, n_pcs=15, n_neighbors=30)
    sc.tl.leiden(adata_hvg, resolution=res, key_added="leiden")
    adata.obs["leiden"] = adata_hvg.obs["leiden"].values
    sc.tl.umap(adata_hvg, min_dist=0.3, spread=1.0)
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    
    var_names = list(adata.raw.var_names)
    Xr = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else np.array(adata.raw.X)
    isec_found = [g for g in ISEC_MARKERS if g in var_names]
    adata = annotate_cell_types(adata)
    isec_s = np.mean(Xr[:, [var_names.index(g) for g in isec_found]], axis=1)
    adata.obs["isec_score"] = isec_s
    cluster_mean = adata.obs.groupby("leiden")["isec_score"].mean()
    thresh = cluster_mean.quantile(0.75)
    isec_clusters = set(cluster_mean[cluster_mean >= thresh].index)
    adata.obs["is_isec"] = adata.obs["leiden"].isin(isec_clusters)
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
    
    # Run LIANA SingleCellSignalR for cell-cell communication
    try:
        if "cell_type" in adata.obs.columns:
            liana.mt.singlecellsignalr(adata, groupby="cell_type", expr_prop=0.1, verbose=False, inplace=True, key_added="liana_res")
    except Exception:
        pass
    
    return adata, gene_expr, n_isec

st.sidebar.title("胰腺微环境评估")
st.sidebar.markdown("基于 Nat Commun 2025")
st.sidebar.markdown("---")

st.title("胰腺微环境评估")
st.markdown("上传健康与疾病单细胞样本进行对比分析")

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    healthy_file = st.file_uploader("健康样本", type=["h5","h5ad","csv"], key="h")
with c2:
    disease_file = st.file_uploader("疾病样本", type=["h5","h5ad","csv"], key="d")
with c3:
    mg = st.slider("最小基因数", 50, 1000, 200, 50)
    res = st.slider("Resolution", 0.1, 2.0, 0.8, 0.1)
    mt_pct = st.slider("最大线粒体比例", 5, 50, 20, 5)

if not healthy_file or not disease_file:
    st.info("请上传健康与疾病样本以开始分析")
    st.stop()

with st.spinner("Loading files..."):
    adata_h = load_file(healthy_file)
    adata_d = load_file(disease_file)
    common = sorted(set(adata_h.var_names) & set(adata_d.var_names))
    adata_h = adata_h[:, common]
    adata_d = adata_d[:, common]
    st.success(f"Healthy: {adata_h.n_obs:,} cells | Disease: {adata_d.n_obs:,} cells | {len(common):,} genes")

if st.button("开始分析", type="primary"):
    with st.spinner("Processing Healthy sample..."):
        adata_h, expr_h, n_h = process_sample(adata_h, mg, res, mt_pct)
    with st.spinner("Processing Disease sample..."):
        adata_d, expr_d, n_d = process_sample(adata_d, mg, res, mt_pct)
    
    tab1, tab2, tab3 = st.tabs(["UMAP", "样本概览", "微环境评估"])
    with tab1:
        st.subheader("UMAP")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx2, (adata2, name2) in enumerate([(adata_h, "Healthy"), (adata_d, "Disease")]):
            ax = axes[idx2]
            if "cell_type" in adata2.obs.columns:
                for ct in adata2.obs["cell_type"].value_counts().index.tolist():
                    mask = adata2.obs["cell_type"].values == ct
                    ax.scatter(adata2.obsm["X_umap"][mask,0], adata2.obsm["X_umap"][mask,1], c=PANCREAS_CELL_COLORS.get(ct, "#636e72"), label=ct, s=2, alpha=0.4)
            else:
                ax.scatter(adata2.obsm["X_umap"][:,0], adata2.obsm["X_umap"][:,1], c="#d3d3d3", s=2, alpha=0.4)
            ax.legend(markerscale=2, fontsize=7, loc="best")
            ax.set_title(name2, fontsize=14); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        st.pyplot(fig); plt.close()
    with tab2:
        st.subheader("样本概览")
        st.markdown("**质控指标**")
        q1,q2,q3,q4 = st.columns(4)
        with q1: st.metric("健康细胞数", f"{adata_h.n_obs:,}")
        with q2: st.metric("疾病细胞数", f"{adata_d.n_obs:,}")
        with q3: st.metric("健康基因数", f"{adata_h.n_vars:,}")
        with q4: st.metric("疾病基因数", f"{adata_d.n_vars:,}")
        st.markdown("---")
        st.markdown("**细胞类型组成**")
        ct_h = adata_h.obs["cell_type"].value_counts()
        ct_d = adata_d.obs["cell_type"].value_counts()
        ct_df = pd.DataFrame({"健康(%)": (ct_h/ct_h.sum()*100).round(2), "疾病(%)": (ct_d/ct_d.sum()*100).round(2)}).fillna(0)
        ct_df = ct_df.sort_values("健康(%)", ascending=False)
        colors_ct = [PANCREAS_CELL_COLORS.get(ct, "#636e72") for ct in ct_df.index]
        ca,cb,cc = st.columns([2,2,1])
        with ca:
            f1,a1 = plt.subplots(figsize=(5,5))
            a1.pie(ct_df["健康(%)"], labels=None, colors=colors_ct, autopct=lambda p: f"{p:.0f}%" if p >= 10 else "", startangle=90, textprops={"fontsize":8})
            a1.set_title("Healthy", fontsize=12); st.pyplot(f1); plt.close()
        with cb:
            f2,a2 = plt.subplots(figsize=(5,5))
            a2.pie(ct_df["疾病(%)"], labels=None, colors=colors_ct, autopct=lambda p: f"{p:.0f}%" if p >= 10 else "", startangle=90, textprops={"fontsize":8})
            a2.set_title("Disease", fontsize=12); st.pyplot(f2); plt.close()
        with cc:
            st.markdown("**图例**")
            for ct in ct_df.index:
                ct_color = PANCREAS_CELL_COLORS.get(ct, "#636e72")
                svg_dot = f"<span style='display:inline-block;width:12px;height:12px;background:{ct_color};border-radius:50%;margin-right:6px'></span>{ct}"
                st.markdown(svg_dot, unsafe_allow_html=True)
        st.dataframe(ct_df, use_container_width=True)
    with tab3:
        st.subheader("微环境评估")
        st.markdown("通过LIANA SingleCellSignalR分析健康与疾病样本中3条T2D相关通讯轴（CXCL12-CXCR4、VEGF-A-KDR、DLL4-NOTCH1）的活性变化，通讯轴来源为Nat Commun 2025 Fig.5-7。")
        def extract_axis(res_df, ligand, receptor):
            if res_df is None or len(res_df)==0: return None
            mask = res_df["ligand_complex"].apply(lambda x: ligand in x) & res_df["receptor_complex"].apply(lambda x: receptor in x)
            sub = res_df[mask]
            return sub if len(sub)>0 else None
        axes_def = [
            ("CXCL12-CXCR4", "Vascular Protection", "down", "LUM+基质细胞分泌CXCL12，ISEC通过CXCR4接收血管保护信号，维持胰岛微血管稳态和β细胞功能。"),
            ("VEGF-A-KDR", "Angiogenesis", "up", "β细胞和基质细胞分泌VEGF-A，与ISEC表面KDR结合，引导窗孔形成、维持血管通透性和营养交换。"),
            ("DLL4-NOTCH1", "EC Maturation", "up", "ISEC内部的DLL4-NOTCH1信号调控内皮细胞成熟、血管新生过程中的尖端细胞-柄细胞分化。"),
        ]
        res_h = adata_h.uns.get("liana_res", None)
        res_d = adata_d.uns.get("liana_res", None)
        bio_meanings = {
            "CXCL12-CXCR4": {
                "up": "——疾病中通讯上调，在T2D中不典型，可能反映代偿性反应。",
                "down": "——疾病中通讯丢失或显著减弱，提示ISEC失去关键的血管保护信号，导致胰岛微循环障碍、β细胞营养供应不足和功能受损，是T2D微环境破坏的核心事件。",
                "normal": "——无明显变化，血管保护信号完整，ISEC微环境功能正常。",
            },
            "VEGF-A-KDR": {
                "up": "——疾病中通讯上调，提示异常血管生成信号激活，导致血管窗孔结构紊乱和通透性异常。",
                "down": "——该通路下调在T2D中不典型，可能导致ISEC窗孔形成不足和微血管功能减退。",
                "normal": "——无明显变化，VEGF-A信号调控正常，血管生成功能平衡。",
            },
            "DLL4-NOTCH1": {
                "up": "——疾病中通讯上调，提示内皮成熟程序紊乱和血管新生异常。",
                "down": "——该通路下调在T2D中不典型，可能影响内皮细胞分化和血管成熟。",
                "normal": "——无明显变化，内皮成熟信号调控正常。",
            },
        }
        for name, label, direction, desc in axes_def:
            parts = name.split("-")
            sub_h = extract_axis(res_h, parts[0], parts[1])
            sub_d = extract_axis(res_d, parts[0], parts[1])
            s_h = sub_h["lrscore"].mean() if sub_h is not None else 0
            s_d = sub_d["lrscore"].mean() if sub_d is not None else 0
            if s_h > 0 and s_d > 0:
                ratio = s_d / s_h; change = f"{ratio-1:+.0%}"
            elif s_h > 0 and s_d == 0:
                change = "lost"; ratio = 0
            elif s_h == 0 and s_d > 0:
                change = "new"; ratio = 999
            else:
                change = "N/A"; ratio = 1
            is_t2d = False
            if direction == "down":
                if (s_h > 0 and s_d == 0) or (s_h > 0 and s_d > 0 and ratio < 0.8): is_t2d = True
            else:
                if (s_h == 0 and s_d > 0) or (s_h > 0 and s_d > 0 and ratio > 1.2): is_t2d = True
            bm = bio_meanings.get(name, {})
            if s_h > 0 and s_d > 0:
                health_str = "健康通讯强度（均值" + f"{s_h:.2f}" + "），疾病通讯强度（均值" + f"{s_d:.2f}" + "），相对变化" + f"{ratio-1:.0%}" + "。"
                base = health_str
                if is_t2d:
                    tag = bm.get("up", "") if direction == "up" else bm.get("down", "")
                else:
                    tag = bm.get("normal", "")
                analysis = base + tag
            elif s_h > 0 and s_d == 0:
                analysis = "健康通讯强度（均值" + f"{s_h:.2f}" + "），疾病中通讯完全丢失。" + bm.get("down", "")
            elif s_h == 0 and s_d > 0:
                analysis = "健康中未检测到通讯，疾病中出现新通讯（强度" + f"{s_d:.2f}" + "）。" + bm.get("up", "")
            else:
                analysis = "健康与疾病样本中均未检测到该通讯通路活性。"
            col1_txt = f"<p style='font-size:20px;font-weight:bold;margin:2px 0;'>{s_h:.2f} \u2192 {s_d:.2f}</p>"
            change_color = "#c0392b" if change=="lost" or (change.startswith("+") and change!="+0%") else "#27ae60"
            html_box = (
                "<div style='border:2px solid #888;border-radius:8px;padding:15px;margin-bottom:18px;background:#fafafa;'>"
                + f"<p style='font-size:16px;font-weight:bold;margin:0 0 4px 0;'>{name} ({label})</p>"
                + f"<p style='font-size:13px;color:#666;margin:0 0 10px 0;'>{desc}</p>"
                + "<table style='width:100%;border:none;'><tr>"
                + "<td style='width:35%;vertical-align:top;border:none;padding-right:10px;'>"
                + f"<p style='font-size:13px;color:#888;margin:0;'>通讯强度</p>"
                + col1_txt
                + f"<p style='font-size:14px;color:{change_color};margin:0;'>{change}</p>"
                + "</td>"
                + "<td style='width:65%;vertical-align:top;border:none;padding-left:10px;border-left:1px solid #ddd;'>"
                + f"<p style='font-size:14px;margin:0;line-height:1.6;'>{analysis}</p>"
                + "</td></tr></table></div>"
            )
            st.markdown(html_box, unsafe_allow_html=True)
        
