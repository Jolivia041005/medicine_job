import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
gene_file = os.path.join(BASE_DIR, 'gene.csv')
drug_file = os.path.join(BASE_DIR, 'drug.csv')
disease_file = os.path.join(BASE_DIR, 'disease.csv')
rel_file = os.path.join(BASE_DIR, 'relationship.csv')

# ==================== 数据加载（无缓存） ====================
def load_data():
    for fname, label in [(gene_file, 'gene.csv'),
                         (drug_file, 'drug.csv'),
                         (disease_file, 'disease.csv'),
                         (rel_file, 'relationship.csv')]:
        if not os.path.exists(fname):
            st.error(f"❌ 文件缺失：{label} 未在 {BASE_DIR} 找到。")
            st.stop()

    gene_df = pd.read_csv(gene_file, encoding='utf-8-sig')
    drug_df = pd.read_csv(drug_file, encoding='utf-8-sig')
    disease_df = pd.read_csv(disease_file, encoding='utf-8-sig')
    rel_df = pd.read_csv(rel_file, encoding='utf-8-sig')

    # 列名检查
    if 'symbol' not in gene_df.columns:
        st.error(f"❌ gene.csv 缺少 'symbol' 列，现有列：{list(gene_df.columns)}")
        st.stop()
    if 'name' not in drug_df.columns:
        st.error(f"❌ drug.csv 缺少 'name' 列，现有列：{list(drug_df.columns)}")
        st.stop()
    if 'name' not in disease_df.columns:
        st.error(f"❌ disease.csv 缺少 'name' 列，现有列：{list(disease_df.columns)}")
        st.stop()
    for col in ['source', 'target', 'rel_type']:
        if col not in rel_df.columns:
            st.error(f"❌ relationship.csv 缺少 '{col}' 列，现有列：{list(rel_df.columns)}")
            st.stop()

    # 中文翻译检查
    suspicious = ['零食', '移民局', '生物标志物', '二甲双胍', '格列吡嗪', '恩格列净']
    for col in ['source', 'target', 'rel_type']:
        if col in rel_df.columns:
            for word in suspicious:
                if rel_df[col].astype(str).str.contains(word).any():
                    st.error(f"❌ relationship.csv 中检测到翻译中文词语（如'{word}'），"
                             "请用记事本重新编辑该文件，确保全部使用英文。")
                    st.stop()

    # 调试输出
    with st.sidebar:
        st.write("📋 gene.csv 前3行", gene_df.head(3))
        st.write("📋 drug.csv 前3行", drug_df.head(3))
        st.write("📋 relationship.csv 前3行", rel_df.head(3))
        st.write(f"✅ 基因 {len(gene_df)} 行，药物 {len(drug_df)} 行，"
                 f"疾病 {len(disease_df)} 行，关系 {len(rel_df)} 行")

    return gene_df, drug_df, disease_df, rel_df

gene_df, drug_df, disease_df, rel_df = load_data()

# ==================== 构建图 ====================
G = nx.MultiDiGraph()

for _, row in gene_df.iterrows():
    if pd.notna(row['symbol']):
        G.add_node(row['symbol'], type='Gene',
                   name=row.get('name', '') if pd.notna(row.get('name', '')) else '',
                   desc=row.get('desc', '') if pd.notna(row.get('desc', '')) else '')

for _, row in drug_df.iterrows():
    if pd.notna(row['name']):
        G.add_node(row['name'], type='Drug',
                   class_=row.get('class', '') if pd.notna(row.get('class', '')) else '')

for _, row in disease_df.iterrows():
    if pd.notna(row['name']):
        G.add_node(row['name'], type='Disease',
                   full=row.get('full', '') if pd.notna(row.get('full', '')) else '')

for _, row in rel_df.iterrows():
    src = row['source']
    tgt = row['target']
    rel = row['rel_type']
    if pd.isna(src) or pd.isna(tgt) or pd.isna(rel):
        continue
    for n in [src, tgt]:
        if not G.has_node(n):
            G.add_node(n, type='Gene')
    G.add_edge(src, tgt, relation=rel)

# ==================== Streamlit 主界面 ====================
st.title("🧬 T2D 可视化搜索引擎 (MVP)")
query = st.text_input("请输入基因/药物/疾病名称（例如：Metformin、INS、T2D）")

if query:
    candidates = [n for n in G.nodes if query.lower() in str(n).lower()]
    if candidates:
        entity = candidates[0]
        node_data = G.nodes[entity]
        st.subheader(f"{entity} ({node_data.get('type', '')})")
        if 'name' in node_data: st.write(f"全称：{node_data['name']}")
        if 'desc' in node_data: st.write(f"简介：{node_data['desc']}")
        if 'class_' in node_data: st.write(f"药物类别：{node_data['class_']}")
        if 'full' in node_data: st.write(f"疾病全称：{node_data['full']}")

        neighbors = list(set(list(G.neighbors(entity)) +
                             [n for n, _ in G.in_edges(entity)]))
        if neighbors:
            st.write("**关联实体**")
            rows = []
            # ✅ 安全遍历出边和入边
            for u, v, data in G.out_edges(entity, data=True):
                if v in neighbors:
                    rows.append({'源': u, '关系': data['relation'], '目标': v})
            for u, v, data in G.in_edges(entity, data=True):
                if u in neighbors:
                    rows.append({'源': u, '关系': data['relation'], '目标': v})
            if rows:
                st.table(pd.DataFrame(rows).drop_duplicates())

            # 网络图
            sub_nodes = [entity] + neighbors
            sub_g = G.subgraph(sub_nodes).copy()
            fig, ax = plt.subplots(figsize=(7, 5))
            pos = nx.circular_layout(sub_g) if len(sub_nodes) <= 10 else nx.spring_layout(sub_g)
            color_map = {'Gene': 'skyblue', 'Drug': 'lightgreen', 'Disease': 'salmon'}
            node_colors = [color_map.get(sub_g.nodes[n].get('type', ''), 'gray') for n in sub_g.nodes]
            nx.draw(sub_g, pos, with_labels=True, node_color=node_colors,
                    edge_color='gray', node_size=1200, font_size=9, ax=ax)
            patches = [plt.Line2D([0], [0], marker='o', color='w', label=t,
                                  markerfacecolor=c, markersize=10)
                       for t, c in color_map.items()]
            ax.legend(handles=patches, loc='upper left')
            st.pyplot(fig)
        else:
            st.warning("该实体暂无关联数据。")
    else:
        st.warning("未找到匹配实体，请尝试其他关键词。")