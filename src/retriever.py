import duckdb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HybridRetriever:
    def __init__(self, data_path="data/patient_data.parquet"):
        # 加载数据
        self.df = pd.read_parquet(data_path)
        
        # 创建 DuckDB 内存连接
        self.conn = duckdb.connect(database=':memory:')
        # 将 DataFrame 注册为 DuckDB 的临时视图（可直接在 SQL 中引用）
        self.conn.register('df_view', self.df)
        # 创建 patients 表，数据来源于注册的视图
        self.conn.execute("CREATE TABLE patients AS SELECT * FROM df_view")
        
        # 构建 TF-IDF 向量器
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["description"])
    
    def search(self, query_text, sql_filter=None, top_k=10):
        # 将查询转换为向量
        query_vec = self.vectorizer.transform([query_text])
        # 计算相似度
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        # 获取 top_k 索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        # 获取对应的 patient_id
        hit_ids = self.df.iloc[top_indices]["patient_id"].tolist()
        if not hit_ids:
            return pd.DataFrame()
        id_tuple = tuple(hit_ids)
        sql = f"SELECT * FROM patients WHERE patient_id IN {id_tuple}"
        if sql_filter:
            sql = f"SELECT * FROM ({sql}) AS sub WHERE {sql_filter}"
        result = self.conn.execute(sql).fetchdf()
        return result