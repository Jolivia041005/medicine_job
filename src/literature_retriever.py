import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

class LiteratureRetriever:
    def __init__(self, corpus_path=None, corpus_df=None):
        if corpus_df is not None:
            self.df = corpus_df
        elif corpus_path is not None:
            self.df = pd.read_parquet(corpus_path) if corpus_path.endswith('.parquet') else pd.read_csv(corpus_path)
        else:
            self.df = self._build_default_corpus()
        
        # 自定义分词
        def tokenizer(text):
            return jieba.lcut(text)
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['snippet'])
        
        # 结论性关键词列表（可扩充）
        self.keyword_list = [
            '结论', '表明', '显示', '证实', '发现', '有效', '疗效', '机制', '原理',
            '提示', '建议', '推荐', '改善', '降低', '增加', '减少',
            'meta分析', '系统评价', '随机对照', 'RCT', '指南', '共识',
            '方法', '目的', '结果', '讨论', '分析', '总结'
        ]
    
    def _build_default_corpus(self):
        data = {
            'snippet': [
                "SGLT2抑制剂可降低2型糖尿病合并慢性肾病患者的肾脏疾病进展风险，推荐用于eGFR≥20 mL/min/1.73m²的患者。",
                "对于老年糖尿病患者（≥65岁），ADA指南建议将HbA1c目标放宽至7.5%-8.0%，以避免低血糖风险。",
                "CREDENCE试验显示卡格列净可降低糖尿病肾病患者终末期肾病风险30%（HR 0.70, 95%CI 0.59-0.82）。",
                "2024年ADA指南推荐对于合并心血管疾病的2型糖尿病患者，优先使用SGLT2抑制剂或GLP-1受体激动剂。",
                "糖尿病肾病的早期筛查应每年检测尿白蛋白/肌酐比值（UACR）和估算肾小球滤过率（eGFR）。",
                "SGLT2抑制剂在老年人群中的安全性良好，但需注意生殖器感染和容量不足风险。",
                "二甲双胍仍是2型糖尿病的一线用药，但eGFR<30时禁用。",
                "针对糖尿病合并肥胖，GLP-1受体激动剂在减重方面优于SGLT2抑制剂。",
                "ACCORD试验强化降糖组（HbA1c<6.0%）未降低主要心血管事件，反而增加死亡率。",
                "UKPDS长期随访显示早期强化血糖控制可带来长期心血管获益。",
                "2025年中国2型糖尿病防治指南推荐对于合并动脉粥样硬化性心血管病的患者，首选SGLT2抑制剂或GLP-1受体激动剂。",
                "糖尿病合并心力衰竭的患者，SGLT2抑制剂可降低住院风险达30%。"
            ],
            'source': [
                "2024 ADA指南，糖尿病肾病章节",
                "2024 ADA指南，老年糖尿病管理",
                "CREDENCE试验（2019）",
                "2024 ADA指南，心血管病风险管理",
                "KDIGO 2024糖尿病肾病临床实践指南",
                "2024 ADA指南，药物安全性",
                "2024 ADA指南，降糖药物选择",
                "2025 ADA指南，肥胖管理",
                "ACCORD试验（2008）",
                "UKPDS长期随访（2008）",
                "2025中国2型糖尿病防治指南",
                "2024 ADA指南，心力衰竭管理"
            ]
        }
        return pd.DataFrame(data)
    
    def search(self, query, top_k=5):
        # 基础相似度
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 计算关键词得分
        keyword_scores = []
        for snippet in self.df['snippet']:
            count = 0
            for kw in self.keyword_list:
                count += snippet.count(kw)
            # 归一化：使用 sqrt(长度) 避免长片段占优
            norm_count = count / (len(snippet)**0.5 + 1)
            keyword_scores.append(min(norm_count, 1.0))
        keyword_scores = np.array(keyword_scores)
        
        # 综合得分：0.6 相似度 + 0.4 关键词得分
        combined = 0.6 * similarities + 0.4 * keyword_scores
        
        # 按综合得分排序
        top_indices = np.argsort(combined)[::-1][:top_k]
        results = self.df.iloc[top_indices][['snippet', 'source']].copy()
        results['score'] = combined[top_indices]
        return results
    
    def append_corpus(self, new_df):
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, stop_words=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['snippet'])
        return self.df