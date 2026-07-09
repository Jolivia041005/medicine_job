import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # ----- 队列历史表 -----
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            sql_filter TEXT,
            top_k INTEGER,
            result_count INTEGER,
            duration REAL,
            timestamp TEXT,
            evidence_text TEXT,
            llm_output TEXT
        )
    ''')
    # 兼容旧表，添加新列
    c.execute("PRAGMA table_info(history)")
    columns = [col[1] for col in c.fetchall()]
    if 'evidence_text' not in columns:
        c.execute("ALTER TABLE history ADD COLUMN evidence_text TEXT")
    if 'llm_output' not in columns:
        c.execute("ALTER TABLE history ADD COLUMN llm_output TEXT")
    
    # ----- 文献历史表 -----
    c.execute('''
        CREATE TABLE IF NOT EXISTS lit_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            top_k INTEGER,
            result_count INTEGER,
            duration REAL,
            timestamp TEXT,
            top_results TEXT,
            llm_summary TEXT
        )
    ''')
    c.execute("PRAGMA table_info(lit_history)")
    columns = [col[1] for col in c.fetchall()]
    if 'top_results' not in columns:
        c.execute("ALTER TABLE lit_history ADD COLUMN top_results TEXT")
    if 'llm_summary' not in columns:
        c.execute("ALTER TABLE lit_history ADD COLUMN llm_summary TEXT")
    
    # ----- 新增：预测历史表 -----
    c.execute('''
        CREATE TABLE IF NOT EXISTS predict_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_features TEXT,   -- JSON字符串
            prediction INTEGER,
            probability REAL,
            top_features TEXT,       -- JSON
            timestamp TEXT
        )
    ''')
    # 无需 ALTER，因为是新建表
    
    conn.commit()
    conn.close()

# ---------- 队列历史 ----------
def add_history(query, sql_filter, top_k, result_count, duration, evidence_text="", llm_output=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO history (query, sql_filter, top_k, result_count, duration, timestamp, evidence_text, llm_output)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (query, sql_filter, top_k, result_count, duration, datetime.now().isoformat(), evidence_text, llm_output))
    conn.commit()
    # 只保留最近10条
    c.execute("DELETE FROM history WHERE id NOT IN (SELECT id FROM history ORDER BY timestamp DESC LIMIT 10)")
    conn.commit()
    conn.close()

def get_history(limit=10):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM history ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    return df

def clear_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()

# ---------- 文献历史 ----------
def add_lit_history(query, top_k, result_count, duration, top_results="", llm_summary=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO lit_history (query, top_k, result_count, duration, timestamp, top_results, llm_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (query, top_k, result_count, duration, datetime.now().isoformat(), top_results, llm_summary))
    conn.commit()
    c.execute("DELETE FROM lit_history WHERE id NOT IN (SELECT id FROM lit_history ORDER BY timestamp DESC LIMIT 10)")
    conn.commit()
    conn.close()

def get_lit_history(limit=10):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM lit_history ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    return df

def clear_lit_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM lit_history")
    conn.commit()
    conn.close()

# ---------- 新增：预测历史 ----------
def add_predict_history(patient_features, prediction, probability, top_features):
    """
    插入一条预测历史记录。
    :param patient_features: JSON字符串，患者特征
    :param prediction: 整数，预测结果（0/1或类别）
    :param probability: 浮点数，预测概率
    :param top_features: JSON字符串，影响预测的关键特征
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predict_history (patient_features, prediction, probability, top_features, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (patient_features, prediction, probability, top_features, datetime.now().isoformat()))
    conn.commit()
    # 只保留最近10条
    c.execute("DELETE FROM predict_history WHERE id NOT IN (SELECT id FROM predict_history ORDER BY timestamp DESC LIMIT 10)")
    conn.commit()
    conn.close()

def get_predict_history(limit=10):
    """获取最近的预测历史记录，返回DataFrame"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM predict_history ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    return df

def clear_predict_history():
    """清空预测历史表"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM predict_history")
    conn.commit()
    conn.close()