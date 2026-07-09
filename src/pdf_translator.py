import re
import time
import pandas as pd
from openai import OpenAI
import tiktoken

class PDFTranslator:
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        """
        初始化翻译器
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") # DeepSeek 兼容

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        return len(self.encoding.encode(text))

    def _split_text_into_chunks(self, text: str, max_tokens: int = 500, overlap: int = 50) -> list:
        """
        将文本分割成有意义的片段，并保持上下文连贯。
        1. 按段落分割
        2. 段落过长则按句子分割，并尝试合并短句
        """
        if not text or not text.strip():
            return []

        # 1. 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 2. 如果段落本身适合，直接作为一个片段
            if self.count_tokens(para) <= max_tokens:
                candidate = para
            else:
                # 3. 段落过长，按句子分割
                sentences = re.split(r'(?<=[.?!])\s+', para)
                temp_chunk = ""
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if self.count_tokens(temp_chunk + " " + sent) <= max_tokens:
                        temp_chunk += " " + sent if temp_chunk else sent
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sent
                candidate = temp_chunk

            # 尝试将候选片段合并到当前片段，避免片段过短
            if current_chunk:
                combined = current_chunk + " " + candidate
                if self.count_tokens(combined) <= max_tokens:
                    current_chunk = combined
                    continue
                else:
                    chunks.append(current_chunk)
                    current_chunk = candidate
            else:
                current_chunk = candidate

        if current_chunk:
            chunks.append(current_chunk)

        # 过滤掉过短的片段（少于10个字符）
        return [c for c in chunks if len(c.strip()) > 10]

    def translate_chunk(self, chunk: str) -> str:
        """使用 DeepSeek API 翻译单个片段"""
        prompt = f"""你是一位专业的生物医学文献翻译专家。请将以下英文段落翻译成准确、流畅、专业的中文。
要求：
1. 严格遵循医学术语翻译规范
2. 保持原文的学术严谨性
3. 只返回翻译结果，不要添加任何额外说明

英文原文：
{chunk}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的生物医学文献翻译专家。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3  # 较低温度确保翻译准确
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"翻译出错: {e}")
            return f"[翻译失败: {e}]"

    def translate_pdf_text(self, full_text: str, source_name: str = "PDF") -> pd.DataFrame:
        """
        翻译整个 PDF 文本，返回包含原文、译文和来源的 DataFrame
        """
        print(f"开始处理文献: {source_name}")
        print(f"原始文本长度: {len(full_text)} 字符")

        # 1. 分割文本
        chunks = self._split_text_into_chunks(full_text)
        print(f"文本已分割为 {len(chunks)} 个片段")

        results = []
        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            print(f"正在翻译片段 {idx+1}/{total_chunks} (约 {self.count_tokens(chunk)} tokens)")
            translated = self.translate_chunk(chunk)
            results.append({
                'chunk_id': f"{source_name}_{idx+1:04d}",
                'original_text': chunk,
                'translated_text': translated,
                'source': f"PDF: {source_name}"
            })
            # 添加延时，避免 API 限流
            time.sleep(0.5)

        df = pd.DataFrame(results)
        print(f"翻译完成，共生成 {len(df)} 条双语片段")
        return df