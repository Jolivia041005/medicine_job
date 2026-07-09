# T2D 微生物文献证据 seed 数据说明

## 文件

- `taxon_evidence.csv`：29 条 T2D 分类单元文献证据。
- `functional_guilds.csv`：16 条潜在功能菌群标签。
- 编码：UTF-8。
- 两个文件通过 `canonical_taxon` 关联。

## 重要格式说明

用户给出的说明写“15 个字段”，但实际列举了 16 个字段（包含 `limitations`）。
本文件保留了全部 16 个明确要求的字段：

```text
evidence_id,canonical_taxon,taxonomic_level,aliases,reported_direction,study,year,population,sequencing_type,metformin_adjusted,evidence_type,evidence_summary,pmid,doi,url,limitations
```

## 证据设计

本 seed 数据刻意保留了：

1. **跨研究重复方向**：如 Faecalibacterium、Roseburia、Akkermansia 多次报告降低。
2. **冲突方向**：如 Blautia、Ruminococcus 在不同队列中方向不一致。
3. **药物混杂证据**：Forslund et al. 的 Escherichia 和 Intestinibacter 记录明确标注为二甲双胍相关方向。
4. **新诊断未用药队列**：Li et al. 的北方中国队列用于减少二甲双胍混杂。
5. **属/种分辨率差异**：种水平别名包含空格、下划线及 `s__` 形式。

## 使用建议

### 1. 不要把多条记录压缩成单一“真值”

同一 taxon 可能出现相反方向。页面建议显示：

- 支持 increased 的研究数；
- 支持 decreased 的研究数；
- 是否存在冲突；
- 是否校正二甲双胍；
- 当前用户队列与各研究是否一致。

### 2. Forslund 记录需要特殊标识

`Escherichia` 和 `Intestinibacter` 两条记录的方向主要反映二甲双胍效应。
建议页面添加 badge：

```text
药物相关证据，不等同于疾病本身方向
```

### 3. 功能标签不是功能预测

`functional_guilds.csv` 只用于展示“潜在功能菌群标签”：

- 不能称为 PICRUSt2 结果；
- 不能称为真实通路丰度；
- 不能由相对丰度直接推算代谢物产量；
- 属水平标签应优先展示 `strain_specific` 和 `limitations`。

### 4. 推荐的网站聚合逻辑

对每个差异 taxon 分别计算：

```text
same_direction_count
opposite_direction_count
metformin_adjusted_count
unique_population_count
systematic_review_count
```

不要自动产生临床风险评分。

## 主要文献

- Qin et al., Nature 2012. PMID 23023125.
- Gurung et al., EBioMedicine 2020. PMID 31901868.
- Forslund et al., Nature 2015. PMID 26633628.
- Li et al., Scientific Reports 2020. PMID 32214153.
- Fassatoui et al., Bioscience Reports 2019. PMID 31147456.
- Navab-Moghadam et al., Microbial Pathogenesis 2017. PMID 28739439.
- Gaike et al., mSystems 2020. PMID 32234773.
- Chong et al., Frontiers in Endocrinology 2025. PMID 39897957.
- Kinoshita et al., Journal of Diabetes Investigation 2025. PMID 40331921.

## 校验结果

- taxon evidence rows: 29
- functional guild rows: 16
- unique evidence IDs: yes
- allowed direction values only: yes
- allowed guild labels only: yes
- guild taxa all linked to evidence taxa: yes
