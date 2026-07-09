import streamlit as st
import pandas as pd

from microbiome_src.evidence.repository import load_repository
from microbiome_src.evidence.scoring import compute_direction_consistency


def show():
    st.title("证据解释")

    if "dataset" not in st.session_state or st.session_state.dataset is None:
        st.warning("请先上传数据并完成质控，或前往差异分析页面。")
        st.caption(
            "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
        )
        return

    st.markdown(
        "本页面将当前队列的统计结果与已发表的 T2D 微生物组文献进行匹配。"
        "**统计显著性与文献证据应分别解读。**"
    )

    repo = load_repository()

    if "diff_results" not in st.session_state:
        st.info("请先前往 **差异分析** 页面完成分析，结果将自动显示于此。")
        st.markdown("---")
        st.markdown("### 手动搜索文献证据")
        manual_taxon = st.text_input("输入 taxon 名称（如 Bacteroides 或 g__Bacteroides）")
        if manual_taxon:
            _show_taxon_evidence(repo, manual_taxon, cohort_direction=None)

        st.caption(
            "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
            "文献证据来自已发表研究，不构成医学建议。"
        )
        return

    diff_results = st.session_state.diff_results
    candidates = diff_results[diff_results["fdr"] < 0.10].copy() if not diff_results.empty else diff_results

    if candidates.empty:
        st.info("当前阈值下无候选差异菌。请调整 FDR 阈值后重新分析。")
    else:
        st.markdown(f"### 候选差异菌证据（共 {len(candidates)} 个）")
        selected_evidence = st.selectbox(
            "选择 taxon 查看证据",
            options=[""] + candidates["taxon"].tolist(),
            format_func=lambda x: "— 请选择 —" if x == "" else x,
        )
        if selected_evidence:
            cohort_row = candidates[candidates["taxon"] == selected_evidence].iloc[0]
            cohort_direction = cohort_row["direction"]
            cohort_log2fc = cohort_row["log2fc"]
            cohort_fdr = cohort_row["fdr"]
            cohort_effect = cohort_row["effect_size"]

            _show_taxon_evidence(
                repo, selected_evidence,
                cohort_direction=cohort_direction,
                cohort_log2fc=cohort_log2fc,
                cohort_fdr=cohort_fdr,
                cohort_effect=cohort_effect,
            )

    st.caption(
        "⚠️ 免责声明：本工具为研究与工程支持工具，不是临床诊断系统。"
        "文献证据来自已发表研究，不构成医学建议。"
    )


def _show_taxon_evidence(repo, taxon_name, cohort_direction=None,
                         cohort_log2fc=None, cohort_fdr=None, cohort_effect=None):
    if cohort_direction:
        st.markdown("#### 本队列统计结果")
        cols = st.columns(4)
        cols[0].metric("log2FC", f"{cohort_log2fc:.3f}")
        cols[1].metric("FDR", f"{cohort_fdr:.4g}")
        cols[2].metric("效应量", f"{cohort_effect:.3f}")
        cols[3].metric("方向", cohort_direction)
        st.divider()

    evidence_list = repo.search(taxon_name)
    guild_list = repo.get_functional_guilds(taxon_name)

    st.markdown("#### 潜在功能菌群标签")
    if guild_list:
        for g in guild_list:
            conf_color = {"high": "green", "medium": "orange", "low": "gray"}
            color = conf_color.get(g["confidence"], "gray")
            st.markdown(
                f":{color}[**{g['functional_guild']}**] "
                f"(置信度: {g['confidence']})"
            )
            with st.expander("详细信息"):
                st.markdown(g["mechanism_summary"])
                if g.get("strain_specific"):
                    st.caption("⚠️ 该功能具有菌株特异性")
                if g.get("limitations"):
                    st.caption(f"限制: {g['limitations']}")
    else:
        st.info("当前本地证据库未收录该 taxon 的功能标签信息。")

    st.markdown("#### 既往文献证据")
    if evidence_list:
        for i, ev in enumerate(evidence_list):
            consistency = compute_direction_consistency(cohort_direction or "", ev["reported_direction"])
            with st.expander(
                f"{ev['canonical_taxon']} — {ev['study']} ({ev['year']}) "
                f"[匹配: {ev['match_confidence']}]"
            ):
                st.markdown(f"**方向**: {ev['reported_direction']}")
                st.markdown(f"**人群**: {ev['population']}")
                st.markdown(f"**测序**: {ev['sequencing_type']}")
                st.markdown(f"**二甲双胍校正**: {ev['metformin_adjusted']}")

                if cohort_direction:
                    st.markdown(f"**一致性**: {consistency['label']}")

                st.markdown(f"**摘要**: {ev['evidence_summary']}")
                st.markdown(f"**限制**: {ev.get('limitations', '未提供')}")

                links = []
                if ev.get("pmid"):
                    links.append(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{ev['pmid']}/)")
                if ev.get("doi"):
                    links.append(f"[DOI](https://doi.org/{ev['doi']})")
                if ev.get("url") and str(ev["url"]) != "nan":
                    links.append(f"[原文]({ev['url']})")
                if links:
                    st.markdown(" | ".join(links))
    else:
        st.info("当前本地证据库未收录该 taxon 的文献证据（不代表无相关研究）。")
        st.caption(
            "证据库为手工整理，仅包含部分已发表研究。"
            "未匹配不等同于无菌群-疾病关联证据。"
        )


show()
