def compute_direction_consistency(
    cohort_direction: str,
    evidence_direction: str,
) -> dict:
    if not cohort_direction or not evidence_direction:
        return {"consistent": None, "label": "无法判断"}

    cohort_is_t2d_up = "T2D >" in cohort_direction
    evidence_is_t2d_up = "T2D >" in evidence_direction or "increased" in evidence_direction

    cohort_is_control_up = "Control >" in cohort_direction
    evidence_is_control_up = "Control >" in evidence_direction or "decreased" in evidence_direction

    if cohort_is_t2d_up and evidence_is_t2d_up:
        return {"consistent": True, "label": "一致：本队列与文献均显示 T2D 组升高"}
    if cohort_is_control_up and evidence_is_control_up:
        return {"consistent": True, "label": "一致：本队列与文献均显示 T2D 组降低"}
    if (cohort_is_t2d_up and evidence_is_control_up) or (cohort_is_control_up and evidence_is_t2d_up):
        return {"consistent": False, "label": "不一致：本队列方向与既往文献相反"}
    return {"consistent": None, "label": "方向不明确"}
