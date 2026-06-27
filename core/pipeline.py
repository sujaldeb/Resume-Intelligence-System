"""
core/pipeline.py

Deterministic pipeline functions for the Resume Intelligence Platform.
All functions are pure and stateless — they take inputs and return outputs
with no side effects and no Streamlit dependency.

Each function is independently callable from app.py so session state
can be updated incrementally as each step completes.

Function execution order:
    1. score_ats()
    2. run_semantic_matching()
    3. compute_skill_gap()
    4. compute_percentiles()

The orchestration helper run_pipeline() calls these in order and returns
a combined results dict. app.py may call it directly or call each function
individually for incremental session state updates.
"""

import re
from bisect import bisect_left

import numpy as np

from config.settings import (
    COMPLETENESS_THRESHOLD_FULL,
    COMPLETENESS_THRESHOLD_PARTIAL,
    EXPERIENCE_MAX_YEARS,
    GAP_SKILLS_IN_CONTEXT,
    JD_SNIPPET_LENGTH,
    SKILL_SCORE_FLOOR,
    SKILL_SCORE_MAX_RAW,
    SUPPORTED_DOMAINS,
)


# Experience band helper

def get_experience_band(years: int) -> str:
    """Returns the display label for a given years of experience value."""
    if years is None or years <= 0:
        return "Unknown"
    if years <= 3:
        return "Junior (1-3)"
    if years <= 6:
        return "Mid (4-6)"
    if years <= 10:
        return "Senior (7-10)"
    return "Expert (11-20)"


# ATS scoring

def score_education(highest_education: str, education_scores: dict) -> float:
    """
    Maps education tier to a score out of 20.
    Unknown education scores 7 — partial credit, not zero.
    Masters, MBA, and Postgraduate are treated equivalently at 20.
    """
    return float(education_scores.get(highest_education, 7))


def score_experience(years: int | None) -> float:
    """
    Log-scale experience score out of 25.
    Diminishing returns reflect that each additional year matters less
    at senior levels. 20 years maps to the maximum of 25.
    """
    if not years or years <= 0:
        return 0.0
    raw     = np.log1p(min(years, EXPERIENCE_MAX_YEARS))
    max_raw = np.log1p(EXPERIENCE_MAX_YEARS)
    return round((raw / max_raw) * 25, 2)


def score_skills(
    canonical_profile: list,
    skill_concentration_weights: dict,
) -> float:
    """
    Concentration-weighted skill coverage score out of 30.
    Domain-specific skills receive weight 1.0.
    Cross-domain generic tools receive lower weights.
    A floor prevents generic-only profiles from scoring near zero.
    """
    if not canonical_profile:
        return 0.0
    weighted_sum = sum(
        skill_concentration_weights.get(skill, 0.1)
        for skill in canonical_profile
    )
    effective = max(weighted_sum, SKILL_SCORE_FLOOR)
    return round(min((effective / SKILL_SCORE_MAX_RAW) * 30, 30), 2)


def score_flags(
    flags: dict,
    flag_weights: dict,
    flag_cols: list,
) -> float:
    """
    Variance-weighted binary flag score out of 15.
    Flags with zero within-domain variance contribute near-zero weight.
    Flags are extracted from resume free text in the parser.
    """
    if not flags:
        return 0.0
    weighted_sum = sum(
        flags.get(col, 0) * flag_weights.get(col, 0.0)
        for col in flag_cols
    )
    max_possible = sum(flag_weights.values())
    if max_possible == 0:
        return 0.0
    return round((weighted_sum / max_possible) * 15, 2)


def score_completeness(
    experience_summary: str,
    project_summary: str,
    key_achievements: str,
    soft_skills_raw: str,
) -> float:
    """
    Text field presence and length score out of 10.
    Each of the four fields contributes 2.5 points.
    Full credit requires more than COMPLETENESS_THRESHOLD_FULL characters.
    Half credit for content above COMPLETENESS_THRESHOLD_PARTIAL.
    Empty or missing sections score zero — no fallback substitution.
    """
    fields    = [experience_summary, project_summary, key_achievements, soft_skills_raw]
    score     = 0.0
    per_field = 10.0 / len(fields)

    for val in fields:
        length = len(str(val).strip()) if val else 0
        if length > COMPLETENESS_THRESHOLD_FULL:
            score += per_field
        elif length > COMPLETENESS_THRESHOLD_PARTIAL:
            score += per_field * 0.5

    return round(score, 2)


def score_ats(parsed: dict, scoring_artifacts: dict) -> dict:
    """
    Computes all five ATS score components and the composite total.

    Args:
        parsed:            Output dict from resume_parser.parse_resume().
        scoring_artifacts: Loaded from ats_scoring_artifacts.json via resources.

    Returns dict with keys:
        ats_score, score_education, score_experience, score_skills,
        score_flags, score_completeness, experience_band.
    """
    s_edu  = score_education(
        parsed.get("highest_education", "Unknown"),
        scoring_artifacts["education_scores"],
    )
    s_exp  = score_experience(parsed.get("years_experience"))
    s_skill = score_skills(
        parsed.get("canonical_skill_profile", []),
        scoring_artifacts["skill_concentration_weights"],
    )
    s_flag = score_flags(
        parsed.get("flags", {}),
        scoring_artifacts["flag_weights"],
        scoring_artifacts["flag_cols"],
    )
    s_comp = score_completeness(
        parsed.get("experience_summary", ""),
        parsed.get("project_summary", ""),
        parsed.get("key_achievements", ""),
        parsed.get("soft_skills_raw", ""),
    )

    total = round(s_edu + s_exp + s_skill + s_flag + s_comp, 2)

    return {
        "ats_score":          total,
        "score_education":    s_edu,
        "score_experience":   s_exp,
        "score_skills":       s_skill,
        "score_flags":        s_flag,
        "score_completeness": s_comp,
        "experience_band":    get_experience_band(parsed.get("years_experience")),
    }


# Semantic matching

def build_skill_sentence(full_skill_profile: list) -> str:
    """
    Constructs the natural language skill sentence used for embedding.
    Consistent with the format used during benchmark population embedding.
    """
    if not full_skill_profile:
        return "Skills include: not specified"
    return "Skills include: " + ", ".join(sorted(full_skill_profile))


def run_semantic_matching(
    full_skill_profile: list,
    confirmed_domain: str,
    embedding_artifacts: dict,
    model,
    job_desc_index,
) -> dict:
    """
    Embeds the candidate skill sentence and scores against the domain job pool.
    Returns raw similarity, rescaled display score, and top job metadata.

    Args:
        full_skill_profile:  Combined canonical + supplementary skills list.
        confirmed_domain:    User-confirmed domain string.
        embedding_artifacts: From runtime_embedding_artifacts.pkl via resources.
        model:               Loaded SentenceTransformer instance from resources.
        job_desc_index:      job_desc DataFrame indexed by job_id from resources.

    Returns dict with keys:
        display_score, top_similarity_raw, top_job_id, top_job_title,
        top_job_description_snippet, skill_sentence.
    """
    if confirmed_domain not in SUPPORTED_DOMAINS:
        raise ValueError(
            f"Domain '{confirmed_domain}' is not in SUPPORTED_DOMAINS. "
            f"Confirm domain before running semantic matching."
        )

    skill_sentence = build_skill_sentence(full_skill_profile)

    candidate_emb = model.encode(
        [skill_sentence],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    job_embeddings  = embedding_artifacts["job_embeddings"]
    domain_job_idx  = embedding_artifacts["domain_job_indices"].get(confirmed_domain, [])
    rescaling       = embedding_artifacts["similarity_rescaling"]

    if not domain_job_idx:
        raise RuntimeError(
            f"No job embeddings found for domain '{confirmed_domain}'. "
            f"Check runtime_embedding_artifacts.pkl."
        )

    domain_job_embs = job_embeddings[domain_job_idx]
    sim_scores      = (candidate_emb @ domain_job_embs.T)[0]
    top_pos         = int(sim_scores.argmax())
    top_similarity  = float(sim_scores.max())

    pop_min = rescaling["pop_min"]
    pop_max = rescaling["pop_max"]
    display_score = round(float(np.clip(
        (top_similarity - pop_min) / (pop_max - pop_min) * 100,
        0, 100,
    )), 2)

    # resolving top job metadata
    # idx_to_job_id is inverted from job_id_to_idx here rather than read from a
    # precomputed key, because job_id_to_idx is the only one of the two that is
    # actually present on embedding_artifacts (the raw runtime_embedding_artifacts.pkl
    # content). A precomputed idx_to_job_id lives on the top-level resources dict in
    # resources.py, a sibling of embedding_artifacts, so reading it off
    # embedding_artifacts here always returned {} and silently broke this lookup.
    top_job_global_idx = domain_job_idx[top_pos]
    idx_to_job_id = {
        idx: job_id
        for job_id, idx in embedding_artifacts["job_id_to_idx"].items()
    }
    top_job_id         = idx_to_job_id.get(top_job_global_idx)

    top_job_title   = "Unknown"
    top_job_snippet = ""

    if top_job_id is not None:
        try:
            job_row         = job_desc_index.loc[int(top_job_id)]
            top_job_title   = str(job_row["title"])
            top_job_snippet = str(job_row["description"])[:JD_SNIPPET_LENGTH]
        except KeyError:
            pass

    return {
        "display_score":                  display_score,
        "top_similarity_raw":             round(top_similarity, 6),
        "top_job_id":                     top_job_id,
        "top_job_title":                  top_job_title,
        "top_job_description_snippet":    top_job_snippet,
        "skill_sentence":                 skill_sentence,
    }


# Skill gap analysis

def compute_skill_gap(
    canonical_skill_profile: list,
    confirmed_domain: str,
    embedding_artifacts: dict,
) -> dict:
    """
    Computes missing skills as set difference against the filtered domain
    frequency table. Missing skills are ranked by domain posting frequency.

    Uses canonical skills only — consistent with how the benchmark population
    gap was computed in Notebook 06.

    Args:
        canonical_skill_profile: ESCO-normalized canonical tokens from parser.
        confirmed_domain:        User-confirmed domain string.
        embedding_artifacts:     From runtime_embedding_artifacts.pkl via resources.

    Returns dict with keys:
        missing_skills, top_missing_skills, gap_count, domain_coverage_pct.
    """
    domain_freq     = embedding_artifacts["domain_skill_freq_filtered"].get(
        confirmed_domain, {}
    )
    suppress_tokens = set(embedding_artifacts.get("gap_suppress_tokens", []))
    candidate_set   = set(canonical_skill_profile)

    missing = {
        skill: count
        for skill, count in domain_freq.items()
        if skill not in candidate_set
        and skill not in suppress_tokens
    }

    ranked_missing = sorted(missing.items(), key=lambda x: x[1], reverse=True)
    missing_skills = [skill for skill, _ in ranked_missing]

    # coverage computed against the filtered pool excluding suppressed tokens
    filtered_pool = {
        skill for skill in domain_freq
        if skill not in suppress_tokens
    }
    matched      = len(candidate_set & filtered_pool)
    pool_size    = len(filtered_pool)
    coverage_pct = round(matched / pool_size * 100, 2) if pool_size > 0 else 0.0

    return {
        "missing_skills":      missing_skills,
        "top_missing_skills":  missing_skills[:GAP_SKILLS_IN_CONTEXT],
        "gap_count":           len(missing_skills),
        "domain_coverage_pct": coverage_pct,
    }


# Percentile computation

def compute_percentile(
    score: float,
    domain: str,
    metric: str,
    benchmark_lookup: dict,
) -> tuple[float | None, bool]:
    """
    Bisect-based percentile against pre-computed sorted arrays.
    No pandas or scipy dependency at runtime.

    Args:
        score:            The candidate's raw score for this metric.
        domain:           The confirmed domain string.
        metric:           One of 'ats_score', 'display_score', 'domain_coverage_pct'.
        benchmark_lookup: Loaded from runtime_benchmark_lookup.json via resources.

    Returns:
        (percentile, low_confidence_flag)
        percentile is None if the domain or metric is not found.
        low_confidence_flag is True for Engineering and Management.
    """
    domain_data    = benchmark_lookup.get(domain, {})
    sorted_values  = domain_data.get("score_arrays", {}).get(metric, [])
    low_confidence = domain_data.get("low_confidence", True)

    if not sorted_values:
        return None, low_confidence

    idx        = bisect_left(sorted_values, score)
    percentile = round(idx / len(sorted_values) * 100, 2)
    return percentile, low_confidence


def compute_percentiles(
    ats_score: float,
    display_score: float,
    domain_coverage_pct: float,
    confirmed_domain: str,
    benchmark_lookup: dict,
) -> dict:
    """
    Computes domain-stratified percentile ranks for all three scored metrics.

    Args:
        ats_score:           Composite ATS score from score_ats().
        display_score:       Rescaled semantic similarity from run_semantic_matching().
        domain_coverage_pct: Skill coverage percentage from compute_skill_gap().
        confirmed_domain:    User-confirmed domain string.
        benchmark_lookup:    Loaded from runtime_benchmark_lookup.json via resources.

    Returns dict with keys:
        ats_percentile, semantic_percentile, coverage_percentile, low_confidence_flag.
    """
    ats_pct,  low_conf = compute_percentile(
        ats_score, confirmed_domain, "ats_score", benchmark_lookup
    )
    sem_pct,  _        = compute_percentile(
        display_score, confirmed_domain, "display_score", benchmark_lookup
    )
    cov_pct,  _        = compute_percentile(
        domain_coverage_pct, confirmed_domain, "domain_coverage_pct", benchmark_lookup
    )

    return {
        "ats_percentile":      ats_pct,
        "semantic_percentile": sem_pct,
        "coverage_percentile": cov_pct,
        "low_confidence_flag": low_conf,
    }


# Orchestration helper

def run_pipeline(
    parsed: dict,
    confirmed_domain: str,
    resources: dict,
) -> dict:
    """
    Calls all four pipeline stages in sequence and returns a combined results dict.
    app.py may use this for simplicity or call individual functions for incremental
    session state updates.

    Args:
        parsed:           Output from resume_parser.parse_resume().
        confirmed_domain: User-confirmed domain string from session state.
        resources:        Loaded resources dict from load_resources().

    Returns flat dict containing all scoring, matching, gap, and percentile outputs.
    Raises ValueError if confirmed_domain is not in SUPPORTED_DOMAINS.
    """
    if confirmed_domain not in SUPPORTED_DOMAINS:
        raise ValueError(
            f"'{confirmed_domain}' is not a supported domain. "
            f"Supported: {sorted(SUPPORTED_DOMAINS)}"
        )

    ats_results = score_ats(parsed, resources["scoring"])

    semantic_results = run_semantic_matching(
        full_skill_profile  = parsed.get("full_skill_profile", []),
        confirmed_domain    = confirmed_domain,
        embedding_artifacts = resources["embedding"],
        model               = resources["model"],
        job_desc_index      = resources["job_desc_index"],
    )

    gap_results = compute_skill_gap(
        canonical_skill_profile = parsed.get("canonical_skill_profile", []),
        confirmed_domain        = confirmed_domain,
        embedding_artifacts     = resources["embedding"],
    )

    percentile_results = compute_percentiles(
        ats_score           = ats_results["ats_score"],
        display_score       = semantic_results["display_score"],
        domain_coverage_pct = gap_results["domain_coverage_pct"],
        confirmed_domain    = confirmed_domain,
        benchmark_lookup    = resources["benchmark"],
    )

    return {
        **ats_results,
        **semantic_results,
        **gap_results,
        **percentile_results,
        "confirmed_domain": confirmed_domain,
    }