# app.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore", message=".*__path__.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import warnings
warnings.filterwarnings("ignore", message=".*__path__.*")
warnings.filterwarnings("ignore", category=FutureWarning)


import hashlib
import html
import re
import sys
from pathlib import Path

import streamlit as st



# adding project root to path so core/ and resume_parser/ import cleanly
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DOMAIN_DISPLAY_ORDER
from core.resources import load_resources
from core.pipeline import (
    run_pipeline,
    score_ats,
    run_semantic_matching,
    compute_skill_gap,
    compute_percentiles,
)
from core.ai_feedback import (
    is_ai_available,
    build_feedback_context,
    get_ai_feedback,
)
from resume_parser.resume_parser import parse_resume, get_parse_warnings

# page config — must be the first Streamlit call in the file
st.set_page_config(
    page_title="Resume Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# global CSS — reusable design-system classes, matching the approved wireframe.
# Step 11 will do a final pass once all stages are built; these are the
# foundational classes reused across stages 0-3.
# global CSS — loaded from static/styles.css, the single source of truth.
# Step 11 will do a final pass once all stages are built; this file holds the
# foundational classes reused across stages 0-3.
@st.cache_data
def load_css(css_path: str) -> str:
    with open(css_path, "r", encoding="utf-8") as f:
        return f.read()

_CSS_PATH = Path(__file__).parent / "static" / "styles.css"
st.markdown(f"<style>{load_css(str(_CSS_PATH))}</style>", unsafe_allow_html=True)

# resource loading — cached across all sessions, runs once per instance
try:
    resources = load_resources()
except RuntimeError as _startup_err:
    st.error("Application failed to start.")
    st.error(str(_startup_err))
    st.info(
        "Ensure all five runtime artifacts are present in the outputs/ directory "
        "and the sentence-transformer model is available."
    )
    st.stop()

# session state initialization — guard key prevents re-run on every re-render
if "initialized" not in st.session_state:

    st.session_state.stage = 0

    # stage 1 — parse outputs
    st.session_state.uploaded_filename         = None
    st.session_state.uploaded_file_hash        = None
    st.session_state.uploaded_file_bytes       = None
    st.session_state.raw_text                  = None
    st.session_state.sections                  = None
    st.session_state.parse_error               = None
    st.session_state.parse_warnings            = []
    st.session_state.years_experience          = None
    st.session_state.exp_confidence            = None
    st.session_state.highest_education         = None
    st.session_state.institution_tier          = None
    st.session_state.canonical_skill_profile   = []
    st.session_state.supplementary_skill_profile = []
    st.session_state.full_skill_profile        = []
    st.session_state.experience_summary        = None
    st.session_state.project_summary           = None
    st.session_state.key_achievements          = None
    st.session_state.soft_skills_raw           = None
    st.session_state.flags                     = {}
    st.session_state.detected_domain           = None
    st.session_state.domain_method             = None
    st.session_state.primary_role              = None

    # stage 2 — scoring and semantic outputs
    st.session_state.confirmed_domain          = None
    st.session_state.ats_score                 = None
    st.session_state.score_education           = None
    st.session_state.score_experience          = None
    st.session_state.score_skills              = None
    st.session_state.score_flags               = None
    st.session_state.score_completeness        = None
    st.session_state.experience_band           = None
    st.session_state.ats_percentile            = None
    st.session_state.display_score             = None
    st.session_state.semantic_percentile       = None
    st.session_state.top_job_id                = None
    st.session_state.top_job_title             = None
    st.session_state.top_job_description_snippet = None
    st.session_state.top_similarity_raw        = None
    st.session_state.missing_skills            = []
    st.session_state.top_missing_skills        = []
    st.session_state.gap_count                 = None
    st.session_state.domain_coverage_pct       = None
    st.session_state.coverage_percentile       = None
    st.session_state.low_confidence_flag       = None
    st.session_state.key_takeaways             = []

    # stage 3 — ai feedback
    st.session_state.ai_feedback               = None
    st.session_state.ai_error                  = None
    st.session_state.ai_feedback_requested     = False

    # guard — set last so partial initialization is never silently accepted
    st.session_state.initialized               = True

def _reset_all_state():
    """
    Resets every key below stage tracking to defaults.
    Called when a new file is uploaded (hash differs from stored hash).
    Stage is set to 0 after this call.
    """
    # stage 1
    st.session_state.uploaded_filename           = None
    st.session_state.uploaded_file_hash          = None
    st.session_state.uploaded_file_bytes         = None
    st.session_state.raw_text                    = None
    st.session_state.sections                    = None
    st.session_state.parse_error                 = None
    st.session_state.parse_warnings              = []
    st.session_state.years_experience            = None
    st.session_state.exp_confidence              = None
    st.session_state.highest_education           = None
    st.session_state.institution_tier            = None
    st.session_state.canonical_skill_profile     = []
    st.session_state.supplementary_skill_profile = []
    st.session_state.full_skill_profile          = []
    st.session_state.experience_summary          = None
    st.session_state.project_summary             = None
    st.session_state.key_achievements            = None
    st.session_state.soft_skills_raw             = None
    st.session_state.flags                       = {}
    st.session_state.detected_domain             = None
    st.session_state.domain_method               = None
    st.session_state.primary_role                = None

    # stage 2
    st.session_state.confirmed_domain            = None
    st.session_state.ats_score                   = None
    st.session_state.score_education             = None
    st.session_state.score_experience            = None
    st.session_state.score_skills                = None
    st.session_state.score_flags                 = None
    st.session_state.score_completeness          = None
    st.session_state.experience_band             = None
    st.session_state.ats_percentile              = None
    st.session_state.display_score               = None
    st.session_state.semantic_percentile         = None
    st.session_state.top_job_id                  = None
    st.session_state.top_job_title               = None
    st.session_state.top_job_description_snippet = None
    st.session_state.top_similarity_raw          = None
    st.session_state.missing_skills              = []
    st.session_state.top_missing_skills          = []
    st.session_state.gap_count                   = None
    st.session_state.domain_coverage_pct         = None
    st.session_state.coverage_percentile         = None
    st.session_state.low_confidence_flag         = None
    st.session_state.key_takeaways               = []

    # stage 3
    st.session_state.ai_feedback                 = None
    st.session_state.ai_error                    = None
    st.session_state.ai_feedback_requested       = False

    st.session_state.stage = 0


def _reset_stage_2_and_3():
    """
    Resets scoring and AI feedback keys without touching parse outputs.
    Called when the user changes the confirmed domain after scoring.
    """
    st.session_state.confirmed_domain            = None
    st.session_state.ats_score                   = None
    st.session_state.score_education             = None
    st.session_state.score_experience            = None
    st.session_state.score_skills                = None
    st.session_state.score_flags                 = None
    st.session_state.score_completeness          = None
    st.session_state.experience_band             = None
    st.session_state.ats_percentile              = None
    st.session_state.display_score               = None
    st.session_state.semantic_percentile         = None
    st.session_state.top_job_id                  = None
    st.session_state.top_job_title               = None
    st.session_state.top_job_description_snippet = None
    st.session_state.top_similarity_raw          = None
    st.session_state.missing_skills              = []
    st.session_state.top_missing_skills          = []
    st.session_state.gap_count                   = None
    st.session_state.domain_coverage_pct         = None
    st.session_state.coverage_percentile         = None
    st.session_state.low_confidence_flag         = None
    st.session_state.key_takeaways               = []
    st.session_state.ai_feedback                 = None
    st.session_state.ai_error                    = None
    st.session_state.ai_feedback_requested       = False
    st.session_state.stage                       = 1


def _handle_upload(uploaded_file):
    """
    Called on every re-run when st.file_uploader returns a file.
    Computes the file hash and decides whether to re-parse.

    If the hash matches the stored hash: no-op. Parse outputs are still valid.
    If the hash differs: reset all state, then parse.
    """
    

    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    file_hash  = hashlib.md5(file_bytes).hexdigest()

    if file_hash == st.session_state.uploaded_file_hash:
        return

    # new file — reset everything and parse
    _reset_all_state()

    st.session_state.uploaded_filename   = uploaded_file.name
    st.session_state.uploaded_file_hash  = file_hash
    st.session_state.uploaded_file_bytes = file_bytes

    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

    with st.spinner("Parsing resume..."):
        parsed = parse_resume(
            file_bytes           = file_bytes,
            file_extension       = ext,
            canonical_tokens     = resources["canonical_tokens"],
            normalization_lookup = resources["normalization_lookup"],
        )

    # extraction failed completely — no usable text
    if parsed.get("parse_error") and not parsed.get("raw_text"):
        st.session_state.parse_error = parsed["parse_error"]
        st.session_state.stage       = 0
        return

    # parse succeeded — populate all stage 1 keys
    st.session_state.parse_error               = parsed.get("parse_error")
    st.session_state.raw_text                  = parsed.get("raw_text")
    st.session_state.sections                  = parsed.get("sections")
    st.session_state.years_experience          = parsed.get("years_experience")
    st.session_state.exp_confidence            = parsed.get("exp_confidence")
    st.session_state.highest_education         = parsed.get("highest_education")
    st.session_state.institution_tier          = parsed.get("institution_tier")
    st.session_state.canonical_skill_profile   = parsed.get("canonical_skill_profile", [])
    st.session_state.supplementary_skill_profile = parsed.get("supplementary_skill_profile", [])
    st.session_state.full_skill_profile        = parsed.get("full_skill_profile", [])
    st.session_state.experience_summary        = parsed.get("experience_summary")
    st.session_state.project_summary           = parsed.get("project_summary")
    st.session_state.key_achievements          = parsed.get("key_achievements")
    st.session_state.soft_skills_raw           = parsed.get("soft_skills_raw")
    st.session_state.flags                     = parsed.get("flags", {})
    st.session_state.detected_domain           = parsed.get("detected_domain")
    st.session_state.domain_method             = parsed.get("domain_method")
    st.session_state.primary_role              = parsed.get("primary_role")
    st.session_state.parse_warnings            = get_parse_warnings(parsed)

    st.session_state.stage = 1


def _html(s: str) -> str:
    """
    Collapses a multi-line HTML template to a single line with no leading
    whitespace. Markdown's CommonMark spec treats lines indented 4+ spaces
    as a code block unless they are the very first line of a recognized
    raw-HTML block; once multiple pretty-indented f-strings are concatenated
    together (as in the Stage 2 card grids), later fragments fall on the
    wrong side of that rule and render as literal escaped text instead of
    HTML. Flattening to one line removes the ambiguity entirely. Pure
    formatting helper — no content change.
    """
    return " ".join(line.strip() for line in s.strip().splitlines())


def _render_progress_indicator(active_label: str):
    """
    Renders the four-step progress indicator (Upload, Review, Results, AI Feedback).
    Pure visual helper — reads no session state, writes nothing.
    Matches the flex/span markup used in the approved wireframe exactly.
    """
    steps = ["Upload", "Review", "Results", "AI Feedback"]
    active_idx = steps.index(active_label)
    parts = []
    for i, label in enumerate(steps):
        if i < active_idx:
            parts.append(f'<span class="progress-done">{label}</span>')
        elif i == active_idx:
            parts.append(f'<span class="progress-active">● {label}</span>')
        else:
            parts.append(f'<span class="progress-future">{label}</span>')
        if i < len(steps) - 1:
            parts.append('<div class="progress-line"></div>')
    st.markdown(
        f'<div style="display:flex;align-items:center;margin-bottom:18px;gap:4px">'
        f'{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def _on_domain_select_change():
    """
    Marks that the user has actively interacted with the domain selectbox.
    Used only to drive the Stage 1 State C visual (disabled confirm button
    until a domain has been explicitly picked when no domain was detected).
    Not part of the validated parse/pipeline session state architecture.
    """
    st.session_state.domain_user_selected = True


def generate_key_takeaways(pipeline_results: dict, confirmed_domain: str) -> list:
    """
    Deterministic, rule-based takeaways derived strictly from existing pipeline
    outputs. No new scoring, no recalculation of ATS/semantic/gap/percentile
    values — every value here is read directly from pipeline_results.
    Returns a list of plain strings for direct rendering in the Stage 2
    Key Takeaways panel. Presentation-layer logic only; core/pipeline.py
    is not touched.
    """
    takeaways = []

    ats_pct = pipeline_results.get("ats_percentile")
    sem_pct = pipeline_results.get("semantic_percentile")
    cov_pct = pipeline_results.get("coverage_percentile")

    # 1. ATS readiness vs semantic alignment divergence
    if ats_pct is not None and sem_pct is not None:
        diff = sem_pct - ats_pct
        if diff >= 15:
            takeaways.append(
                "Semantic alignment exceeds ATS readiness, suggesting strong "
                "role relevance but resume optimization opportunities."
            )
        elif diff <= -15:
            takeaways.append(
                "ATS readiness exceeds semantic alignment, suggesting a "
                "well-structured resume that may benefit from stronger "
                "domain-specific vocabulary."
            )

    # 2. percentile-vs-median summary across the three ranked signals
    signal_map = [
        (ats_pct, "ATS readiness"),
        (sem_pct, "semantic alignment"),
        (cov_pct, "skill coverage"),
    ]
    above = [label for pct, label in signal_map if pct is not None and pct >= 50]
    below = [label for pct, label in signal_map if pct is not None and pct < 50]

    def _join(labels):
        if len(labels) == 1:
            return labels[0]
        return ", ".join(labels[:-1]) + " and " + labels[-1]

    if above and below:
        takeaways.append(
            f"Candidate ranks above the domain median for {_join(above)}, "
            f"but below median for {_join(below)}."
        )
    elif above:
        takeaways.append(f"Candidate ranks above the domain median for {_join(above)}.")
    elif below:
        takeaways.append(f"Candidate ranks below the domain median for {_join(below)}.")

    # 3. top missing skill gap, with cloud-platform grouping as a special case
    top_missing = pipeline_results.get("top_missing_skills") or []
    gap_count   = pipeline_results.get("gap_count")
    if top_missing:
        cloud_tokens = {"aws", "azure", "gcp", "google cloud"}
        if any(s in cloud_tokens for s in top_missing[:3]):
            headline = "Cloud platform skills appear among the highest-priority gaps."
        else:
            shown  = top_missing[:2]
            plural = len(shown) > 1
            headline = (
                f"{' and '.join(shown)} {'appear' if plural else 'appears'} "
                f"among the highest-priority gaps."
            )
        if gap_count is not None:
            headline += f" ({gap_count} skills unmatched overall.)"
        takeaways.append(headline)

    # 4. experience band context
    experience_band = pipeline_results.get("experience_band")
    if experience_band:
        takeaways.append(f"Experience level is consistent with a {experience_band} profile.")

    # 5. low confidence benchmark caveat
    if pipeline_results.get("low_confidence_flag"):
        takeaways.append(
            f"The {confirmed_domain} benchmark population is below the reliability "
            f"threshold — percentile ranks should be treated as directional only."
        )

    return takeaways



# stage dispatch — the entire UI lives inside this block
# each stage renders independently based on session state

uploaded_file = st.file_uploader(
    label       = "Upload your resume",
    type        = ["pdf", "docx"],
    label_visibility = "collapsed",
    key         = "file_uploader",
)

# process the upload on every re-run where a file is present
if uploaded_file is not None:
    _handle_upload(uploaded_file)

# hard parse failure — file was uploaded but text extraction failed completely
# --- Stage 3: AI feedback panel helpers ---
# Matches the five-state design in stage2_refined_and_stage3_ai_feedback.html.
# "Loading" is handled by st.spinner at the point of the Gemini call (Stage 2 CTA),
# since this app's synchronous request model has no progressive state to render
# a skeleton against. The remaining four states (success, partial, unavailable,
# no key) are rendered here.

_AI_SECTION_UNAVAILABLE = "This section was not returned by the AI. Regenerate feedback to retry."

_AI_SECTION_LABELS = {
    "resume_review":       "01 \u00b7 Resume Review",
    "grammar_feedback":    "02 \u00b7 Grammar & Clarity",
    "achievement_framing": "03 \u00b7 Achievement Framing",
    "skill_gap_narrative": "04 \u00b7 Skill Gap Narrative",
    "action_plan":         "05 \u00b7 Action Plan",
}
_AI_FULL_WIDTH_KEYS = ["resume_review", "grammar_feedback"]
_AI_GRID_KEYS       = ["achievement_framing", "skill_gap_narrative", "action_plan"]


def _classify_ai_feedback(feedback):
    """
    Returns "unavailable", "partial", or "success".
    get_ai_feedback()'s failure paths fill every key with the same fallback
    message, so a single distinct value across all five keys means total
    failure rather than a genuine coincidental match. Any individual key still
    carrying the per-section _AI_SECTION_UNAVAILABLE marker means partial.
    """
    if not feedback:
        return "unavailable"
    values = list(feedback.values())
    if len(set(values)) == 1:
        return "unavailable"
    if any(v == _AI_SECTION_UNAVAILABLE for v in values):
        return "partial"
    return "success"


def _markdown_lite(text):
    """
    Converts the small, known subset of markdown Gemini produces into HTML.
    Not a general markdown parser — only handles what the feedback prompt in
    core/ai_feedback.py actually instructs the model to use: **bold** emphasis
    and paragraph/line breaks. Escapes first so user/model text can never break
    out of the surrounding HTML card.
    """
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = escaped.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return escaped


def _render_ai_section_card(key, feedback):
    label = _AI_SECTION_LABELS[key]
    value = feedback.get(key, _AI_SECTION_UNAVAILABLE)
    if value == _AI_SECTION_UNAVAILABLE:
        return (
            f'<div class="panel-ai-section-muted">'
            f'<div class="ai-section-label" style="color:#4b5563">{label}</div>'
            f'<div class="muted">This section could not be generated. '
            f'Your deterministic scores above remain accurate.</div>'
            f'</div>'
        )
    safe_value = _markdown_lite(value)
    return (
        f'<div class="panel-ai-section">'
        f'<div class="ai-section-label">{label}</div>'
        f'<div style="font-size:12px;color:#d1d5db;line-height:1.75">{safe_value}</div>'
        f'</div>'
    )


def _render_ai_panel_header(status, feedback):
    if status == "success":
        badge_text = "Gemini Flash 2.0 \u00b7 5 of 5 sections generated"
        pill = '<span class="pt" style="margin:0">\u2713 Complete</span>'
    elif status == "partial":
        n_ok = sum(1 for v in feedback.values() if v != _AI_SECTION_UNAVAILABLE)
        badge_text = f"Gemini Flash 2.0 \u00b7 {n_ok} of 5 sections generated"
        pill = '<span class="pa" style="margin:0">Partial response</span>'
    else:
        badge_text = "Gemini Flash 2.0"
        pill = '<span class="pa" style="margin:0">Unavailable</span>'

    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">'
        f'<div><div class="white" style="margin-bottom:3px;font-size:14px">AI Resume Feedback</div>'
        f'<div class="ai-badge">\u2726 {badge_text}</div></div>{pill}</div>'
    )


def _render_ai_feedback_panel(feedback):
    """Builds the success or partial state panel — sections 1-2 full width, 3-5 in a grid."""
    status = _classify_ai_feedback(feedback)
    header = _render_ai_panel_header(status, feedback)
    full_width_html = "".join(_render_ai_section_card(k, feedback) for k in _AI_FULL_WIDTH_KEYS)
    grid_cards = "".join(_render_ai_section_card(k, feedback) for k in _AI_GRID_KEYS)
    grid_html = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">{grid_cards}</div>'
    )
    return _html(f'<div class="panel-ai">{header}{full_width_html}{grid_html}</div>')


def _render_ai_unavailable_panel(feedback):
    """Builds the unavailable/error state panel — message plus retry guidance."""
    header = _render_ai_panel_header("unavailable", feedback)
    message = (
        _markdown_lite(next(iter(feedback.values())))
        if feedback
        else "AI feedback was not generated."
    )
    body = (
        f'<div style="background:rgba(28,18,8,0.6);border:0.5px solid rgba(217,119,6,0.35);'
        f'border-radius:8px;padding:12px">'
        f'<div style="font-size:12px;color:#d97706;font-weight:500;margin-bottom:4px">'
        f'\u26a0 AI feedback could not be generated</div>'
        f'<div class="muted" style="line-height:1.6">{message} '
        f'Your deterministic scores above are unaffected.</div></div>'
    )
    return _html(f'<div class="panel-ai">{header}{body}</div>')


if st.session_state.stage == 0 and st.session_state.parse_error:
    st.error(
        f"Resume could not be read: {st.session_state.parse_error}. "
        "Ensure the file is a text-based PDF or a standard DOCX document."
    )

# stage 0 — no file uploaded yet, or upload failed
if st.session_state.stage == 0:

    st.markdown(
        """
        <div class="hero-title">Resume Intelligence</div>
        <div class="hero-sub">
        NLP-based ATS readiness scoring, semantic role matching, and skill gap
        analysis for Data Science, IT, HR, Legal, Engineering, and Management
        candidates.
        </div>
        """,
        unsafe_allow_html=True,
    )

    _features = [
        ("ATS Readiness", "Five-component score covering education, experience, "
                           "skills, flags, and resume completeness.", "#60a5fa"),
        ("Semantic Match", "Embedding-based comparison against real job descriptions "
                            "in your domain.", "#afa9ec"),
        ("Skill Gap Analysis", "Ranked list of missing skills by demand frequency "
                                "in your domain's job pool.", "#67e8f9"),
        ("Benchmark Percentiles", "See where you rank against a reference population "
                                  "in your domain.", "#5dcaa5"),
    ]
    _feature_cards_html = "".join(
        f'<div class="feature-card" style="border-left:3px solid {_accent}">'
        f'<div class="feature-title" style="color:{_accent}">{_title}</div>'
        f'<div class="feature-desc">{_desc}</div></div>'
        for _title, _desc, _accent in _features
    )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">'
        f'{_feature_cards_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='muted'>Upload a PDF or DOCX resume above to begin.</div>",
        unsafe_allow_html=True,
    )

# stage 1 — parsed, domain pending confirmation
elif st.session_state.stage == 1:

    _render_progress_indicator("Review")

    # file bar — read-only summary; the uploader above remains the
    # single re-upload affordance (upload lifecycle is untouched)
    _ext = (
        st.session_state.uploaded_filename.rsplit(".", 1)[-1].upper()
        if st.session_state.uploaded_filename and "." in st.session_state.uploaded_filename
        else "FILE"
    )
    _char_count = len(st.session_state.raw_text) if st.session_state.raw_text else 0

    st.markdown(
        _html(f"""
        <div class="panel" style="display:flex;justify-content:space-between;
             align-items:center;margin-bottom:8px">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
            <span class="white">{st.session_state.uploaded_filename}</span>
            <span class="pg">{_ext}</span>
            <span class="muted">{_char_count:,} characters extracted</span>
          </div>
          <span class="pt">Parsed</span>
        </div>
        <div class="muted" style="margin-bottom:14px">
          Use the uploader above to replace this file.
        </div>
        """),
        unsafe_allow_html=True,
    )

    # parse warnings — high severity first, then medium/low
    _high_warnings = [w for w in st.session_state.parse_warnings if w["severity"] == "high"]
    _low_warnings  = [w for w in st.session_state.parse_warnings if w["severity"] in ("medium", "low")]

    for w in _high_warnings:
        st.markdown(
            f'<div class="warn-high">'
            f'<span style="font-size:11px;color:#d97706;font-weight:500">⚠ {w["message"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    for w in _low_warnings:
        st.markdown(f'<div class="annot">{w["message"]}</div>', unsafe_allow_html=True)

    # skills panel
    _canon = st.session_state.canonical_skill_profile
    _supp  = st.session_state.supplementary_skill_profile
    _supp_shown = _supp[:8]
    _supp_more  = len(_supp) - 8

    _canon_html = (
        "".join(f'<span class="pb">{s}</span>' for s in _canon)
        if _canon else '<span class="muted">No canonical skills detected.</span>'
    )
    _supp_html = "".join(f'<span class="pg">{s}</span>' for s in _supp_shown)
    if _supp_more > 0:
        _supp_html += f'<span class="pg">+{_supp_more} more</span>'

    st.markdown(
        _html(f"""
        <div class="panel" style="margin-bottom:14px">
          <div class="label-sec">Detected Skills</div>
          <div style="margin-bottom:8px">{_canon_html}</div>
          <div>{_supp_html}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # profile card — parser confidence is sections detected out of 4 expected
    _n_sections   = len(st.session_state.sections.keys()) if st.session_state.sections else 0
    _confidence   = min(_n_sections / 4, 1.0) * 100
    _years        = st.session_state.years_experience
    _years_display = f"{_years} yr" if _years else "Not detected"

    st.markdown(
        _html(f"""
        <div class="panel" style="margin-bottom:14px">
          <div class="label-sec">Profile Summary</div>
          <div class="field-row"><span class="muted">Detected role</span>
               <span class="white">{st.session_state.primary_role or "Not detected"}</span></div>
          <div class="field-row"><span class="muted">Experience</span>
               <span class="white">{_years_display}</span></div>
          <div class="field-row"><span class="muted">Education</span>
               <span class="white">{st.session_state.highest_education or "Unknown"}</span></div>
          <div class="field-row"><span class="muted">Sections detected</span>
               <span class="white">{_n_sections} of 4 expected</span></div>
          <div class="field-row"><span class="muted">Canonical skills</span>
               <span class="white">{len(_canon)}</span></div>
          <div class="field-row"><span class="muted">Supplementary skills</span>
               <span class="white">{len(_supp)}</span></div>
          <div class="field-row"><span class="muted">Domain detection method</span>
               <span class="white">{st.session_state.domain_method or "—"}</span></div>
          <div class="muted" style="margin-top:8px">Parser confidence</div>
          <div class="pct-bar-bg"><div style="width:{_confidence:.0f}%;height:100%;
               border-radius:4px;background:linear-gradient(90deg,#1d4ed8,#3b82f6)"></div></div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # domain confirmation card
    if "domain_user_selected" not in st.session_state:
        st.session_state.domain_user_selected = False

    detected = st.session_state.detected_domain
    method   = st.session_state.domain_method

    st.markdown(
        """
        <div class="panel" style="margin-bottom:8px">
          <div class="label-sec">Confirm Your Professional Domain</div>
          <div class="muted" style="margin-bottom:12px">
            Domain determines the job pool used for matching, the benchmark
            population for percentile ranking, and the skill gap analysis targets.
          </div>
        """,
        unsafe_allow_html=True,
    )

    default_idx = (
        DOMAIN_DISPLAY_ORDER.index(detected)
        if detected and detected in DOMAIN_DISPLAY_ORDER
        else 0
    )

    selected_domain = st.selectbox(
        label     = "Select domain",
        options   = DOMAIN_DISPLAY_ORDER,
        index     = default_idx,
        key       = "domain_selectbox",
        on_change = _on_domain_select_change,
    )

    # state A / B / C
    if detected is None:
        domain_resolved = st.session_state.domain_user_selected
        if not domain_resolved:
            st.markdown(
                '<div class="domain-hint-amber">⚠ No domain could be auto-detected. '
                'Select a domain to continue.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="domain-hint-blue">Manually selected: <strong>{selected_domain}</strong></div>',
                unsafe_allow_html=True,
            )
    else:
        domain_resolved = True
        if selected_domain == detected:
            st.markdown(
                f'<div class="domain-hint-green">Detected domain: <strong>{detected}</strong> '
                f'({"from job title" if method == "title" else "from skill profile"})</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="domain-hint-blue">Manually selected: <strong>{selected_domain}</strong> '
                f'(detected was {detected})</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    if not domain_resolved:
        st.markdown(
            '<div class="btn-disabled">Confirm domain and analyse</div>',
            unsafe_allow_html=True,
        )
    elif st.button("Confirm domain and analyse", type="primary"):
        # if domain changed after a previous scoring run, reset stages 2 and 3
        if (
            st.session_state.stage == 1
            and st.session_state.confirmed_domain is not None
            and st.session_state.confirmed_domain != selected_domain
        ):
            _reset_stage_2_and_3()

        st.session_state.confirmed_domain = selected_domain

        # run the full deterministic pipeline
        with st.spinner("Analysing resume..."):
            pipeline_results = run_pipeline(
                parsed           = {
                    "years_experience":        st.session_state.years_experience,
                    "highest_education":       st.session_state.highest_education,
                    "canonical_skill_profile": st.session_state.canonical_skill_profile,
                    "full_skill_profile":      st.session_state.full_skill_profile,
                    "flags":                   st.session_state.flags,
                    "experience_summary":      st.session_state.experience_summary,
                    "project_summary":         st.session_state.project_summary,
                    "key_achievements":        st.session_state.key_achievements,
                    "soft_skills_raw":         st.session_state.soft_skills_raw,
                },
                confirmed_domain = selected_domain,
                resources        = resources,
            )

        # populate stage 2 session state keys from pipeline results
        st.session_state.ats_score                   = pipeline_results["ats_score"]
        st.session_state.score_education             = pipeline_results["score_education"]
        st.session_state.score_experience            = pipeline_results["score_experience"]
        st.session_state.score_skills                = pipeline_results["score_skills"]
        st.session_state.score_flags                 = pipeline_results["score_flags"]
        st.session_state.score_completeness          = pipeline_results["score_completeness"]
        st.session_state.experience_band             = pipeline_results["experience_band"]
        st.session_state.ats_percentile              = pipeline_results["ats_percentile"]
        st.session_state.display_score               = pipeline_results["display_score"]
        st.session_state.semantic_percentile         = pipeline_results["semantic_percentile"]
        st.session_state.top_job_id                  = pipeline_results["top_job_id"]
        st.session_state.top_job_title               = pipeline_results["top_job_title"]
        st.session_state.top_job_description_snippet = pipeline_results["top_job_description_snippet"]
        st.session_state.top_similarity_raw          = pipeline_results["top_similarity_raw"]
        st.session_state.missing_skills              = pipeline_results["missing_skills"]
        st.session_state.top_missing_skills          = pipeline_results["top_missing_skills"]
        st.session_state.gap_count                   = pipeline_results["gap_count"]
        st.session_state.domain_coverage_pct         = pipeline_results["domain_coverage_pct"]
        st.session_state.coverage_percentile         = pipeline_results["coverage_percentile"]
        st.session_state.low_confidence_flag         = pipeline_results["low_confidence_flag"]

        # presentation-layer takeaways — derived from the pipeline results
        # already computed above; no recalculation, no new scoring
        st.session_state.key_takeaways = generate_key_takeaways(
            pipeline_results = pipeline_results,
            confirmed_domain = selected_domain,
        )

        st.session_state.stage = 2
        st.rerun()

# stage 2 — domain confirmed, scored
elif st.session_state.stage == 2:

    _render_progress_indicator("Results")

    domain = st.session_state.confirmed_domain
    _benchmark_info = resources["benchmark"].get(domain, {})
    _n_candidates   = _benchmark_info.get("n_candidates", "—")

    # dashboard heading bar — candidate context + benchmark badge
    st.markdown(
        _html(f"""
        <div class="panel" style="display:flex;justify-content:space-between;
             align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:10px">
          <div>
            <div class="white" style="font-size:17px;margin-bottom:6px">{domain} · Resume Analysis</div>
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
              <span class="muted">{st.session_state.uploaded_filename}</span>
              <span class="muted">·</span>
              <span class="pg">{st.session_state.years_experience or 0} yr experience</span>
              <span class="pp">{st.session_state.experience_band or "—"}</span>
              <span class="pg">{st.session_state.highest_education or "Unknown"}</span>
            </div>
          </div>
          <div style="text-align:center">
            <div class="label-sec" style="margin-bottom:2px">Synthetic Benchmark</div>
            <div class="white">{_n_candidates} Candidates</div>
            <div class="muted">{domain} pool</div>
          </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # ROW 1 — ATS hero / semantic / coverage
    _ats_score = st.session_state.ats_score
    _ats_pct   = st.session_state.ats_percentile
    _ats_pct_disp = f"{_ats_pct:.0f}" if _ats_pct is not None else "—"
    _sem_pct   = st.session_state.semantic_percentile
    _sem_pct_disp = f"{_sem_pct:.0f}" if _sem_pct is not None else "—"
    _disp      = st.session_state.display_score
    _cov_pct   = st.session_state.coverage_percentile
    _cov_pct_disp = f"{_cov_pct:.0f}" if _cov_pct is not None else "—"
    _coverage  = st.session_state.domain_coverage_pct

    _card_ats = _html(f"""
        <div class="panel-glow-blue">
          <div class="label-sec">ATS Readiness Score</div>
          <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:0;
               margin-bottom:10px;align-items:center">
            <div>
              <div class="muted" style="text-transform:uppercase;letter-spacing:.05em">Score</div>
              <div style="font-size:36px;font-weight:600;color:#60a5fa;line-height:1">{_ats_score}</div>
              <div class="muted">out of 100</div>
            </div>
            <div style="width:1px;background:rgba(55,65,81,0.6);height:50px;margin:0 14px"></div>
            <div>
              <div class="muted" style="text-transform:uppercase;letter-spacing:.05em">Percentile</div>
              <div style="font-size:36px;font-weight:600;color:#afa9ec;line-height:1">{_ats_pct_disp}<span style="font-size:18px;font-weight:400">th</span></div>
              <div class="muted">{domain} pool</div>
            </div>
          </div>
          <div class="pct-bar-bg"><div style="width:{_ats_score or 0}%;height:100%;border-radius:4px;
               background:linear-gradient(90deg,#1d4ed8,#3b82f6)"></div></div>
          <div class="muted" style="margin-top:6px">Experience band: {st.session_state.experience_band or "Not available"}.
          Score is absolute — see breakdown below for component detail.</div>
        </div>
    """)

    _card_sem = _html(f"""
        <div class="panel-glow-purple">
          <div class="label-sec">Job Alignment</div>
          <div class="muted" style="text-transform:uppercase;letter-spacing:.05em">Percentile</div>
          <div style="font-size:30px;font-weight:600;color:#afa9ec;line-height:1;margin-bottom:6px">
            {_sem_pct_disp}<span style="font-size:16px;font-weight:400">th</span></div>
          <div class="pct-bar-bg"><div style="width:{_sem_pct or 0}%;height:100%;border-radius:4px;
               background:linear-gradient(90deg,#534ab7,#7f77dd)"></div></div>
          <hr class="divider">
          <div class="muted">Display score</div>
          <div style="color:#afa9ec;font-weight:500">{_disp} / 100</div>
          <div class="muted" style="font-size:9px">within {domain} only — not cross-domain comparable</div>
        </div>
    """)

    _card_cov = _html(f"""
        <div class="panel-glow-teal">
          <div class="label-sec">Skill Coverage</div>
          <div class="muted" style="text-transform:uppercase;letter-spacing:.05em">Percentile</div>
          <div style="font-size:30px;font-weight:600;color:#5dcaa5;line-height:1;margin-bottom:6px">
            {_cov_pct_disp}<span style="font-size:16px;font-weight:400">th</span></div>
          <div class="pct-bar-bg"><div style="width:{_coverage or 0}%;height:100%;border-radius:4px;
               background:linear-gradient(90deg,#065f46,#1d9e75)"></div></div>
          <hr class="divider">
          <div class="muted">Domain coverage</div>
          <div style="color:#5dcaa5;font-weight:500">{_coverage}%</div>
        </div>
    """)

    st.markdown(
        f'<div style="display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:12px;margin-bottom:12px">'
        f'{_card_ats}{_card_sem}{_card_cov}</div>',
        unsafe_allow_html=True,
    )

    # ROW 2 — Key Takeaways (deterministic, from generate_key_takeaways())
    if st.session_state.key_takeaways:
        _items_html = "".join(
            f'<div class="takeaway-item"><div class="takeaway-dot">✦</div>'
            f'<div class="muted" style="font-size:12px">{t}</div></div>'
            for t in st.session_state.key_takeaways
        )
    else:
        _items_html = '<div class="muted">No notable takeaways for this profile.</div>'

    st.markdown(
        f"""
        <div class="panel" style="margin:12px 0;border-color:rgba(83,74,183,0.3)">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div class="label-sec" style="margin-bottom:0">Key Takeaways</div>
            <span class="pg" style="margin:0">Deterministic · No AI</span>
          </div>
          {_items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ROW 3 — ATS breakdown + Top Job Match
    _components = [
        ("Experience",          st.session_state.score_experience,   25),
        ("Skill Coverage",      st.session_state.score_skills,       30),
        ("Education",           st.session_state.score_education,    20),
        ("Experience Flags",    st.session_state.score_flags,        15),
        ("Resume Completeness", st.session_state.score_completeness, 10),
    ]
    _rows_html = ""
    for label, score, max_score in _components:
        score = score if score is not None else 0
        pct   = (score / max_score * 100) if max_score else 0
        _rows_html += (
            f'<div style="margin-bottom:9px">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
            f'<span class="muted">{label}</span>'
            f'<span style="font-size:11px;color:#60a5fa">{score} / {max_score}</span></div>'
            f'<div class="pct-bar-bg"><div style="width:{pct:.1f}%;height:100%;border-radius:4px;'
            f'background:#3b82f6"></div></div></div>'
        )

    _card_breakdown = _html(f"""
        <div class="panel">
          <div class="label-sec">ATS Score Breakdown</div>
          {_rows_html}
          <hr class="divider">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span class="muted">Total ATS Score</span>
            <span style="font-size:16px;color:#60a5fa;font-weight:600">{_ats_score} / 100</span>
          </div>
        </div>
    """)

    _snippet  = (st.session_state.top_job_description_snippet or "")[:240]
    _sim_raw  = st.session_state.top_similarity_raw
    _sim_disp = f"{_sim_raw:.3f}" if _sim_raw is not None else "—"
    _sem_disp = f"{_sem_pct:.0f}th" if _sem_pct is not None else "—"

    _card_job = _html(f"""
        <div class="panel-purple">
          <div class="label-sec">Best Matching Role</div>
          <div class="white" style="margin-bottom:8px">{st.session_state.top_job_title or "Not available"}</div>
          <div style="background:rgba(83,74,183,0.08);border:0.5px solid rgba(83,74,183,0.25);
               border-radius:8px;padding:10px 12px;margin-bottom:10px">
            <div class="muted" style="margin-bottom:4px">Job description excerpt</div>
            <div class="muted">{_snippet}{"..." if len(_snippet) == 240 else ""}</div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <div><div class="muted">Cosine similarity</div>
                 <div style="color:#afa9ec;font-weight:500">{_sim_disp}</div></div>
            <div style="text-align:right"><div class="muted">Semantic percentile</div>
                 <div style="color:#afa9ec;font-weight:500">{_sem_disp}</div></div>
          </div>
        </div>
    """)

    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">'
        f'{_card_breakdown}{_card_job}</div>',
        unsafe_allow_html=True,
    )

    # ROW 4 — Skill gap panel
    _top_missing = st.session_state.top_missing_skills or []
    _all_missing = st.session_state.missing_skills or []
    _gap_count   = st.session_state.gap_count

    if _top_missing:
        _high = _top_missing[:3]
        _rest = [s for s in _all_missing if s not in _high][:8]
        _high_html = "".join(f'<div class="gap-high"><div class="white">{s}</div></div>' for s in _high)
        _rest_html = "".join(f'<span class="pg">{s}</span>' for s in _rest)
        st.markdown(
            f"""
            <div class="panel" style="margin:12px 0">
              <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px">
                <div class="label-sec" style="margin-bottom:0">Top Skill Gaps</div>
                <div class="muted">{_gap_count if _gap_count is not None else 0} skills missing ·
                     ranked by demand in {domain} postings</div>
              </div>
              <div style="display:grid;grid-template-columns:repeat({len(_high)},1fr);gap:8px;
                   margin-bottom:{12 if _rest else 0}px">
                {_high_html}
              </div>
              <div>{_rest_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="panel" style="margin:12px 0">'
            '<div class="label-sec">Top Skill Gaps</div>'
            '<div class="muted">No skill gaps identified against the domain pool.</div></div>',
            unsafe_allow_html=True,
        )

    # ROW 5 — mandatory caveats per Memory V10
    _caveat_synthetic = (
        '<div class="annot">Percentile ranks are benchmarked against a synthetic '
        'population of 5,000 candidates and are directional only.</div>'
    )

    if st.session_state.low_confidence_flag:
        _caveat_low_conf = (
            f'<div class="warn-high"><span style="font-size:11px;color:#d97706;font-weight:500">'
            f'The {domain} domain has fewer than 200 benchmark candidates. '
            f'Percentile ranks carry reduced reliability.</span></div>'
        )
        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">'
            f'{_caveat_synthetic}{_caveat_low_conf}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(_caveat_synthetic, unsafe_allow_html=True)

    # --- AI Feedback CTA ---
    ai_available = is_ai_available(st.secrets)

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,rgba(83,74,183,0.15),rgba(59,130,246,0.1));
             border:0.5px solid rgba(83,74,183,0.4);border-radius:12px;padding:16px 20px;
             margin:16px 0 10px 0">
          <div class="white" style="margin-bottom:3px">Get AI-powered feedback on your resume</div>
          <div class="muted">Resume review · Grammar · Achievement framing · Skill gap narrative · Action plan</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ai_available:
        if st.session_state.ai_feedback is None:
            if st.button("Generate AI feedback", type="primary"):
                st.session_state.ai_feedback_requested = True
                with st.spinner("Generating feedback..."):
                    context  = build_feedback_context(
                        parsed = {
                            "full_skill_profile":  st.session_state.full_skill_profile,
                            "experience_summary":  st.session_state.experience_summary,
                            "project_summary":     st.session_state.project_summary,
                            "key_achievements":    st.session_state.key_achievements,
                        },
                        ats_results = {
                            "ats_score":          st.session_state.ats_score,
                            "score_education":    st.session_state.score_education,
                            "score_experience":   st.session_state.score_experience,
                            "score_skills":       st.session_state.score_skills,
                            "score_flags":        st.session_state.score_flags,
                            "score_completeness": st.session_state.score_completeness,
                            "experience_band":    st.session_state.experience_band,
                        },
                        semantic_results = {
                            "top_job_title":               st.session_state.top_job_title,
                            "top_job_description_snippet": st.session_state.top_job_description_snippet,
                        },
                        gap_results = {
                            "top_missing_skills":  st.session_state.top_missing_skills,
                            "gap_count":           st.session_state.gap_count,
                            "domain_coverage_pct": st.session_state.domain_coverage_pct,
                        },
                        percentile_results = {
                            "ats_percentile":      st.session_state.ats_percentile,
                            "semantic_percentile": st.session_state.semantic_percentile,
                            "coverage_percentile": st.session_state.coverage_percentile,
                            "low_confidence_flag": st.session_state.low_confidence_flag,
                        },
                        confirmed_domain = st.session_state.confirmed_domain,
                    )
                    st.session_state.ai_feedback = get_ai_feedback(context, st.secrets)
                st.session_state.stage = 3
                st.rerun()
        else:
            st.markdown(
                '<div class="annot">AI feedback has been generated. '
                'Use the button below to view it.</div>',
                unsafe_allow_html=True,
            )
            if st.button("View feedback", type="primary"):
                st.session_state.stage = 3
                st.rerun()
    else:
        st.markdown(
            '<div class="muted">AI feedback is not available. Add a GEMINI_API_KEY to '
            '.streamlit/secrets.toml to enable it.</div>',
            unsafe_allow_html=True,
        )

    # allow re-analysis with a different domain
    if st.button("Change domain or re-analyse"):
        _reset_stage_2_and_3()
        st.rerun()


# stage 3 — AI feedback generated
elif st.session_state.stage == 3:

    st.markdown(f"### {st.session_state.uploaded_filename}")
    st.caption(f"Domain: {st.session_state.confirmed_domain}  |  {st.session_state.experience_band}")

    # compact score recap, styled consistently with Stage 2 — not bare st.metric
    _recap_html = _html(f"""
        <div class="panel" style="margin-bottom:10px">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">
            <div><div class="muted">ATS Score</div>
                 <div class="white" style="font-size:18px">{st.session_state.ats_score} / 100</div>
                 <div class="muted">{st.session_state.ats_percentile}th percentile</div></div>
            <div><div class="muted">Semantic Percentile</div>
                 <div class="white" style="font-size:18px">{st.session_state.semantic_percentile}th</div>
                 <div class="muted">Display score: {st.session_state.display_score} / 100</div></div>
            <div><div class="muted">Coverage Percentile</div>
                 <div class="white" style="font-size:18px">{st.session_state.coverage_percentile}th</div>
                 <div class="muted">{st.session_state.domain_coverage_pct}% domain coverage</div></div>
          </div>
        </div>
    """)
    st.markdown(_recap_html, unsafe_allow_html=True)
    st.markdown(
        '<div class="annot" style="border-left:2px solid rgba(127,119,221,0.4);'
        'border-radius:0 6px 6px 0;margin-bottom:14px">'
        'Your analysis is above. The AI feedback below is layered on top of those '
        'scores \u2014 it does not modify them.</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.low_confidence_flag:
        st.warning(
            f"The {st.session_state.confirmed_domain} domain has fewer than 200 benchmark "
            f"candidates. Percentile ranks carry reduced reliability."
        )

    # AI feedback panel
    feedback = st.session_state.ai_feedback
    status   = _classify_ai_feedback(feedback)

    if status in ("success", "partial"):
        st.markdown(_render_ai_feedback_panel(feedback), unsafe_allow_html=True)
    else:
        st.markdown(_render_ai_unavailable_panel(feedback), unsafe_allow_html=True)

    st.markdown("")

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if is_ai_available(st.secrets):
            _retry_label = "Retry" if status == "unavailable" else "Regenerate feedback"
            if st.button(_retry_label, type="primary"):
                st.session_state.ai_feedback = None
                st.session_state.ai_feedback_requested = False
                st.session_state.stage = 2
                st.rerun()

    with col_btn2:
        if st.button("Change domain or re-analyse"):
            _reset_stage_2_and_3()
            st.rerun()