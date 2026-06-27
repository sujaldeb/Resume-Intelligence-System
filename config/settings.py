"""
config/settings.py

Central configuration for the Resume Intelligence Platform.
All paths, constants, and validation schemas live here.
No other module should define these values independently.
"""

from pathlib import Path

# Root paths

# resolves correctly whether imported from app.py, core/, or a notebook
PROJECT_ROOT = Path(__file__).parent.parent

OUTPUTS_DIR      = PROJECT_ROOT / "outputs"
RESUME_PARSER_DIR = PROJECT_ROOT / "resume_parser"
STREAMLIT_DIR    = PROJECT_ROOT / ".streamlit"

# Runtime artifact paths

ARTIFACT_PATHS = {
    "embedding": OUTPUTS_DIR / "runtime_embedding_artifacts.pkl",
    "benchmark": OUTPUTS_DIR / "runtime_benchmark_lookup.json",
    "scoring":   OUTPUTS_DIR / "ats_scoring_artifacts.json",
    "job_desc":  OUTPUTS_DIR / "curated_job_descriptions.csv",
    "esco":      OUTPUTS_DIR / "esco_skill_mapping.csv",
}

PARSER_CONFIG_PATH = RESUME_PARSER_DIR / "parser_config.json"

# Model

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384

# Domains

SUPPORTED_DOMAINS = {
    "Data Science",
    "Engineering",
    "HR",
    "IT",
    "Legal",
    "Management",
}

# display order used in dropdowns and UI elements
DOMAIN_DISPLAY_ORDER = [
    "Data Science",
    "Engineering",
    "HR",
    "IT",
    "Legal",
    "Management",
]

# domains flagged as low confidence due to thin benchmark population
LOW_CONFIDENCE_DOMAINS = {"Engineering", "Management"}

# ATS scoring

EXPERIENCE_MAX_YEARS = 20

SKILL_SCORE_MAX_RAW  = 6.0
SKILL_SCORE_FLOOR    = 5 * 0.167     # minimum weighted sum before scaling

SCORE_COMPONENT_MAXIMA = {
    "education":    20,
    "experience":   25,
    "skills":       30,
    "flags":        15,
    "completeness": 10,
}

COMPLETENESS_THRESHOLD_FULL    = 50   # characters — full credit
COMPLETENESS_THRESHOLD_PARTIAL = 10   # characters — half credit

EDUCATION_TIERS = {
    "Postgraduate": 20,
    "Masters":      20,
    "MBA":          20,
    "Bachelors":    14,
    "Unknown":       7,
}

EXPERIENCE_BANDS = [
    (1,  3,  "Junior (1-3)"),
    (4,  6,  "Mid (4-6)"),
    (7,  10, "Senior (7-10)"),
    (11, 20, "Expert (11-20)"),
]

# Semantic matching

# rescaling parameters are loaded from runtime_embedding_artifacts.pkl
# at startup — these are not hardcoded here

SEMANTIC_METRICS = {
    "display_score":       "Semantic Match",
    "ats_score":           "ATS Readiness",
    "domain_coverage_pct": "Skill Coverage",
}

# Artifact validation schemas

REQUIRED_EMBEDDING_KEYS = {
    "job_embeddings",
    "job_id_to_idx",
    "domain_job_indices",
    "similarity_rescaling",
    "domain_skill_freq_filtered",
    "gap_suppress_tokens",
}

REQUIRED_SCORING_KEYS = {
    "education_scores",
    "flag_weights",
    "flag_cols",
    "skill_concentration_weights",
    "component_max_points",
}

REQUIRED_JOB_DESC_COLS = {
    "job_id",
    "title",
    "description",
    "domain",
    "normalized_skills",
}

REQUIRED_ESCO_COLS = {
    "canonical_token",
    "esco_preferred_label",
    "token_category",
}

BENCHMARK_SCORE_METRICS = (
    "ats_score",
    "display_score",
    "domain_coverage_pct",
)

# AI feedback
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE   = 0.4
GEMINI_MAX_TOKENS    = 8192
DEBUG_MODE = False

AI_FEEDBACK_KEYS = (
    "resume_review",
    "grammar_feedback",
    "achievement_framing",
    "skill_gap_narrative",
    "action_plan",
)

# characters of JD description text passed to the AI context block
JD_SNIPPET_LENGTH = 800

# number of missing skills passed to the AI context block
GAP_SKILLS_IN_CONTEXT = 5

# Session state stages

STAGE_NO_FILE        = 0
STAGE_PARSED         = 1
STAGE_SCORED         = 2
STAGE_AI_COMPLETE    = 3

# UI presentation

APP_TITLE       = "Resume Intelligence"
APP_SUBTITLE    = "ATS Readiness and Role Matching"

# warning severity levels returned by get_parse_warnings()
WARNING_HIGH   = "high"
WARNING_MEDIUM = "medium"
WARNING_LOW    = "low"

# synthetic data caveat shown on all percentile displays
SYNTHETIC_CAVEAT = (
    "Percentile ranks are benchmarked against a synthetic candidate population "
    "of 5,000 records and are directional only."
)

LOW_CONFIDENCE_CAVEAT = (
    "This domain has fewer than 200 benchmark candidates. "
    "Percentile ranks carry reduced reliability."
)