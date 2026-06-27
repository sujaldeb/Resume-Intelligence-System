"""
core/ai_feedback.py

Gemini Flash 2.5 integration for the Resume Intelligence Platform.
Produces structured AI feedback across five dimensions using a single API call.

This module is deliberately isolated from the deterministic pipeline.
It never reads or modifies any numeric score.
The application functions fully if this module fails or the API is unavailable.

Public interface:
    is_ai_available(secrets)        -> bool
    build_feedback_context(...)     -> dict
    get_ai_feedback(context, secrets) -> dict
    parse_feedback_response(raw)    -> dict
"""

import json
import logging
import datetime
from pathlib import Path

from config.settings import (
    AI_FEEDBACK_KEYS,
    GEMINI_MAX_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GAP_SKILLS_IN_CONTEXT,
    JD_SNIPPET_LENGTH,
    SCORE_COMPONENT_MAXIMA,
    SYNTHETIC_CAVEAT,
    LOW_CONFIDENCE_CAVEAT,
    LOW_CONFIDENCE_DOMAINS,
    DEBUG_MODE,
)

logger = logging.getLogger(__name__)

if DEBUG_MODE:
    print("DEBUG ai_feedback.py module loaded — version marker AAFB-001")

# API availability check

def is_ai_available(secrets: object) -> bool:
    """
    Returns True if a Gemini API key is present in st.secrets.
    Does not validate the key — just checks presence.
    Call this before rendering the AI feedback button in the UI.
    """
    try:
        key = secrets.get("GEMINI_API_KEY", "")
        return bool(key and str(key).strip())
    except Exception:
        return False


# Context builder

def build_feedback_context(
    parsed: dict,
    ats_results: dict,
    semantic_results: dict,
    gap_results: dict,
    percentile_results: dict,
    confirmed_domain: str,
) -> dict:
    """
    Assembles the structured context block passed to the Gemini prompt.
    Separates structured signals from raw resume text explicitly.

    Args:
        parsed:             Output from resume_parser.parse_resume().
        ats_results:        Output from pipeline.score_ats().
        semantic_results:   Output from pipeline.run_semantic_matching().
        gap_results:        Output from pipeline.compute_skill_gap().
        percentile_results: Output from pipeline.compute_percentiles().
        confirmed_domain:   User-confirmed domain string.

    Returns a context dict consumed by build_prompt().
    """
    low_conf = percentile_results.get("low_confidence_flag", False)

    return {
        # structured scoring signals
        "domain":               confirmed_domain,
        "experience_band":      ats_results.get("experience_band", "Unknown"),
        "ats_score":            ats_results.get("ats_score"),
        "score_education":      ats_results.get("score_education"),
        "score_experience":     ats_results.get("score_experience"),
        "score_skills":         ats_results.get("score_skills"),
        "score_flags":          ats_results.get("score_flags"),
        "score_completeness":   ats_results.get("score_completeness"),
        "component_maxima":     SCORE_COMPONENT_MAXIMA,
        "ats_percentile":       percentile_results.get("ats_percentile"),
        "semantic_percentile":  percentile_results.get("semantic_percentile"),
        "coverage_percentile":  percentile_results.get("coverage_percentile"),
        "low_confidence_flag":  low_conf,
        "top_job_title":        semantic_results.get("top_job_title", ""),
        "top_job_snippet":      semantic_results.get("top_job_description_snippet", "")[:JD_SNIPPET_LENGTH],
        "full_skill_profile":   parsed.get("full_skill_profile", []),
        "top_missing_skills":   gap_results.get("top_missing_skills", [])[:GAP_SKILLS_IN_CONTEXT],
        "gap_count":            gap_results.get("gap_count", 0),
        "domain_coverage_pct":  gap_results.get("domain_coverage_pct", 0.0),

        # raw resume text — used for resume_review, grammar_feedback, achievement_framing
        "experience_summary":   parsed.get("experience_summary", ""),
        "project_summary":      parsed.get("project_summary", ""),
        "key_achievements":     parsed.get("key_achievements", ""),
    }


# Prompt builder

def _build_prompt(context: dict) -> str:
    """
    Constructs the full Gemini prompt from the assembled context block.
    The prompt instructs the model to return a single JSON object only.
    No preamble, no markdown, no explanation outside the JSON structure.
    """
    low_conf_note = (
        f"\nNote: {LOW_CONFIDENCE_CAVEAT}"
        if context.get("low_confidence_flag") and context.get("domain") in LOW_CONFIDENCE_DOMAINS
        else ""
    )

    ats_breakdown = (
        f"  Education:    {context['score_education']} / {context['component_maxima']['education']}\n"
        f"  Experience:   {context['score_experience']} / {context['component_maxima']['experience']}\n"
        f"  Skills:       {context['score_skills']} / {context['component_maxima']['skills']}\n"
        f"  Flags:        {context['score_flags']} / {context['component_maxima']['flags']}\n"
        f"  Completeness: {context['score_completeness']} / {context['component_maxima']['completeness']}\n"
        f"  Total:        {context['ats_score']} / 100"
    )

    percentile_note = (
        f"  ATS percentile:      {context['ats_percentile']}th (within {context['domain']} domain)\n"
        f"  Semantic percentile: {context['semantic_percentile']}th (within {context['domain']} domain)\n"
        f"  Coverage percentile: {context['coverage_percentile']}th (within {context['domain']} domain)\n"
        f"  Caveat: {SYNTHETIC_CAVEAT}"
        f"{low_conf_note}"
    )

    skill_profile_str   = ", ".join(context["full_skill_profile"]) or "none detected"
    missing_skills_str  = ", ".join(context["top_missing_skills"]) or "none identified"

    prompt = f"""You are an expert resume reviewer and career coach specializing in the {context['domain']} domain.

You have been given a candidate's resume data and their ATS analysis results.
Your task is to produce structured, actionable feedback across five dimensions.

Return ONLY a valid JSON object with exactly these five keys:
  resume_review
  grammar_feedback
  achievement_framing
  skill_gap_narrative
  action_plan

Do not include any text before or after the JSON object.
Do not use markdown code blocks.
Do not add any keys beyond the five specified.

Every value for all five keys must be a single plain text string.
Never use a JSON array or a JSON object as a value, even for content
that is naturally a list, such as achievement_framing or action_plan.
Represent lists, numbered steps, and before/after comparisons as plain
text inside the string itself, using newline characters and standard
numbering or hyphens. The entire response must be a flat JSON object
with five string values and nothing more.

---

CANDIDATE PROFILE

Domain: {context['domain']}
Experience Band: {context['experience_band']}

ATS Score Breakdown:
{ats_breakdown}

Benchmark Percentiles:
{percentile_note}

Current Skills: {skill_profile_str}

Top Missing Skills ({context['gap_count']} total gaps, showing top {GAP_SKILLS_IN_CONTEXT}):
{missing_skills_str}

Best Matching Job Title: {context['top_job_title']}

Job Description Excerpt:
{context['top_job_snippet']}

---

RESUME TEXT

Experience:
{context['experience_summary']}

Projects:
{context['project_summary']}

Key Achievements:
{context['key_achievements']}

---

FEEDBACK INSTRUCTIONS

resume_review:
  2-3 paragraphs assessing overall resume quality for the {context['domain']} domain.
  Reference specific strengths and specific weaknesses.
  Be direct and professional. Do not use generic praise.

grammar_feedback:
  Identify specific grammar, punctuation, or clarity issues found in the resume text.
  Quote the problematic phrase, then provide the corrected version.
  If no issues are found, state that clearly rather than inventing problems.
  Format as a list of specific findings.

achievement_framing:
  Rewrite 3-5 bullet points from the experience or project sections to be stronger.
  Use the STAR format (Situation, Task, Action, Result) where the original is vague.
  Focus on quantified outcomes. If a number is already present, keep it.
  Show the original and the improved version for each bullet.

skill_gap_narrative:
  Explain the skill gaps in plain language relevant to the {context['domain']} domain.
  Prioritise the top missing skills by their hiring market relevance.
  Suggest specific, realistic ways to close each gap.
  Do not simply list the skills — contextualise why each matters.

action_plan:
  Provide 4-6 concrete, prioritised next steps the candidate should take.
  Each step should be specific and actionable within 30-90 days.
  Order by impact. Do not include generic advice like "update your LinkedIn".
"""

    return prompt


# API call

def _call_gemini(prompt: str, api_key: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai package is not installed. "
            "Add it to requirements.txt and reinstall."
        )

    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_TOKENS,
        response_mime_type="application/json",
    )

    model    = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=generation_config,
    )

    if DEBUG_MODE:
        try:
            finish_reason = response.candidates[0].finish_reason
            print("DEBUG Gemini finish_reason:", finish_reason, flush=True)
        except Exception:
            print("DEBUG Could not read finish_reason from response.", flush=True)

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response. "
            "The request may have been blocked by the safety filter."
        )

    if DEBUG_MODE:
        print("DEBUG Gemini raw response length:", len(response.text), "chars", flush=True)
        print("DEBUG Gemini raw response head:", response.text[:300], flush=True)
        print("DEBUG Gemini raw response tail:", response.text[-300:], flush=True)

    return response.text


# Response parser

DEBUG_LOG_DIR = Path(__file__).parent / "debug_logs"


def _save_failed_response(raw: str, cleaned: str, error: str) -> Path:
    """
    Writes a failing Gemini response to disk for diagnosis.
    Captures the original raw text, the text after fence/brace stripping,
    and the exact JSONDecodeError message. Returns the path written to.
    Runs regardless of DEBUG_MODE — this is a silent file write, not console
    output, and is the only record of a failure once the process moves on.
    """
    DEBUG_LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath  = DEBUG_LOG_DIR / f"failed_response_{timestamp}.txt"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"ERROR: {error}\n")
        f.write("=" * 60 + "\n")
        f.write("RAW RESPONSE (as received from Gemini)\n")
        f.write("=" * 60 + "\n")
        f.write(raw)
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("CLEANED RESPONSE (after fence/brace stripping)\n")
        f.write("=" * 60 + "\n")
        f.write(cleaned)

    return filepath


def parse_feedback_response(raw: str) -> dict:
    """
    Parses the raw Gemini response string into a structured feedback dict.
    Strips markdown fences if present.
    Extracts the outermost JSON object as a defensive backstop in case
    the model includes stray text outside the JSON despite instructions.
    On parse failure, saves the raw response to disk for diagnosis and
    surfaces the exact JSONDecodeError message to the caller.
    Returns a dict with all five feedback keys.
    Missing keys are filled with a graceful fallback string rather than raising.

    Returns dict with keys matching AI_FEEDBACK_KEYS.
    Each value is a non-empty string.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        lines   = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    try:
        parsed = json.loads(cleaned, strict=False)
    except json.JSONDecodeError as exc:
        filepath = _save_failed_response(raw, cleaned, str(exc))
        if DEBUG_MODE:
            print("DEBUG Gemini response was not valid JSON:", exc, flush=True)
            print("DEBUG Failed response saved to:", filepath, flush=True)
        return _fallback_response(
            f"The AI response could not be parsed (JSONDecodeError: {exc}). "
            f"Raw response saved to {filepath.name} for diagnosis."
        )

    result = {}
    for key in AI_FEEDBACK_KEYS:
        value = parsed.get(key, "")
        if not value or not str(value).strip():
            result[key] = _section_unavailable(key)
        else:
            result[key] = str(value).strip()

    return result


def _section_unavailable(key: str) -> str:
    return f"This section was not returned by the AI. Regenerate feedback to retry."


def _fallback_response(message: str) -> dict:
    return {key: message for key in AI_FEEDBACK_KEYS}


# Public entry point

def get_ai_feedback(context: dict, secrets: object) -> dict:
    if DEBUG_MODE:
        print("DEBUG get_ai_feedback() called", flush=True)

    try:
        api_key = str(secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        api_key = ""

    if not api_key:
        return _fallback_response(
            "AI feedback is not available. "
            "A Gemini API key has not been configured for this deployment."
        )

    try:
        prompt   = _build_prompt(context)
        raw      = _call_gemini(prompt, api_key)
        feedback = parse_feedback_response(raw)
        return feedback

    except RuntimeError as exc:
        if DEBUG_MODE:
            print("DEBUG AI feedback failed:", exc, flush=True)
        return _fallback_response(
            f"AI feedback could not be generated: {exc}"
        )
    except Exception as exc:
        if DEBUG_MODE:
            print("DEBUG Unexpected error in AI feedback:", exc, flush=True)
        return _fallback_response(
            "AI feedback encountered an unexpected error. "
            "Scores and analysis are unaffected."
        )