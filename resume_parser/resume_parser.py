# -*- coding: utf-8 -*-

import re
import json
import hashlib
import pdfplumber
import docx
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "parser_config.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)

SECTION_HEADINGS     = CONFIG["section_headings"]
HEADING_MAX_CHARS    = CONFIG["section_heading_max_chars"]
DEGREE_MAP           = CONFIG["degree_keyword_map"]
DOMAIN_TITLE_KW      = CONFIG["domain_title_keywords"]
DOMAIN_SKILL_SIGNALS = CONFIG["domain_skill_signals"]
FLAG_RULES           = CONFIG["flag_rules"]
SUPP_VOCAB           = CONFIG["supplementary_vocabulary"]

HEADING_LOOKUP = {
    variant.lower().strip(): section
    for section, variants in SECTION_HEADINGS.items()
    for variant in variants
}

TRAILING_DATE = re.compile(
    r"\s+(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[\w\s\-\u2013\u2014]*\d{4}|\d{4}\s*[-\u2013\u2014]\s*\d{4}|\d{4})\s*$",
    re.IGNORECASE
)

CURRENT_YEAR = datetime.now().year

SUPPLEMENTARY_TOKENS = sorted(
    {t for group in SUPP_VOCAB.values() for t in group},
    key=len, reverse=True
)

GAP_SUPPRESS_TOKENS = set(CONFIG.get("gap_suppress_tokens", []))


def extract_text(file_bytes, file_extension):
    if file_extension == "pdf":
        return _extract_pdf(file_bytes)
    elif file_extension in ("docx", "doc"):
        return _extract_docx(file_bytes)
    return "", "unsupported_format"


def _extract_pdf(file_bytes):
    import io
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(pages).strip()
        if len(text) < 200:
            return text, "low_character_yield"
        return text, None
    except Exception as e:
        return "", f"pdf_error: {e}"


def _extract_docx(file_bytes):
    import io
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs).strip()
        if len(text) < 200:
            return text, "low_character_yield"
        return text, None
    except Exception as e:
        return "", f"docx_error: {e}"


def detect_sections(raw_text):
    sections = {}
    current  = "header"
    sections[current] = []

    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        normalized = TRAILING_DATE.sub("", stripped).strip().lower()
        is_heading = (
            normalized in HEADING_LOOKUP
            and "•" not in stripped
            and "-" not in stripped[:2]
            and len(normalized) <= HEADING_MAX_CHARS
            and len(normalized) >= 2
        )
        if is_heading:
            current = HEADING_LOOKUP[normalized]
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(stripped)

    return sections


def extract_years_experience(experience_lines):
    year_pat    = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
    present_pat = re.compile(
        r"\b(present|current|now|till date|to date)\b", re.IGNORECASE
    )
    all_years   = []
    has_present = False

    for line in experience_lines:
        all_years.extend(int(y) for y in year_pat.findall(line))
        if present_pat.search(line):
            has_present = True

    if not all_years:
        return None, "no_dates_found"

    start = min(all_years)
    end   = CURRENT_YEAR if has_present else max(all_years)
    years = max(1, end - start)
    confidence = "ok" if len(all_years) >= 2 else "low_date_count"
    return years, confidence


def extract_education(education_lines):
    tier_priority = {
        "Postgraduate": 5, "Masters": 4, "MBA": 3,
        "Bachelors": 2, "Associate": 1, "Unknown": 0
    }
    best_tier  = "Unknown"
    best_score = 0
    full_text  = " ".join(education_lines).lower()

    for keywords, tier in DEGREE_MAP:
        for kw in keywords:
            if kw in full_text and tier_priority[tier] > best_score:
                best_tier  = tier
                best_score = tier_priority[tier]
                break

    return best_tier


def extract_recent_title(experience_lines):
    for line in experience_lines[:6]:
        stripped = line.strip()
        if not stripped or stripped.startswith("•"):
            continue
        if re.search(r"\b(19[89]\d|20[0-3]\d)\b", stripped):
            continue
        title = stripped.split(",")[0].strip()
        if title and len(title) > 3:
            return title
    return ""


def assign_domain(job_title, skill_profile):
    title_lower = job_title.lower()
    for domain in ["Data Science", "Legal", "HR", "Engineering", "IT", "Management"]:
        if any(kw in title_lower for kw in DOMAIN_TITLE_KW.get(domain, [])):
            return domain, "title"

    skill_set = set(skill_profile)
    scores    = {}
    for domain, signals in DOMAIN_SKILL_SIGNALS.items():
        score = sum(1 for s in signals if s in skill_set)
        if score > 0:
            scores[domain] = score

    if scores:
        return max(scores, key=scores.get), "skill_vote"

    return None, "unclassified"


# Bug fix: removed suppress_tokens parameter — suppression is handled at
# the module level via GAP_SUPPRESS_TOKENS loaded from parser_config.json
def extract_skills(full_text, canonical_tokens, normalization_lookup):
    text_lower     = full_text.lower()
    canonical_norm = []
    supplementary  = []

    for token in canonical_tokens:
        if token in GAP_SUPPRESS_TOKENS:
            continue
        if re.search(r"\b" + re.escape(token) + r"\b", text_lower):
            canonical_norm.append(normalization_lookup.get(token, token))

    for token in SUPPLEMENTARY_TOKENS:
        if re.search(r"\b" + re.escape(token) + r"\b", text_lower):
            supplementary.append(token)

    seen = set()
    canon_deduped = []
    for t in canonical_norm:
        if t not in seen:
            seen.add(t)
            canon_deduped.append(t)

    supp_deduped = []
    for t in supplementary:
        if t not in seen:
            seen.add(t)
            supp_deduped.append(t)

    return (
        sorted(canon_deduped),
        sorted(supp_deduped),
        sorted(canon_deduped + supp_deduped)
    )


def extract_flags(full_text):
    text_lower = full_text.lower()
    flags = {}
    for flag, keywords in FLAG_RULES.items():
        fired = 0
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                fired = 1
                break
        flags[flag] = fired
    return flags


def parse_resume(file_bytes, file_extension, canonical_tokens,
                 normalization_lookup):
    file_hash = hashlib.md5(file_bytes).hexdigest()
    raw_text, extract_error = extract_text(file_bytes, file_extension)

    if extract_error and not raw_text:
        return {"parse_error": extract_error, "file_hash": file_hash}

    sections = detect_sections(raw_text)

    exp_lines  = sections.get("experience", [])
    edu_lines  = sections.get("education", [])
    proj_lines = sections.get("projects", [])
    sum_lines  = sections.get("summary", [])
    ach_lines  = sections.get("achievements", [])

    years_exp, exp_confidence = extract_years_experience(exp_lines)
    education                 = extract_education(edu_lines)
    canon, supp, full         = extract_skills(raw_text, canonical_tokens,
                                               normalization_lookup)
    flags                     = extract_flags(raw_text)
    recent_title              = extract_recent_title(exp_lines)
    domain, method            = assign_domain(recent_title, canon)

    exp_text  = " ".join(exp_lines)
    proj_text = " ".join(proj_lines) if proj_lines else ""
    # Bug fix: key_achievements no longer falls back to exp_text
    ach_text  = " ".join(ach_lines) if ach_lines else ""
    soft_text = " ".join(sum_lines) if sum_lines else ""

    return {
        "file_hash":                   file_hash,
        "parse_error":                 extract_error,
        "raw_text":                    raw_text,
        "sections":                    dict(sections),
        "years_experience":            years_exp,
        "exp_confidence":              exp_confidence,
        "highest_education":           education,
        "institution_tier":            "Unknown",
        "canonical_skill_profile":     canon,
        "supplementary_skill_profile": supp,
        "full_skill_profile":          full,
        "flags":                       flags,
        "detected_domain":             domain,
        "domain_method":               method,
        "primary_role":                recent_title,
        "experience_summary":          exp_text,
        "project_summary":             proj_text,
        "key_achievements":            ach_text,
        "soft_skills_raw":             soft_text,
    }


def get_parse_warnings(parsed):
    warnings = []

    if parsed.get("years_experience") is None:
        warnings.append({
            "field":    "years_experience",
            "severity": "high",
            "message":  (
                "Experience dates were not detected in your resume. "
                "The experience score may be understated. "
                "Ensure your roles include clear start and end dates."
            )
        })

    if parsed.get("exp_confidence") == "low_date_count":
        warnings.append({
            "field":    "years_experience",
            "severity": "medium",
            "message":  (
                "Only one date was detected in your experience section. "
                "Career span may be inaccurate."
            )
        })

    if not parsed.get("canonical_skill_profile"):
        warnings.append({
            "field":    "canonical_skill_profile",
            "severity": "high",
            "message":  (
                "No recognizable skills were extracted from your resume. "
                "The skill score will be at its minimum. "
                "Ensure your skills section uses standard terminology."
            )
        })

    if parsed.get("detected_domain") is None:
        warnings.append({
            "field":    "detected_domain",
            "severity": "high",
            "message":  (
                "Your professional domain could not be determined automatically. "
                "Please select your domain manually before scoring."
            )
        })

    if not parsed.get("project_summary"):
        warnings.append({
            "field":    "project_summary",
            "severity": "low",
            "message":  (
                "No projects section was detected. "
                "Adding a projects section can improve your completeness score."
            )
        })

    if not parsed.get("soft_skills_raw"):
        warnings.append({
            "field":    "soft_skills_raw",
            "severity": "low",
            "message":  (
                "No professional summary or soft skills section was detected."
            )
        })

    return warnings