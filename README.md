# Resume Intelligence Platform
### End-to-End NLP Pipeline + Streamlit Application
**Python · sentence-transformers · pdfplumber · Streamlit · Gemini API**

---

## One-Line Summary

> Built an end-to-end resume screening system spanning 9 research notebooks and a production Streamlit application — engineering a domain-stratified ATS scoring engine, a semantic role-matching layer validated to correctly separate six job domains (IT-vs-IT similarity 0.836 vs IT-vs-HR 0.677), and a two-tier skill extraction architecture that resolves the precision/recall tradeoff between auditable scoring and real-world resume coverage, all running on CPU with a 383 KB runtime footprint.

---

## Live Demo

Deployment to Streamlit Community Cloud in progress — link will be added here once live. The application runs locally via `streamlit run app.py`; see Setup & Installation below.

---

## Problem Statement

Applicant Tracking Systems are a black box to most job seekers — a resume can be rejected by automated screening before a human ever reads it, with no visibility into why. Most public "ATS checkers" are either pure keyword counters with no semantic understanding, or thin wrappers around a single LLM call with no deterministic, auditable scoring underneath.

**Core problems this project addresses:**
- Resume readiness scoring needs to be deterministic and explainable, not just an LLM's opinion
- Keyword matching alone produces false positives (`aws` matching inside `draws`) and misses genuine semantic alignment between a resume and a job description
- A scoring system trained or calibrated on one candidate population will silently misrank candidates from a different population (seniority, domain, vocabulary) unless that is explicitly modeled
- Real resumes contain far more skill vocabulary than any small, manually curated taxonomy can cover — but expanding that taxonomy without discipline breaks the calibration of any score built on it

**Design response:** keep every score component deterministic and inspectable, use semantic embeddings only where they add real signal, and use an LLM exclusively for qualitative narrative feedback that sits on top of the scores without ever modifying them.

---

## Repository Structure

```
resume-intelligence-platform/
│
├── app.py                          # Streamlit application — 4-stage session flow
├── config/
│   └── settings.py                 # Central configuration, single source of truth
├── core/
│   ├── resources.py                 # Runtime artifact loading and cross-validation
│   ├── pipeline.py                  # ATS scoring, semantic matching, gap analysis,
│   │                                 # percentile benchmarking
│   └── ai_feedback.py               # Gemini integration, single-call architecture
├── resume_parser/
│   ├── resume_parser.py             # PDF/DOCX parsing, section detection
│   ├── parser_config.json           # All extraction vocabularies and keyword rules
│   └── validation_report.md         # Real-resume validation results
├── static/
│   └── styles.css                   # Custom dark-theme component library
├── notebooks/                       # 01-09, frozen research record
│   ├── 01_data_exploration_validation.ipynb
│   ├── 02_preprocessing_skill_profiles.ipynb
│   ├── 03_esco_skill_normalization.ipynb
│   ├── 04_job_description_corpus_preparation.ipynb
│   ├── 05_ats_scoring_engine.ipynb
│   ├── 06_semantic_matching_and_skill_gap_analysis.ipynb
│   ├── 07_benchmarking_and_candidate_ranking.ipynb
│   ├── 08_runtime_resume_processing.ipynb
│   └── 09_application_architecture_and_deployment_preparation.ipynb
├── outputs/                         # 5 runtime artifacts (see below)
├── requirements.txt
└── README.md
```

---

## Data Description

| Source | Scope | Role |
|---|---|---|
| LinkedIn Job Postings (Kaggle) | 252 postings curated, 42 per domain | Job description corpus for semantic matching and gap analysis |
| ESCO Skills Taxonomy | 35 canonical skill tokens, normalized from 217 raw tokens | Skill vocabulary normalization layer |
| Synthetic candidate population | 5,000 records, 6 domains | Benchmark reference population for percentile scoring only — never used to score real resumes |
| Real resumes | 6 personal resumes across Data Science and IT | Parser and end-to-end pipeline validation |

**Domain distribution (synthetic benchmark population):** IT 2,772 · Data Science 1,002 · HR 501 · Legal 397 · Engineering 189 · Management 139.

Engineering and Management fall below the 200-record threshold for stable percentile estimation and are explicitly flagged `low_confidence` everywhere they appear in the UI and the benchmark lookup table.

---

## Methodology

### Stage 1 — Data Exploration & Skill Vocabulary Cleaning (Notebooks 01-02)
- Profiled the 5,000-record synthetic population: confirmed several binary flags are domain-level constants with zero within-domain variance, identified `LLM` education label as a generic postgraduate tier (not law-specific) requiring remapping to `Masters`
- Deduplicated overlapping skill tokens across two raw fields (e.g. `aws` appeared in both `technical_skills_raw` and `tools_platforms_raw`)
- 217 raw skill tokens collapsed to 35 canonical tokens via a 181-entry typo-correction map

### Stage 2 — ESCO Skill Normalization (Notebook 03)
- Fuzzy-matched canonical tokens against ESCO preferred labels — 14 mapped cleanly, 14 retained as platform/tool names with no ESCO equivalent (`aws`, `docker`, `jira`), 7 retained as legitimate skills absent from ESCO at resume abstraction level (`stakeholder management`, `due diligence`)
- Explicit design decision: ESCO normalization is supplemental enrichment only — canonical tokens remain the primary matching representation throughout the pipeline, so an unmapped skill is never silently dropped

### Stage 3 — Job Description Corpus & ATS Scoring Engine (Notebooks 04-05)
- Curated 252 job descriptions (42 per domain) with mandatory word-boundary regex extraction — plain substring matching was rejected after confirming it produces false positives such as `aws` matching inside `draws`
- Five-component ATS score (Experience 25 pts, Skill Coverage 30 pts, Education 20 pts, Experience Flags 15 pts, Profile Completeness 10 pts), each independently computed and explainable
- Skill coverage uses domain-concentration weighting rather than raw count — a skill appearing in only one domain scores full weight; a skill appearing in all six domains (`aws`, `jira`) scores 0.167, preventing generic tools from inflating domain-specific skill signal
- Flag scoring uses within-domain variance weighting — four flags that were domain-level constants in the synthetic data received near-zero weight automatically rather than requiring manual exclusion

### Stage 4 — Semantic Matching & Skill Gap Analysis (Notebook 06)
- Job descriptions embedded from full description text; candidate profiles embedded from a constructed natural-language skill sentence (`"Skills include: token1, token2..."`) — both using `all-MiniLM-L6-v2` (384-dim, CPU-only)
- Domain-stratified cosine similarity only — a candidate is scored against their own domain's 42 job descriptions, with cross-domain matching deliberately deferred to runtime where the user selects a specific posting
- Skill gap computed as canonical-token set difference against a domain frequency table, with two cross-domain noise tokens (`strategy`, `ensure ongoing compliance with regulations`) explicitly suppressed after they appeared as the top "skill" in all six domains — a clear language artifact rather than a genuine signal

### Stage 5 — Benchmarking (Notebook 07)
- Domain-stratified percentile ranks (not global) for three independent signals: ATS score, semantic display score, skill coverage percentage
- Rejected global percentiles outright — the data shows genuine seniority differences across domains (Management mean experience 10.92 years vs Data Science 5.54 years), so a global rank would conflate domain effects with candidate quality
- `low_confidence_flag` applied to any domain under 200 records rather than silently producing an equally-confident-looking percentile

### Stage 6 — Real-World Parser Validation (Notebooks 08-09)
- Two-tier skill architecture (**Option C**): a 35-token canonical vocabulary drives the calibrated ATS skill score; a 115-token supplementary vocabulary (real frameworks, libraries, cloud services) extends recall for semantic matching and gap analysis only — this was a direct response to discovering the canonical vocabulary alone captured only 20-25% of a real resume's extractable skills
- Section detection required a trailing-date-pattern fix after discovering headings like `"Projects Jan 2026"` failed naive heading lookup
- Validated end-to-end against 6 real resumes: 4 completed the full pipeline successfully; 2 correctly failed automatic domain classification (Digital Marketing, UI/UX — genuinely out-of-scope domains) and triggered the manual-selection warning path exactly as designed, with zero blocking parser bugs

### Stage 7 — Deployment Architecture (Notebook 09)
- Stripped the 5,000-candidate embedding matrix from the runtime artifact entirely — `embedding_artifacts.pkl` (7.77 MB) reduced to `runtime_embedding_artifacts.pkl` (383 KB), a 95.2% reduction, since only the 252 job embeddings are needed at runtime
- Replaced `scipy.stats.percentileofscore` with a pre-computed sorted-array + `bisect` lookup for runtime percentile computation — sub-millisecond, stdlib-only, max deviation of 1.12 percentile points from the exact value, eliminating `pandas` and `scipy` as runtime dependencies entirely

### Stage 8 — AI Feedback Layer
- Single Gemini API call per user-triggered request returns a structured 5-key JSON response (resume review, grammar feedback, achievement reframing, skill gap narrative, action plan)
- Hard architectural boundary: the AI layer reads pipeline outputs as context and never writes back to any score — every numeric value in the application is deterministic regardless of AI availability
- Per-section graceful degradation — a partial or failed API response still renders whichever sections succeeded, with explicit unavailability notices for the rest, rather than failing the whole panel

---

## Key Results

### ATS Scoring (5,000-candidate synthetic population)

| Component | Max Points | Mean | Basis |
|---|---|---|---|
| Experience | 25 | 15.67 | Log-scaled years, diminishing returns at senior levels |
| Skill Coverage | 30 | 18.49 | Domain-concentration weighted |
| Education | 20 | 17.95 | Tier-mapped, Unknown receives partial credit not zero |
| Experience Flags | 15 | 8.21 | Within-domain variance weighted |
| Profile Completeness | 10 | 8.75 | Text section presence and length |
| **Total ATS Score** | **100** | **69.07** | std 10.61, range 36.9-89.96 |

### Semantic Matching Cross-Domain Separation

| Comparison | Mean Cosine Similarity |
|---|---|
| IT vs IT | 0.836 |
| IT vs Data Science | 0.751 |
| IT vs HR | 0.677 |

Confirms the embedding space correctly orders within-domain similarity above cross-domain similarity before any scoring logic is built on top of it.

### Runtime Footprint

| Metric | Value |
|---|---|
| Total runtime artifact size | 2.87 MB across 5 files |
| Embedding artifact reduction | 7.77 MB to 383 KB (95.2%) |
| Percentile lookup latency | <0.1 ms (bisect on pre-sorted arrays) |
| Percentile lookup accuracy | Within 1.12 points of exact `percentileofscore` |

### Real-Resume Parser Validation

| Resume | Domain | Pipeline Result |
|---|---|---|
| Data Science (x3 resumes) | Data Science | Complete |
| IT | IT | Complete — high-severity warning fired correctly for missing dates |
| Digital Marketing | Out of scope | Correctly unclassified, manual-selection path triggered |
| UI/UX | Out of scope | Correctly unclassified, manual-selection path triggered |

Zero blocking parser bugs across the validation set.

### Pipeline Performance

| Stage | Latency |
|---|---|
| Sentence-transformer model load (cached once via `st.cache_resource`) | ~9-11s, cold start only |
| Deterministic pipeline (parse + score + match + gap + benchmark) | Under 5s |
| With AI feedback (single Gemini call) | 8-12s estimated |

The deterministic path stays within the original 10-second target from the project's architecture phase; the model load cost is paid once per application instance, not per request.

### Skill Vocabulary Coverage

| Tier | Tokens | Role |
|---|---|---|
| Canonical (ESCO-normalized) | 35 | Drives the calibrated ATS skill score |
| Supplementary (real-world tools, frameworks, cloud services) | 115 | Extends recall for semantic matching and gap analysis only |

Split across 5 supplementary categories: data science/ML tooling, databases, IT/DevOps, cloud platforms, and general office tools — added after discovering the canonical-only vocabulary captured just 20-25% of a real resume's extractable skills.

---

## Key Architectural Decisions

**Domain-stratified scoring over global** — Management candidates are genuinely more senior (mean 10.92 years vs population 6.73) and more flag-active than other domains. A global score or percentile would punish every other domain for this real difference rather than measuring candidates against their actual peer group.

**Two-tier skill vocabulary (Option C) over a single expanded taxonomy** — ESCO-normalizing 150+ real-world skill tokens before deployment would have taken disproportionate effort for the value gained. Splitting into a calibration-locked canonical set (ATS scoring) and a recall-maximizing supplementary set (semantic matching, gap analysis) solved the precision/recall tradeoff without re-deriving every weight in the scoring engine.

**Word-boundary regex over substring matching, without exception** — confirmed early that substring matching produces unacceptable false positives (`aws` inside `draws`) for short skill tokens. Applied universally across every skill-extraction step in the pipeline, not just where the bug was first found.

**bisect over scipy/pandas for runtime percentiles** — the benchmark population is fixed, so a one-time precomputed sorted-array lookup eliminates two heavy runtime dependencies and gets sub-millisecond latency, at a cost of under 1.2 percentile points of precision — an explicitly acceptable tradeoff given the benchmark is already a synthetic-population caveat.

**Suppressing cross-domain noise tokens in gap analysis** — `strategy` and a generic compliance phrase appeared as the top "skill" in every single domain's job postings, which is a language artifact of how the postings were written, not a real differentiating signal. Left unsuppressed, they would have dominated every candidate's gap analysis regardless of domain.

**AI feedback as a read-only layer over the deterministic pipeline** — every score in this application is computed identically whether or not the Gemini API is reachable. The LLM is used exclusively for narrative feedback that a deterministic pipeline cannot produce (grammar correction, achievement reframing), never for anything a number already answers.

**Three-tier parse warnings over silent degradation** — when the parser can't find experience dates or a domain, the application surfaces a `high`/`medium`/`low` severity warning rather than silently producing a score the user has no reason to trust. Scoring still completes either way; the warning exists so a low experience score caused by a missing date range doesn't get mistaken for a genuinely weak resume.

---

## Debugging Notable During Development

**Silent lookup failure traced to a structural mismatch, not a logic bug.** The semantic-matching layer's "best matching job" title and snippet were returning empty despite the underlying cosine similarity score computing correctly. The root cause: a precomputed lookup dict had been attached to the wrong object — stored as a sibling key on the top-level resources dictionary rather than nested inside the embedding artifact that the matching function actually receives as its argument. The fix was inverting the index locally inside the function that needed it, rather than trusting a precomputed structure assembled one layer too high. The bug was invisible at the data level — similarity scores and percentiles were correct throughout — which is exactly why it had gone unnoticed until the UI was built far enough to display the missing field.

**HTML escaping failure isolated to a specific Markdown-renderer interaction.** Custom-styled result cards intermittently rendered as literal escaped HTML text instead of styled components. Isolated the cause to Streamlit's underlying CommonMark renderer treating multi-line, indented, concatenated HTML f-strings as code blocks rather than raw HTML — a renderer-specific quirk, not a string-formatting bug. Fixed by collapsing every HTML template to a single line before passing it to the renderer, applied as a project-wide convention rather than a one-off patch.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| Application Framework | Streamlit |
| Semantic Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), PyTorch (CPU-only) |
| PDF / DOCX Parsing | pdfplumber, python-docx |
| Data Processing | pandas, NumPy |
| Skill Normalization | ESCO Skills Taxonomy |
| Percentile Computation | `bisect` (stdlib) — no scipy/pandas at runtime |
| AI Feedback | Google Gemini API |
| Environment | Jupyter Notebook, conda |
| Version Control | Git, GitHub |

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/sujaldeb/Resume-Intelligence-System.git
cd Resume-Intelligence-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Gemini API key (see SECRETS.md)
# Create .streamlit/secrets.toml:
#   GEMINI_API_KEY = "your-key-here"
# The app runs fully without this — AI feedback is the only feature that needs it.

# 4. Run the application
streamlit run app.py
```

> **Note:** The deterministic pipeline (parsing, ATS scoring, semantic matching, benchmarking) requires no external API. AI feedback degrades gracefully and is disabled cleanly in the UI if no key is configured.

---

## Limitations & Future Work

- Benchmark percentiles are derived from a 5,000-record **synthetic** population, not real resume data — treated as directional throughout the application, with an explicit caveat on every percentile display
- Engineering and Management domains have fewer than 200 benchmark records and are flagged `low_confidence` wherever they appear
- Semantic display scores are not comparable across domains due to vocabulary alignment differences in the job posting corpus — the within-domain percentile is the only cross-candidate-comparable signal, and is surfaced as the primary metric in the UI for this reason
- Two-column PDF layouts are not supported; extraction quality degrades on this format
- Job description skill extraction averages 2.81 skills per posting versus 5.91 per candidate profile — gap-analysis coverage percentages reflect this vocabulary asymmetry in the source job postings, not necessarily a true skill deficit in the candidate
- HR, Legal, Engineering, and Management domains are supported by the architecture but have only been validated on synthetic data, not real resumes (Data Science and IT have both)
- Flag-extraction keyword weights were calibrated on synthetic data and have lower precision on real free-text resumes — a clear post-launch recalibration priority once real candidate volume exists

---

## Author

**Sujal Deb** — [GitHub](https://github.com/sujaldeb) · [LinkedIn](https://linkedin.com/in/sujal-deb) · sujaldeb1@gmail.com
