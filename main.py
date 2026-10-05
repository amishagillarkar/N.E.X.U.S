"""
NEXUS AI
========
Navigated Execution & eXam Unified System.

An autonomous, agentic career-diagnostic and roadmap system for B.Tech
engineering students who feel overwhelmed by overlapping career paths
(Placements, GATE, CAT, Higher Studies).

What this service does
----------------------
1.  Audits a student's real skill inventory against the deterministic
    requirements of a target path.
2.  Returns a feasibility verdict (HIGH / MODERATE / CRITICAL_TIMELINE) backed
    by weighted coverage percentages and hour-level time estimates.
3.  Runs a multi-path trade-off / elimination matrix so the student can drop
    low-ROI goals *without* guilt or judgement.
4.  Expands every gap into a Basic -> Intermediate -> Advanced learning
    sequence (the simulated RAG syllabus retriever).
5.  Orchestrates all of the above through a LangChain tool-calling
    ReAct-style agent (``ChatOpenAI`` + ``create_tool_calling_agent`` +
    ``AgentExecutor``) wrapped in a FastAPI service.

Design principles
-----------------
*   The **tools are deterministic** (pure Python + curated knowledge base).
    The LLM never invents scores - it only reasons over tool output and
    narrates it empathetically.
*   Tools **never raise** on bad input. They return structured, readable
    error payloads so the agent stays in control and can self-correct.
*   The agent is created **lazily** so the service can boot, serve
    ``/health`` and run the pure-Python audit path even without an
    ``OPENAI_API_KEY`` configured.
*   Runs in offline / deterministic mode when the agent cannot be reached,
    which keeps local development and CI green.

Author : Principal Engineering
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

# ----------------------------------------------------------------- logging --
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("nexus.ai")

# Load .env before anything reads credentials.
load_dotenv(override=False)

# ---------------------------------------------------------- langchain / api --
from fastapi import FastAPI, HTTPException, Request, status  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field, field_validator, model_validator  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from key_manager import key_manager, mask_key  # noqa: E402
from langchain.agents import AgentExecutor, create_tool_calling_agent  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402


# ===========================================================================
# SECTION 1 — RUNTIME CONFIGURATION
# ===========================================================================

APP_NAME = "NEXUS AI"
APP_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

#: Calendar reality: ~4.345 weeks per month, ~52 weeks per year.
WEEKS_PER_MONTH = 4.345
MONTHS_PER_YEAR = 12

#: Hours a student can realistically dedicate per week. Used to convert an
#: hour estimate into a calendar timeline.
DEFAULT_WEEKLY_HOURS = 40

#: Above this utilisation the plan is considered to have no slack left.
BANDWIDTH_SATURATION = 0.88

#: Study-hour scaling factor derived from calendar time. A student who is
#: starting with a small runway is assumed to learn faster (less forgetting,
#: fewer parallel commitments), while a very long runway allows slow, relaxed
#: study. Bounded to a sane band so it never distorts the arithmetic.
MIN_STUDY_MULTIPLIER = 0.70
MAX_STUDY_MULTIPLIER = 1.30

#: Circular imports are impossible in Python; importing LangChain into a
#: module before ``@tool`` decorators run is fine, so the guard only exists to
#: produce a friendly message if the package is missing.
try:  # pragma: no cover - import-time environment concern
    import uvicorn  # noqa: F401

    _UVICORN_AVAILABLE = True
except ImportError:  # pragma: no cover
    uvicorn = None  # type: ignore[assignment]
    _UVICORN_AVAILABLE = False


# ===========================================================================
# SECTION 2 — THE KNOWLEDGE BASE
# ---------------------------------------------------------------------------
# Everything the "RAG" layer indexes. In production this would live in a
# vector store (FAISS / pgvector / Pinecone) built from university syllabi,
# NPTEL transcripts and PYQ archives. Here it is a curated, dependency-free
# corpus so the service is fully reproducible offline.
# ===========================================================================

#: Canonical career paths exposed by the system.
CANONICAL_PATHS: Tuple[str, ...] = (
    "PLACEMENTS",
    "GATE_CS",
    "GATE_DA",
    "CAT",
    "STUDY_ABROAD",
)

PATH_ALIASES: Dict[str, str] = {
    "PLACEMENTS": "PLACEMENTS",
    "PLACEMENT": "PLACEMENTS",
    "CAMPUS_PLACEMENT": "PLACEMENTS",
    "CAMPUS_PLACEMENTS": "PLACEMENTS",
    "SDE": "PLACEMENTS",
    "PRODUCT_COMPANIES": "PLACEMENTS",
    "TECH_PLACEMENTS": "PLACEMENTS",
    "JOB": "PLACEMENTS",
    "PLACEMENT_PREPARATION": "PLACEMENTS",
    "OFFCAMPUS": "PLACEMENTS",
    "OFF_CAMPUS": "PLACEMENTS",
    "GATE_CS": "GATE_CS",
    "GATE": "GATE_CS",
    "GATE_CSE": "GATE_CS",
    "GATE_CS_IT": "GATE_CS",
    "GATE_COMPUTER_SCIENCE": "GATE_CS",
    "GATE_COMPUTER_SCIENCE_AND_IT": "GATE_CS",
    "GATE_IT": "GATE_CS",
    "GATE_2025": "GATE_CS",
    "GATE_2026": "GATE_CS",
    "GATE_CSE_IT": "GATE_CS",
    "GATE_DA": "GATE_DA",
    "GATE_DS": "GATE_DA",
    "GATE_DATA_SCIENCE": "GATE_DA",
    "GATE_DATA_SCIENCE_AI": "GATE_DA",
    "GATE_DATA_SCIENCE_AND_AI": "GATE_DA",
    "GATE_DATASCIENCE": "GATE_DA",
    "CAT": "CAT",
    "IIM_CAT": "CAT",
    "CAT_MBA": "CAT",
    "CAT_XL": "CAT",
    "MBA_CAT": "CAT",
    "STUDY_ABROAD": "STUDY_ABROAD",
    "ABROAD": "STUDY_ABROAD",
    "MS_ABROAD": "STUDY_ABROAD",
    "MS_US": "STUDY_ABROAD",
    "US_MASTERS": "STUDY_ABROAD",
    "HIGHER_STUDIES": "STUDY_ABROAD",
    "HIGHER_STUDIES_ABROAD": "STUDY_ABROAD",
    "FOREIGN_MASTERS": "STUDY_ABROAD",
    "PHD_ABROAD": "STUDY_ABROAD",
}

#: Core syllabus topic registry.
#:
#: ``tier``       - BASIC | INTERMEDIATE | ADVANCED
#: ``hours``      - realistic study hours for a cold-start B.Tech student
#: ``importance`` - weight used for coverage scoring (2.0 = make-or-break)
#: ``prereqs``    - topic ids that should be locked first
TOPIC_CATALOG: Dict[str, Dict[str, Any]] = {
    "programming_basics": {
        "label": "Programming Fundamentals",
        "tier": "BASIC",
        "aliases": [
            "programming",
            "programming basics",
            "programming fundamentals",
            "basic programming",
            "coding",
            "coding basics",
            "basics of programming",
            "syntax",
            "python",
            "python basics",
            "python programming",
            "core python",
            "java",
            "java basics",
            "core java",
            "c language",
            "c basics",
            "c++",
            "c++ basics",
            "dsa coding",
        ],
        "hours": 140,
        "importance": 2.0,
        "prereqs": [],
    },
    "oops": {
        "label": "Object-Oriented Programming",
        "tier": "BASIC",
        "aliases": [
            "oops",
            "oop",
            "object oriented programming",
            "object-oriented programming",
            "object oriented concepts",
            "object orientation",
            "oop concepts",
            "classes and objects",
            "inheritance",
            "encapsulation",
            "abstraction polymorphism",
            "polymorphism",
        ],
        "hours": 70,
        "importance": 1.6,
        "prereqs": ["programming_basics"],
    },
    "maths": {
        "label": "Engineering Mathematics",
        "tier": "BASIC",
        "aliases": [
            "engineering mathematics",
            "engineering maths",
            "engineering math",
            "maths",
            "mathematics",
            "maths basics",
            "math",
            "discrete mathematics",
            "discrete maths",
            "discrete math",
            "engineering mathematics 1",
            "maths 1",
            "maths 2",
        ],
        "hours": 220,
        "importance": 1.8,
        "prereqs": [],
    },
    "calculus": {
        "label": "Calculus for Competitive Exams",
        "tier": "INTERMEDIATE",
        "aliases": [
            "calculus",
            "differentiation",
            "integration",
            "integrals",
            "derivatives",
            "mathematical analysis",
            "limits and continuity",
            "applications of derivatives",
        ],
        "hours": 120,
        "importance": 1.4,
        "prereqs": ["maths"],
    },
    "probability": {
        "label": "Probability & Combinatorics",
        "tier": "INTERMEDIATE",
        "aliases": [
            "probability",
            "probability theory",
            "bayes theorem",
            "random variables",
            "combinatorics",
            "permutations and combinations",
            "p and c",
            "discrete probability",
        ],
        "hours": 110,
        "importance": 1.5,
        "prereqs": ["maths"],
    },
    "linear_algebra": {
        "label": "Linear Algebra & Matrices",
        "tier": "INTERMEDIATE",
        "aliases": [
            "linear algebra",
            "matrices",
            "matrix algebra",
            "vectors and matrices",
            "eigenvalues and eigenvectors",
            "eigen values",
            "determinants",
            "rank of a matrix",
        ],
        "hours": 100,
        "importance": 1.4,
        "prereqs": ["maths"],
    },
    "dsa": {
        "label": "Data Structures & Algorithms",
        "tier": "INTERMEDIATE",
        "aliases": [
            "dsa",
            "data structures and algorithms",
            "data structures",
            "data structure",
            "algorithms",
            "algorithm",
            "arrays",
            "linked list",
            "linked lists",
            "stacks",
            "queues",
            "trees",
            "binary trees",
            "graphs",
            "graph theory",
            "dynamic programming",
            "dp",
            "recursion",
            "sorting",
            "searching",
            "hashing",
            "greedy algorithms",
            "problem solving",
            "leetcode",
            "competitive programming",
        ],
        "hours": 320,
        "importance": 2.0,
        "prereqs": ["programming_basics"],
    },
    "dbms": {
        "label": "Database Management Systems & SQL",
        "tier": "INTERMEDIATE",
        "aliases": [
            "dbms",
            "database management system",
            "database management systems",
            "rdbms",
            "databases",
            "sql",
            "sql queries",
            "normalization",
            "normal forms",
            "indexing",
            "transactions",
            "acid properties",
            "er diagram",
            "joins",
        ],
        "hours": 100,
        "importance": 1.5,
        "prereqs": [],
    },
    "operating_systems": {
        "label": "Operating Systems",
        "tier": "INTERMEDIATE",
        "aliases": [
            "operating system",
            "operating systems",
            "os",
            "os concepts",
            "processes and threads",
            "deadlock",
            "scheduling",
            "memory management",
            "page replacement",
            "concurrency",
            "file systems",
            "unix",
        ],
        "hours": 90,
        "importance": 1.3,
        "prereqs": [],
    },
    "computer_networks": {
        "label": "Computer Networks",
        "tier": "INTERMEDIATE",
        "aliases": [
            "computer network",
            "computer networks",
            "networking",
            "networks",
            "cn",
            "cn concepts",
            "tcp ip",
            "tcp/ip",
            "http https",
            "dns",
            "layered architecture",
            "osi model",
        ],
        "hours": 85,
        "importance": 1.2,
        "prereqs": [],
    },
    "computer_organization": {
        "label": "Computer Organization & Architecture",
        "tier": "INTERMEDIATE",
        "aliases": [
            "computer organization",
            "computer organisation",
            "computer architecture",
            "coa",
            "computer organization and architecture",
            "assembly language",
            "cpu architecture",
            "cache memory",
            "pipelining",
            "number representation",
        ],
        "hours": 85,
        "importance": 1.3,
        "prereqs": [],
    },
    "aptitude": {
        "label": "Quantitative Aptitude",
        "tier": "INTERMEDIATE",
        "aliases": [
            "aptitude",
            "quantitative aptitude",
            "quant",
            "quants",
            "logical reasoning",
            "reasoning",
            "puzzles",
            "verbal reasoning",
            "problem solving skills",
            "number system",
            "modern mathematics",
            "data sufficiency",
        ],
        "hours": 200,
        "importance": 1.9,
        "prereqs": [],
    },
    "verbal_ability": {
        "label": "Verbal Ability & Communication",
        "tier": "INTERMEDIATE",
        "aliases": [
            "verbal ability",
            "verbal",
            "verbal skills",
            "english",
            "english communication",
            "communication skills",
            "communication",
            "soft skills",
            "reading comprehension",
            "business english",
            "spoken english",
        ],
        "hours": 130,
        "importance": 1.5,
        "prereqs": [],
    },
    "dil": {
        "label": "Data Interpretation & Logical Reasoning",
        "tier": "ADVANCED",
        "aliases": [
            "dil",
            "data interpretation",
            "data interpretation and logical reasoning",
            "data interpretation & lr",
            "di and lr",
            "caselets",
            "set theory",
            "tables graphs charts",
            "arrangements",
            "seating arrangement",
        ],
        "hours": 180,
        "importance": 1.8,
        "prereqs": ["aptitude"],
    },
    "statistics": {
        "label": "Statistics for Data Science",
        "tier": "INTERMEDIATE",
        "aliases": [
            "statistics",
            "stats",
            "descriptive statistics",
            "inferential statistics",
            "probability and statistics",
            "hypothesis testing",
            "regression analysis",
            "bayesian statistics",
            "stats for machine learning",
        ],
        "hours": 140,
        "importance": 1.7,
        "prereqs": ["probability"],
    },
    "machine_learning": {
        "label": "Machine Learning",
        "tier": "INTERMEDIATE",
        "aliases": [
            "machine learning",
            "machine learning basics",
            "ml",
            "ml basics",
            "supervised learning",
            "unsupervised learning",
            "regression",
            "classification",
            "scikit learn",
            "sklearn",
            "feature engineering",
            "model evaluation",
        ],
        "hours": 240,
        "importance": 2.0,
        "prereqs": ["programming_basics", "linear_algebra", "statistics"],
    },
    "deep_learning": {
        "label": "Deep Learning",
        "tier": "ADVANCED",
        "aliases": [
            "deep learning",
            "deep learning basics",
            "dl",
            "neural networks",
            "neural network",
            "cnn",
            "rnn",
            "lstm",
            "transformers",
            "pytorch",
            "tensorflow",
            "keras",
        ],
        "hours": 220,
        "importance": 1.6,
        "prereqs": ["machine_learning"],
    },
    "system_design": {
        "label": "System Design",
        "tier": "ADVANCED",
        "aliases": [
            "system design",
            "distributed systems",
            "scalability",
            "high level design",
            "hld",
            "design patterns",
            "load balancing",
            "caching",
            "api design",
            "backend architecture",
        ],
        "hours": 160,
        "importance": 1.4,
        "prereqs": ["dsa", "dbms"],
    },
    "projects": {
        "label": "Projects & Portfolio",
        "tier": "ADVANCED",
        "aliases": [
            "projects",
            "project",
            "mini projects",
            "portfolio",
            "github projects",
            "personal projects",
            "capstone",
            "final year project",
            "fyp",
            "internships",
        ],
        "hours": 200,
        "importance": 1.8,
        "prereqs": [],
    },
    "git": {
        "label": "Git & Collaborative Workflow",
        "tier": "BASIC",
        "aliases": [
            "git",
            "github",
            "version control",
            "git and github",
            "gitlab",
            "bitbucket",
        ],
        "hours": 20,
        "importance": 1.0,
        "prereqs": [],
    },
    "essay_writing": {
        "label": "Essay & Structured Written Communication",
        "tier": "ADVANCED",
        "aliases": [
            "essay",
            "essay writing",
            "essay and writing",
            "writing",
            "english writing",
            "structured writing",
            "descriptive writing",
            "written communication",
            "gd topics",
            "essay topics",
        ],
        "hours": 90,
        "importance": 1.1,
        "prereqs": ["verbal_ability"],
    },
}

#: Helper aliases shared by every path blueprint.
_TIER_ORDER = {"BASIC": 0, "INTERMEDIATE": 1, "ADVANCED": 2}
_TIER_LABEL = {
    "BASIC": "BASIC (Fundamentals & Syntax)",
    "INTERMEDIATE": "INTERMEDIATE (Problem Solving & Core Concepts)",
    "ADVANCED": "ADVANCED (PYQs, Case Work & Mock Tests)",
}
_TIER_GOAL = {
    "BASIC": "Build an accurate mental model and stop making beginner errors. Output = notes + tiny runnable experiments.",
    "INTERMEDIATE": "Convert the mental model into speed and accuracy under time pressure. Output = graded problem sets + 1 applied mini-build.",
    "ADVANCED": "Simulate the real exam. Output = timed PYQ/mock blocks, error log, and revision sheets.",
}

PATH_PROFILES: Dict[str, Dict[str, Any]] = {
    "PLACEMENTS": {
        "label": "Product-company placements (SDE / Data / Backend roles)",
        "summary": (
            "Recruiter-led screening: aptitude + DSA speed, one solid project story, "
            "and CS fundamentals that survive cross-questions."
        ),
        "target_months": 8,
        "weekly_hours": 45,
        "cutoff_percentile": 70.0,
        "mandatory": [
            "programming_basics",
            "dsa",
            "dbms",
            "operating_systems",
            "computer_networks",
            "aptitude",
            "verbal_ability",
            "projects",
            "git",
            "oops",
        ],
        "recommended": ["computer_organization", "system_design", "probability"],
    },
    "GATE_CS": {
        "label": "GATE Computer Science & IT (PSU / M.Tech admission)",
        "summary": (
            "Single-exam, syllabus-wide, negative marking. Depth and breadth both "
            "matter - you cannot skip GATE's less glamorous core papers."
        ),
        "target_months": 12,
        "weekly_hours": 40,
        "cutoff_percentile": 65.0,
        "mandatory": [
            "programming_basics",
            "dsa",
            "dbms",
            "operating_systems",
            "computer_networks",
            "computer_organization",
            "maths",
            "calculus",
            "probability",
            "linear_algebra",
            "aptitude",
        ],
        "recommended": ["oops", "git", "projects"],
    },
    "GATE_DA": {
        "label": "GATE Data Science & AI (new paper)",
        "summary": (
            "A first-year-friendly GATE paper: 70% is pure maths/statistics and "
            "30% is ML + programming. Depth is achievable for most students.",
        ),
        "target_months": 9,
        "weekly_hours": 32,
        "cutoff_percentile": 45.0,
        "mandatory": [
            "programming_basics",
            "maths",
            "linear_algebra",
            "probability",
            "statistics",
            "machine_learning",
            "aptitude",
        ],
        "recommended": ["deep_learning", "programming_basics", "projects", "dsa"],
    },
    "CAT": {
        "label": "IIM CAT (MBA / MPGP admission)",
        "summary": (
            "A speed-and-composure test. VARC + DIL are the differentiators; "
            "quant decides the cut. Coaching-style repetition is the only lever."
        ),
        "target_months": 10,
        "weekly_hours": 45,
        "cutoff_percentile": 80.0,
        "mandatory": [
            "aptitude",
            "verbal_ability",
            "dil",
            "calculus",
            "probability",
            "maths",
        ],
        "recommended": ["programming_basics", "statistics", "essay_writing"],
    },
    "STUDY_ABROAD": {
        "label": "Higher studies abroad (MS / PhD applications)",
        "summary": (
            "Not an exam - a portfolio. Output is proof of capability plus a "
            "credible narrative, so project depth beats problem-count here."
        ),
        "target_months": 12,
        "weekly_hours": 30,
        "cutoff_percentile": 60.0,
        "mandatory": [
            "programming_basics",
            "projects",
            "verbal_ability",
            "git",
            "dsa",
            "machine_learning",
        ],
        "recommended": [
            "statistics",
            "deep_learning",
            "system_design",
            "probability",
        ],
    },
}

#: Structural conflict between paths (context-switching + zero skill transfer).
#: Weighted by how many paths are being pursued simultaneously.
PATH_CONFLICTS: Dict[Tuple[str, ...], float] = {
    ("GATE_CS", "GATE_DA"): 0.30,
    ("CAT", "GATE_CS", "GATE_DA"): 0.22,
    ("CAT", "GATE_CS"): 0.10,
    ("GATE_CS", "PLACEMENTS"): 0.18,
    ("GATE_DA", "PLACEMENTS"): 0.20,
    ("CAT", "GATE_DA"): 0.12,
    ("CAT", "PLACEMENTS"): 0.14,
    ("PLACEMENTS", "STUDY_ABROAD"): 0.10,
    ("CAT", "STUDY_ABROAD"): 0.12,
    ("GATE_DA", "STUDY_ABROAD"): 0.10,
    ("GATE_CS", "STUDY_ABROAD"): 0.10,
}

PATH_SYNERGY: Dict[Tuple[str, str], str] = {
    ("GATE_CS", "PLACEMENTS"): (
        "GATE's OS/CN/COA/DBMS fundamentals are exactly the cross-questions "
        "interviewers ask after a DSA round."
    ),
    ("GATE_CS", "STUDY_ABROAD"): (
        "A strong GATE score plus one research-grade project covers most "
        "admission requirements."
    ),
    ("GATE_DA", "STUDY_ABROAD"): (
        "ML fundamentals map directly onto the research area you apply with."
    ),
    ("PLACEMENTS", "STUDY_ABROAD"): (
        "Strong placement grades become your strongest recommendation letter "
        "and funding narrative."
    ),
}


# ===========================================================================
# SECTION 3 — DETERMINISTIC DOMAIN LOGIC (shared by tools + fallback engine)
# ===========================================================================

_WHITESPACE_RE = re.compile(r"[\s_\-/\\.]+")


def normalize_token(value: str) -> str:
    """Lowercase, strip punctuation-ish separators and collapse whitespace.

    Args:
        value: Raw user / LLM supplied token.

    Returns:
        A canonical comparable string, e.g. ``"Data Structures & Algorithms"``
        becomes ``"data structures algorithms"``.
    """
    text = _WHITESPACE_RE.sub(" ", str(value or "").strip().lower())
    text = re.sub(r"[+]+", " plus ", text)
    text = re.sub(r"[^a-z0-9+# ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_words(value: str) -> List[str]:
    """Split a normalized token into word tokens.

    Args:
        value: Raw token.

    Returns:
        List of lowercase word tokens (may be empty).
    """
    normalized = normalize_token(value)
    return [w for w in normalized.split(" ") if w]


def _build_path_alias_lookup() -> Dict[str, str]:
    """Normalise every path alias key into a lookup table.

    Alias keys are authored in SCREAMING_SNAKE (``"GATE_CS"``) while lookups are
    normalised lowercase, so a naive dict lookup would silently miss. This
    builder registers both the underscored and spaced forms.

    Returns:
        Dict of normalized alias -> canonical path id.
    """
    lookup: Dict[str, str] = {}
    for alias, canonical in PATH_ALIASES.items():
        for key in (alias, alias.replace("_", " ")):
            normalized = normalize_token(key)
            if normalized:
                lookup[normalized] = canonical
                lookup[normalized.replace(" ", "_")] = canonical
    return lookup


def _build_conflict_lookup() -> Dict[Tuple[str, ...], float]:
    """Normalise conflict signatures into order-independent sorted tuples.

    Returns:
        Dict of sorted path-tuple -> overhead fraction.
    """
    return {tuple(sorted(key)): value for key, value in PATH_CONFLICTS.items()}


def _build_synergy_lookup() -> Dict[Tuple[str, str], str]:
    """Normalise synergy pair keys into order-independent sorted tuples.

    Returns:
        Dict of sorted path-pair -> synergy note.
    """
    return {tuple(sorted(key)): value for key, value in PATH_SYNERGY.items()}


PATH_ALIAS_LOOKUP: Dict[str, str] = _build_path_alias_lookup()
CONFLICT_LOOKUP: Dict[Tuple[str, ...], float] = _build_conflict_lookup()
SYNERGY_LOOKUP: Dict[Tuple[str, str], str] = _build_synergy_lookup()


_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "of",
    "in",
    "for",
    "to",
    "basic",
    "basics",
    "fundamentals",
    "concepts",
    "knowledge",
    "advanced",
    "intermediate",
    "learned",
    "learning",
    "studied",
    "study",
    "score",
    "good",
    "average",
    "basic understanding",
    "some",
    "bit",
    "little",
}


def resolve_path(path_type: str) -> Optional[str]:
    """Map any user-supplied path string to a canonical path id.

    Handles casing, spacing, punctuation and a large alias table
    (``"gate cse"``, ``"GATE-CS"``, ``"iim cat"``, ``"masters abroad"`` ...).

    Args:
        path_type: Raw path string from the student or the LLM.

    Returns:
        Canonical path id (one of :data:`CANONICAL_PATHS`) or ``None`` when the
        path is not supported.
    """
    if not path_type:
        return None
    normalized = normalize_token(path_type)
    if not normalized:
        return None
    if normalized in PATH_ALIAS_LOOKUP:
        return PATH_ALIAS_LOOKUP[normalized]
    squashed = normalized.replace(" ", "_")
    if squashed in PATH_ALIAS_LOOKUP:
        return PATH_ALIAS_LOOKUP[squashed]

    words = token_words(path_type)
    joined = "".join(words)

    # Heuristic family detection, e.g. "gate data science ai 2026".
    if "gate" in words:
        if any(w in words for w in ("data", "ds", "ai", "analytics")):
            return "GATE_DA"
        if any(
            w in words
            for w in ("cs", "cse", "computer", "it", "core", "ece", "me")
        ) or "computerscience" in joined:
            return "GATE_CS"
        return "GATE_CS"
    if "cat" in words or "mba" in words:
        return "CAT"
    if any(w in words for w in ("abroad", "foreign", "overseas")):
        return "STUDY_ABROAD"
    if any(
        w in words for w in ("placement", "placements", "sde", "job", "jobs", "hiring")
    ):
        return "PLACEMENTS"
    return None


def _build_topic_index() -> Dict[str, str]:
    """Inverted index from alias phrase -> topic id (phrase-length ordered)."""
    index: Dict[str, str] = {}
    for topic_id, meta in TOPIC_CATALOG.items():
        keys = {topic_id, topic_id.replace("_", " "), meta["label"]}
        keys.update(meta["aliases"])
        for key in keys:
            index.setdefault(normalize_token(key), topic_id)
    return index


TOPIC_INDEX: Dict[str, str] = _build_topic_index()


def resolve_topic(raw_topic: str) -> Optional[str]:
    """Resolve a free-form skill / gap name to a canonical topic id.

    Resolution strategy (first hit wins):
        1. Exact normalized match against the catalog label / id / aliases.
        2. Highest keyword-overlap score across the catalog.
        3. Word-level containment against any alias.
        4. Character-similarity (typo-tolerant) match.

    Args:
        raw_topic: Free-form skill or gap text, e.g. ``"recurison"`` or
            ``"I know a bit of DSA and DBMS"``.

    Returns:
        Canonical topic id or ``None``.
    """
    if not raw_topic:
        return None
    normalized = normalize_token(raw_topic)
    if not normalized:
        return None
    if normalized in TOPIC_INDEX:
        return TOPIC_INDEX[normalized]

    # Keyword-overlap scoring.
    raw_words = [w for w in token_words(raw_topic) if w not in _STOPWORDS]
    if raw_words:
        best_topic: Optional[str] = None
        best_score = 0.0
        for topic_id, meta in TOPIC_CATALOG.items():
            alias_words: set[str] = set()
            for alias in list(meta["aliases"]) + [meta["label"], topic_id]:
                alias_words.update(w for w in token_words(alias) if w not in _STOPWORDS)
            if not alias_words:
                continue
            score = len(set(raw_words) & alias_words) / len(alias_words)
            if score > best_score:
                best_topic, best_score = topic_id, score
        if best_topic and best_score >= 0.34:
            return best_topic

    # Word-level containment fallback ("sql joins" contains "sql"). Matching is
    # done on whole tokens, never on raw substrings, so "quantum" must not be
    # read as the alias "quant".
    input_words = set(token_words(raw_topic))
    if input_words:
        for topic_id, meta in TOPIC_CATALOG.items():
            for alias in list(meta["aliases"]) + [meta["label"]]:
                alias_words = {w for w in token_words(alias) if w not in _STOPWORDS}
                if alias_words and alias_words <= input_words:
                    return topic_id

    # Typo-tolerant fallback ("recurison" -> "recursion").
    best_topic, best_ratio = _fuzzy_topic_match(normalized)
    if best_topic:
        logger.debug("Fuzzy topic match %r -> %s (%.2f)", raw_topic, best_topic, best_ratio)
    return best_topic


def _fuzzy_topic_match(normalized: str, threshold: float = 0.82) -> Tuple[Optional[str], float]:
    """Character-similarity match against every catalog alias.

    Handles typos and transliterations that token overlap cannot catch.

    Args:
        normalized: Normalized query string.
        threshold: Minimum similarity ratio to accept a match.

    Returns:
        ``(topic_id, ratio)`` or ``(None, 0.0)`` when nothing clears the bar.
    """
    if len(normalized) < 4:
        return None, 0.0
    best_topic: Optional[str] = None
    best_ratio = 0.0
    for topic_id, meta in TOPIC_CATALOG.items():
        candidates = [topic_id, meta["label"]] + list(meta["aliases"])
        for candidate in candidates:
            candidate_norm = normalize_token(candidate)
            if len(candidate_norm) < 4:
                continue
            ratio = SequenceMatcher(None, normalized, candidate_norm).ratio()
            if ratio > best_ratio:
                best_topic, best_ratio = topic_id, ratio
    if best_ratio >= threshold:
        return best_topic, best_ratio
    return None, best_ratio


def clean_skills(known_skills: Optional[Sequence[str]]) -> Tuple[List[str], Dict[str, str]]:
    """Normalise a raw skill list into canonical topic ids plus evidence.

    Every declared skill is kept (even unknown ones) so the student can see
    their own words reflected back; ``matched`` maps only resolved ones.

    Args:
        known_skills: Raw skill strings from the payload / LLM tool call.

    Returns:
        ``(unique_raw_skills, matched_map)`` where ``matched_map`` is
        ``{"sql": "dbms", "dsa": "dsa"}`` style evidence for explainability.
    """
    unique: List[str] = []
    seen: set[str] = set()
    matched: Dict[str, str] = {}
    for skill in known_skills or []:
        if not isinstance(skill, str):
            skill = str(skill)
        cleaned = skill.strip()
        if not cleaned:
            continue
        key = normalize_token(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
        topic_id = resolve_topic(cleaned)
        if topic_id:
            matched[key] = topic_id
    return unique, matched


def study_multiplier(months_available: float) -> float:
    """Translate calendar runway into a study-intensity multiplier.

    A student starting today studies differently from one starting in a year:
    short runway implies compressed, near-full-time effort, long runway implies
    slower, sustainable effort. Bounded to
    ``[MIN_STUDY_MULTIPLIER, MAX_STUDY_MULTIPLIER]``.

    Args:
        months_available: Positive number of months before the target date.

    Returns:
        Multiplier applied to every catalogue hour estimate.
    """
    if months_available <= 0:
        return MAX_STUDY_MULTIPLIER
    raw = math_sqrt(6.0 / months_available)
    return max(MIN_STUDY_MULTIPLIER, min(MAX_STUDY_MULTIPLIER, raw))


def math_sqrt(value: float) -> float:
    """Small dependency-free square root.

    Args:
        value: Non-negative float.

    Returns:
        Square root of ``value``.
    """
    if value <= 0:
        return 0.0
    guess = value if value < 1 else value / 2.0
    for _ in range(30):
        guess = 0.5 * (guess + value / guess)
    return guess


def path_topic_ids(path_id: str) -> List[str]:
    """Mandatory + recommended topic ids for a path (deduplicated).

    Args:
        path_id: Canonical path id.

    Returns:
        Ordered list of topic ids.
    """
    profile = PATH_PROFILES[path_id]
    ordered: List[str] = []
    for topic_id in list(profile["mandatory"]) + list(profile["recommended"]):
        if topic_id not in ordered:
            ordered.append(topic_id)
    return ordered


def rank_paths(
    path_stats: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Rank candidate paths using one shared, deterministic ordering.

    This is the single source of truth for "which goal survives", used by the
    elimination-matrix tool, the API endpoint and the offline engine so they can
    never disagree. Ordering, in priority order:

        1. Feasibility flag (HIGH -> MODERATE -> CRITICAL_TIMELINE).
        2. Weighted coverage already held, descending - never discard skills
           the student already has.
        3. Exam cutoff percentile, descending - a harder paper is a better use
           of the same hours.

    Args:
        path_stats: ``{path_id: {"feasibility": str, "weighted_coverage_percent": float}}``.

    Returns:
        Path ids ordered best-first.
    """
    order = {"HIGH": 0, "MODERATE": 1, "CRITICAL_TIMELINE": 2}
    return sorted(
        path_stats,
        key=lambda pid: (
            order.get(path_stats[pid].get("feasibility", "CRITICAL_TIMELINE"), 2),
            -float(path_stats[pid].get("weighted_coverage_percent", 0.0)),
            -float(PATH_PROFILES[pid]["cutoff_percentile"]),
            pid,
        ),
    )


def _missing_prerequisite_chain(topic_id: str, matched: set[str]) -> List[str]:
    """Return the missing ancestors of ``topic_id`` (depth-first, de-duplicated).

    Args:
        topic_id: Topic whose prerequisites we want.
        matched: Already-possessed topic ids.

    Returns:
        Ordered list of missing prerequisite topic ids.
    """
    chain: List[str] = []
    visited: set[str] = set()

    def walk(current: str) -> None:
        for prereq in TOPIC_CATALOG.get(current, {}).get("prereqs", []):
            if prereq in matched or prereq in visited:
                continue
            visited.add(prereq)
            walk(prereq)
            chain.append(prereq)

    walk(topic_id)
    return chain


def estimate_gap(topic_ids: Sequence[str], matched: set[str], multiplier: float) -> Dict[str, Any]:
    """Compute weighted coverage and hour-level remediation effort for a path.

    Args:
        topic_ids: Required topics (mandatory + recommended).
        matched: Topic ids the student already has.
        multiplier: Calendar-driven study multiplier from :func:`study_multiplier`.

    Returns:
        Dict with keys:
        ``coverage_percent``, ``weighted_coverage_percent``,
        ``matched_topics``, ``missing_topics``, ``missing_prerequisites``,
        ``raw_hours``, ``estimated_hours``, ``weeks_required``, ``months_required``,
        ``tier_hours`` and ``blocking_topics``.
    """
    matched = set(matched)
    matched_ids = [t for t in topic_ids if t in matched]
    missing_ids = [t for t in topic_ids if t not in matched]

    total_weight = sum(TOPIC_CATALOG[t]["importance"] for t in topic_ids)
    earned_weight = sum(TOPIC_CATALOG[t]["importance"] for t in matched_ids)
    coverage = 100.0 * len(matched_ids) / len(topic_ids) if topic_ids else 0.0
    weighted_coverage = 100.0 * earned_weight / total_weight if total_weight else 0.0

    # Missing prerequisites of every missing topic (the true blocking set).
    prereq_ids: List[str] = []
    for topic_id in missing_ids:
        for prereq in _missing_prerequisite_chain(topic_id, matched):
            if prereq not in prereq_ids:
                prereq_ids.append(prereq)

    tier_hours: Dict[str, float] = {"BASIC": 0.0, "INTERMEDIATE": 0.0, "ADVANCED": 0.0}
    for topic_id in missing_ids:
        tier = TOPIC_CATALOG[topic_id]["tier"]
        tier_hours[tier] += TOPIC_CATALOG[topic_id]["hours"] * multiplier

    raw_hours = sum(TOPIC_CATALOG[t]["hours"] for t in missing_ids)
    estimated_hours = raw_hours * multiplier
    weeks_required = estimated_hours / DEFAULT_WEEKLY_HOURS if DEFAULT_WEEKLY_HOURS else 0.0
    months_required = estimated_hours / (DEFAULT_WEEKLY_HOURS * WEEKS_PER_MONTH)

    blocking = [
        t for t in prereq_ids if TOPIC_CATALOG[t]["importance"] >= 1.5
    ] or prereq_ids

    return {
        "coverage_percent": round(coverage, 1),
        "weighted_coverage_percent": round(weighted_coverage, 1),
        "matched_topics": matched_ids,
        "missing_topics": missing_ids,
        "missing_prerequisites": prereq_ids,
        "raw_hours": round(raw_hours, 1),
        "estimated_hours": round(estimated_hours, 1),
        "weeks_required": round(weeks_required, 1),
        "months_required": round(months_required, 1),
        "tier_hours": {k: round(v, 1) for k, v in tier_hours.items()},
        "blocking_topics": blocking,
    }


def compute_feasibility(
    months_required: float,
    months_available: float,
    weighted_coverage_percent: float,
) -> str:
    """Assign the deterministic feasibility flag for a path.

    Rules (purely arithmetic - no opinion involved):
        * ``HIGH``               - months_required <= months_available * 0.70
        * ``MODERATE``           - months_required <= months_available * 1.00
        * ``CRITICAL_TIMELINE``  - everything else

    Args:
        months_required: Time needed to close every gap.
        months_available: Time the student actually has.
        weighted_coverage_percent: Weighted syllabus coverage already held. Kept
            for reporting and for callers that want a coverage-aware verdict;
            it deliberately does not gate the flag, so a long runway for a
            zero-skill student still resolves to ``HIGH``.

    Returns:
        One of ``HIGH``, ``MODERATE`` or ``CRITICAL_TIMELINE``.
    """
    del weighted_coverage_percent  # documented, intentionally not gating
    if months_available <= 0:
        return "CRITICAL_TIMELINE"
    ratio = months_required / months_available
    if ratio <= 0.70:
        return "HIGH"
    if ratio <= 1.0:
        return "MODERATE"
    return "CRITICAL_TIMELINE"


def build_topic_blueprint(topic_id: str, path_id: Optional[str] = None) -> Dict[str, Any]:
    """Build the 3-tier learning sequence for one topic.

    Args:
        topic_id: Canonical topic id.
        path_id: Optional canonical path id used to add exam-specific detail.

    Returns:
        Dict with ``topic``, ``label``, ``path_type`` and a ``sequence`` list of
        three tier dicts (name, objective, hours, modules, exit_criteria,
        resources).
    """
    meta = TOPIC_CATALOG[topic_id]
    multiplier = 1.0
    hours = float(meta["hours"])
    prereqs = [TOPIC_CATALOG[p]["label"] for p in meta.get("prereqs", [])]

    basic_h = round(hours * 0.30 * multiplier, 1)
    inter_h = round(hours * 0.45 * multiplier, 1)
    adv_h = round(hours * 0.25 * multiplier, 1)

    if meta["tier"] == "BASIC":
        basic_h = round(hours * 0.50 * multiplier, 1)
        inter_h = round(hours * 0.35 * multiplier, 1)
        adv_h = round(hours * 0.15 * multiplier, 1)

    path_hint = {
        "PLACEMENTS": "Recruiter-tested: expect this in DSA screens and cross-questions.",
        "GATE_CS": "GATE-pyq-heavy: single-correct MCQs punish shallow reading.",
        "GATE_DA": "GATE-DA-pyq-heavy: maths carries the majority of the marks.",
        "CAT": "CAT-style: speed under a 3-hour clock matters more than coverage.",
        "STUDY_ABROAD": "Portfolio framing: turn it into a project + written artefact.",
    }.get(path_id or "", "")

    sequence = [
        {
            "tier": "BASIC",
            "tier_name": _TIER_LABEL["BASIC"],
            "objective": _TIER_GOAL["BASIC"],
            "estimated_hours": basic_h,
            "weeks": max(1, round(basic_h / (DEFAULT_WEEKLY_HOURS * 0.4), 1)),
            "modules": [
                f"Read one authoritative source end-to-end for {meta['label']} (NCERT / NPTEL / Swayam).",
                "Rewrite every definition from memory, then verify against the source.",
                f"Build 5 tiny runnable experiments or handwritten drills around {meta['label']}.",
                "Keep a one-page 'cheat sheet' - this is your revision asset, not notes.",
            ],
            "exit_criteria": (
                "You can explain the core idea of each concept aloud in under 60 seconds "
                "without notes, and you have zero unresolved doubts in the cheat sheet."
            ),
            "resources": _resources_for(topic_id, "BASIC"),
        },
        {
            "tier": "INTERMEDIATE",
            "tier_name": _TIER_LABEL["INTERMEDIATE"],
            "objective": _TIER_GOAL["INTERMEDIATE"],
            "estimated_hours": inter_h,
            "weeks": max(2, round(inter_h / (DEFAULT_WEEKLY_HOURS * 0.6), 1)),
            "modules": [
                f"Solve graded problem sets for {meta['label']} (LeetCode / GFG / GATE PYQ packs).",
                "Time-box every problem set; accuracy under a timer is the real skill.",
                "Write one applied mini-build that uses the concept end-to-end.",
                "Maintain an error log: what you missed, why, and the corrected rule.",
            ],
            "exit_criteria": (
                "70%+ accuracy on a timed set of 25 problems, and you can defend your "
                "approaches in a mock interview / viva without rehearsal."
            ),
            "resources": _resources_for(topic_id, "INTERMEDIATE"),
        },
        {
            "tier": "ADVANCED",
            "tier_name": _TIER_LABEL["ADVANCED"],
            "objective": _TIER_GOAL["ADVANCED"],
            "estimated_hours": adv_h,
            "weeks": max(2, round(adv_h / (DEFAULT_WEEKLY_HOURS * 0.8), 1)),
            "modules": [
                f"Work through the previous-year question archive for {meta['label']} and tag every miss by topic.",
                "Sit full-length timed mocks (180 minutes) under real exam conditions.",
                "Solve 2 starred/numerical questions daily - these decide the rank.",
                "Close the loop: rewrite your error log into a final revision sheet.",
            ],
            "exit_criteria": (
                f"Mock score consistently above the {PATH_PROFILES.get(path_id or '', {}).get('cutoff_percentile', 70.0)}% "
                f"target percentile for {path_id or 'your exam'} across 3 consecutive mocks."
            ),
            "resources": _resources_for(topic_id, "ADVANCED"),
        },
    ]

    return {
        "topic": topic_id,
        "label": meta["label"],
        "catalog_tier": meta["tier"],
        "path_type": path_id,
        "path_hint": path_hint,
        "prerequisites": prereqs,
        "total_hours": round(basic_h + inter_h + adv_h, 1),
        "sequence": sequence,
    }


def _resources_for(topic_id: str, tier: str) -> List[str]:
    """Return curated, free-first resources for a topic tier.

    Args:
        topic_id: Canonical topic id.
        tier: ``BASIC`` | ``INTERMEDIATE`` | ``ADVANCED``.

    Returns:
        List of resource descriptors (resource type + purpose).
    """
    library: Dict[str, Dict[str, List[str]]] = {
        "programming_basics": {
            "BASIC": ["CS50P (CS50) or NPTEL 'Programming, Data Structures & Algorithms'"],
            "INTERMEDIATE": ["Python Tutor (visual tracing)", "Automate the Boring Stuff - exercises"],
            "ADVANCED": ["Refactor Guru (design patterns in code)", "Your own mini-project README"],
        },
        "dsa": {
            "BASIC": ["NPTEL DSA course", "Big-O cheat sheet from the course notes"],
            "INTERMEDIATE": ["LeetCode Easy/Medium by tag", "Striver's A2Z sheet (topicwise)"],
            "ADVANCED": ["Codeforces Div. 2 A-C", "Company-specific curated problem lists"],
        },
        "aptitude": {
            "BASIC": ["RS Aggarwal - Quant & Verbal (basics section)", "Indixxx free aptitude drills"],
            "INTERMEDIATE": ["Topicwise PYQ sets with a 45-second per-question timer"],
            "ADVANCED": ["Oliveboard / CareerRide timed sections", "Previous-year CAT/GATE quant papers"],
        },
        "verbal_ability": {
            "BASIC": ["High-school English grammar revision", "Daily 1 newspaper editorial read-aloud"],
            "INTERMEDIATE": ["RC passage sets with a 7-minute per-passage cap"],
            "ADVANCED": ["VARC past-year papers, one full section per session"],
        },
        "dil": {
            "BASIC": ["Set/table/case-let fundamentals", "Logical-reasoning grid practice"],
            "INTERMEDIATE": ["DI/LR sets with a 2-minute-per-question cap"],
            "ADVANCED": ["Full CAT VARC + DIL section under 65 minutes"],
        },
        "maths": {
            "BASIC": ["NPTEL Engineering Mathematics", "Your university Math-I / Math-II notes"],
            "INTERMEDIATE": ["Formula sheet rebuilt from memory weekly", "GATE/ CAT quant PYQ sets"],
            "ADVANCED": ["Numerical answer practice - 20 per session"],
        },
        "calculus": {
            "BASIC": ["Engineering Mathematics - differentiation/integration chapters"],
            "INTERMEDIATE": ["Applied-derivative and integral PYQ sets"],
            "ADVANCED": ["Numerical problems on limits, maxima-minima, areas"],
        },
        "probability": {
            "BASIC": ["Conditional probability + Bayes drills", "Permutation-combination PYQs"],
            "INTERMEDIATE": ["Random-variable and distribution problems", "GATE DA statistics section PYQs"],
            "ADVANCED": ["Multi-set conditional-probability questions (CAT pattern)"],
        },
        "linear_algebra": {
            "BASIC": ["Matrices, determinants, rank - concept + drill sheet"],
            "INTERMEDIATE": ["Eigenvalues/eigenvectors + orthogonal transformation problems"],
            "ADVANCED": ["ML-flavored applications: PCA, SVD intuition problems"],
        },
        "statistics": {
            "BASIC": ["Descriptive statistics + probability recap"],
            "INTERMEDIATE": ["Hypothesis testing, regression, confidence intervals"],
            "ADVANCED": ["Bayesian inference + full GATE-DA stats section"],
        },
        "machine_learning": {
            "BASIC": ["NPTEL / Andrew Ng ML specialization", "scikit-learn 'fit-predict' drills"],
            "INTERMEDIATE": ["Implement 6 algorithms from scratch (numpy)", "Kaggle tabular starter notebooks"],
            "ADVANCED": ["Feature engineering + model evaluation case study", "Reproduce one paper's result"],
        },
        "deep_learning": {
            "BASIC": ["Fast.ai Practical Deep Learning for Coders"],
            "INTERMEDIATE": ["PyTorch from-zero CNN/RNN implementations"],
            "ADVANCED": ["Attention/transformers from scratch + GATE-DA deep-learning PYQs"],
        },
        "dbms": {
            "BASIC": ["SQL from scratch (DDL/DML/joins)", "3NF normalisation drills"],
            "INTERMEDIATE": ["Indexing, transactions & ACID problems", "Solve 50 LeetCode DB questions"],
            "ADVANCED": ["Query-optimisation case questions", "Consistency/index design trade-offs"],
        },
        "operating_systems": {
            "BASIC": ["NPTEL Operating Systems", "Process/thread/memory cheat sheet"],
            "INTERMEDIATE": ["Scheduling + deadlock + page-replacement numericals"],
            "ADVANCED": ["Full GATE OS section in one sitting", "Consistency questions from past papers"],
        },
        "computer_networks": {
            "BASIC": ["Top-down layered-architecture walkthrough"],
            "INTERMEDIATE": ["TCP handshake, congestion control, and subnetting problems"],
            "ADVANCED": ["Full GATE CN section + interview cross-questions on protocols"],
        },
        "computer_organization": {
            "BASIC": ["Number systems, logic gates, memory hierarchy"],
            "INTERMEDIATE": ["Pipelining, cache, addressing numericals"],
            "ADVANCED": ["Full GATE COA section", "CPU design trade-off questions"],
        },
        "system_design": {
            "BASIC": ["Gaurav Sen / Alex Xu-style fundamentals"],
            "INTERMEDIATE": ["Design Instagram/URL-shortener/chat from scratch on paper"],
            "ADVANCED": ["Load balancing, sharding, and consistency trade-offs"],
        },
        "projects": {
            "BASIC": ["Pick 2 projects that solve a real, visible problem"],
            "INTERMEDIATE": ["Ship, document, and get one external code review"],
            "ADVANCED": ["Write a case study: problem, trade-offs, metrics, and what you'd improve"],
        },
        "git": {
            "BASIC": ["Git basics + branching/rebasing tutorial"],
            "INTERMEDIATE": ["Open-source contribution (docs/code)"],
            "ADVANCED": ["Clean repository README + CI pipeline"],
        },
        "essay_writing": {
            "BASIC": ["Structuring an argument: claim, evidence, counter, close"],
            "INTERMEDIATE": ["2 timed 15-minute essays per week with a review checklist"],
            "ADVANCED": ["Past essay prompts + business-article summarisation practice"],
        },
        "oops": {
            "BASIC": ["The 4 pillars + SOLID, with code examples"],
            "INTERMEDIATE": ["Refactor 3 procedural programs into OOP designs"],
            "ADVANCED": ["Design-pattern trade-off questions for interviews"],
        },
    }
    return library.get(
        topic_id,
        {
            "BASIC": ["NPTEL / Swayam course for this topic", "Rewritten one-page cheat sheet"],
            "INTERMEDIATE": ["Topicwise problem set with a timer"],
            "ADVANCED": ["Previous-year question archive + timed mocks"],
        },
    )[tier]


def _tool_error(tool: str, reason: str, detail: str, **extra: Any) -> str:
    """Serialise a structured tool error as a JSON string.

    Args:
        tool: Name of the failing tool.
        reason: Machine-readable error code.
        detail: Human-readable explanation.
        **extra: Additional structured context.

    Returns:
        JSON string safe for the LLM to read.
    """
    payload = {
        "tool": tool,
        "status": "ERROR",
        "error_code": reason,
        "message": detail,
        "supported_paths": list(CANONICAL_PATHS),
        "supported_topics": [TOPIC_CATALOG[t]["label"] for t in TOPIC_CATALOG],
        "recovery_hint": (
            "Do not guess. Re-read the supported lists above, correct the "
            "arguments, and call the tool again."
        ),
    }
    payload.update(extra)
    return json.dumps(payload, indent=2, sort_keys=False)


# ===========================================================================
# SECTION 4 — LANGCHAIN TOOLS
# ===========================================================================


@tool
def audit_skill_inventory(
    known_skills: List[str],
    target_path: str,
    months_available: int,
) -> str:
    """Audit a student's real skill inventory against one target career path.

    Call this FIRST in every conversation. It is the only source of truth for
    feasibility - never estimate coverage yourself.

    What it does:
        1. Normalises free-form skills onto the canonical syllabus catalog.
        2. Maps them onto the mandatory + recommended requirements of
           ``target_path``.
        3. Computes unweighted and importance-weighted coverage percentages.
        4. Surfaces missing *foundational prerequisites* (the blocking chain, not
           just the surface topic list).
        5. Estimates hours -> weeks -> months at a realistic weekly load.
        6. Assigns a deterministic feasibility flag.

    Args:
        known_skills: Everything the student can currently do, in any wording.
            Empty list means "starting from zero" and is valid.
        target_path: One of ``PLACEMENTS``, ``GATE_CS``, ``GATE_DA``, ``CAT``,
            ``STUDY_ABROAD``. Common aliases ("gate cse", "iim cat", "masters
            abroad", "SDE") are resolved automatically.
        months_available: Whole months of runway until the target date. Use
            ``0`` only when the deadline is already upon them.

    Returns:
        A JSON string containing:
        ``target_path``, ``path_label``, ``coverage_percent``,
        ``weighted_coverage_percent``, ``matched_skills``, ``missing_skills``,
        ``missing_prerequisites``, ``blocking_topics``, ``estimated_hours``,
        ``weeks_required``, ``months_required``, ``months_available``,
        ``buffer_months``, ``feasibility``, ``tier_hours``, ``mandatory_gap``
        and ``recommendation``.
    """
    months = 0 if months_available is None else max(0, int(months_available))

    path_id = resolve_path(target_path)
    if not path_id:
        return _tool_error(
            "audit_skill_inventory",
            "UNMAPPED_PATH",
            f"'{target_path}' is not a supported target path.",
            received_path=target_path,
        )

    if months <= 0:
        # Zero runway is a legitimate state, not an input error.
        months = 0

    raw_skills, matched_map = clean_skills(known_skills)
    matched_topics = set(matched_map.values())

    required_topics = path_topic_ids(path_id)
    mandatory_topics = list(PATH_PROFILES[path_id]["mandatory"])

    gap = estimate_gap(required_topics, matched_topics, study_multiplier(months))

    # Feasibility is judged on the *mandatory* spine of the syllabus - optional
    # gaps can always be cut without breaking the plan.
    mandatory_gap = estimate_gap(
        mandatory_topics, matched_topics, study_multiplier(months)
    )
    feasibility = compute_feasibility(
        mandatory_gap["months_required"], months, gap["weighted_coverage_percent"]
    )

    matched_labels = [
        TOPIC_CATALOG[t]["label"] for t in gap["matched_topics"]
    ]
    missing_labels = [TOPIC_CATALOG[t]["label"] for t in gap["missing_topics"]]
    missing_prereq_labels = [
        TOPIC_CATALOG[t]["label"] for t in gap["missing_prerequisites"]
    ]
    blocking_labels = [TOPIC_CATALOG[t]["label"] for t in gap["blocking_topics"]]

    profile = PATH_PROFILES[path_id]
    cutoff = float(profile["cutoff_percentile"])
    buffer_months = round(months - gap["months_required"], 1)

    if feasibility == "HIGH":
        recommendation = (
            f"Commit to {profile['label']}. You have {buffer_months} months of "
            f"surplus - spend it on PYQs, mocks and a strong project narrative "
            f"rather than extra breadth."
        )
    elif feasibility == "MODERATE":
        recommendation = (
            f"{profile['label']} is achievable but tight. Freeze the "
            f"recommended (non-mandatory) scope, prioritise the blocking "
            f"prerequisites, and hold a 4-week checkpoint to re-audit."
        )
    else:
        recommendation = (
            f"{profile['label']} is on a CRITICAL_TIMELINE: the mandatory gaps "
            f"need about {gap['months_required']} months of focused work and only "
            f"{months} are available. Either extend the runway, cut the path to "
            f"one primary goal, or switch to a lower-bandwidth target. Do not "
            f"attempt it by working longer hours - the arithmetic will still fail."
        )

    if not raw_skills:
        recommendation += (
            "  Note: no prior skills were declared, so this audit assumes a "
            "zero-start baseline - which is normal and workable, just expensive."
        )

    payload = {
        "target_path": path_id,
        "path_label": profile["label"],
        "target_cutoff_percentile": cutoff,
        "months_available": months,
        "skills_declared": raw_skills,
        "skills_unmapped": [s for s in raw_skills if normalize_token(s) not in matched_map],
        "matched_skills": matched_labels,
        "matched_topics": gap["matched_topics"],
        "coverage_percent": gap["coverage_percent"],
        "weighted_coverage_percent": gap["weighted_coverage_percent"],
        "missing_skills": missing_labels,
        "missing_prerequisites": missing_prereq_labels,
        "blocking_topics": blocking_labels,
        "mandatory_gap": {
            "coverage_percent": mandatory_gap["coverage_percent"],
            "missing_topics": [
                TOPIC_CATALOG[t]["label"] for t in mandatory_gap["missing_topics"]
            ],
            "months_required": mandatory_gap["months_required"],
        },
        "estimated_hours": gap["estimated_hours"],
        "weeks_required": gap["weeks_required"],
        "months_required": gap["months_required"],
        "buffer_months": buffer_months,
        "feasibility": feasibility,
        "tier_hours": gap["tier_hours"],
        "recommendation": recommendation,
    }
    return json.dumps(payload, indent=2)


@tool
def query_syllabus_knowledge_base(missing_topic: str, path_type: str) -> str:
    """Retrieve the indexed learning blueprint for a specific syllabus topic.

    This is the simulated RAG retriever over the curated topic blueprints. Use
    it for every gap reported by ``audit_skill_inventory`` - it returns the
    structured 3-tier learning sequence:

        BASIC          -> Fundamentals & Syntax
        INTERMEDIATE   -> Problem Solving & Core Concepts
        ADVANCED       -> PYQs, Case Work & Mock Tests

    Each tier includes hours, modules, exit criteria and free-first resources,
    so you can hand the student an actionable sequence instead of a topic name.

    Args:
        missing_topic: The gap to close, in any wording ("DSA", "recurison",
            "GATE OS memory management", "quantitative aptitude"). Fuzzy input is
            resolved against the catalog.
        path_type: The exam / path the blueprint is being built for - one of
            ``PLACEMENTS``, ``GATE_CS``, ``GATE_DA``, ``CAT``, ``STUDY_ABROAD``.

    Returns:
        A JSON string containing ``topic``, ``label``, ``resolved_input``,
        ``path_type``, ``prerequisites``, ``total_hours`` and ``sequence``
        (exactly three ordered tier objects with ``tier``, ``tier_name``,
        ``objective``, ``estimated_hours``, ``weeks``, ``modules``,
        ``exit_criteria`` and ``resources``).
    """
    path_id = resolve_path(path_type)
    if not path_id:
        return _tool_error(
            "query_syllabus_knowledge_base",
            "UNMAPPED_PATH",
            f"'{path_type}' is not a supported path_type.",
            received_path=path_type,
            received_topic=missing_topic,
        )

    topic_id = resolve_topic(missing_topic)
    if not topic_id:
        suggestions = _suggest_topics(missing_topic)
        return _tool_error(
            "query_syllabus_knowledge_base",
            "UNKNOWN_TOPIC",
            f"'{missing_topic}' could not be mapped to an indexed topic blueprint.",
            received_topic=missing_topic,
            closest_matches=suggestions,
        )

    blueprint = build_topic_blueprint(topic_id, path_id)
    payload = {
        "status": "OK",
        "resolved_input": missing_topic,
        "path_type": path_id,
        **blueprint,
        "search_metadata": {
            "index": "nexus-topic-blueprints-v1",
            "documents_scanned": len(TOPIC_CATALOG),
            "match_strategy": "alias -> keyword-overlap -> containment",
            "confidence": "HIGH" if normalize_token(missing_topic) in TOPIC_INDEX else "MEDIUM",
        },
    }
    return json.dumps(payload, indent=2)


@tool
def calculate_path_elimination_matrix(
    known_skills: List[str],
    target_paths: List[str],
    weekly_hours: int,
) -> str:
    """Run a multi-path trade-off analysis and recommend what to drop.

    Call this whenever the student names two or more targets (the classic
    "GATE + CAT + placements" trap). It quantifies:

        1. **Shared skill overlap** - hours already earned once across paths.
        2. **Structural conflict cost** - context-switching between incompatible
           syllabi (a GATE_CS + GATE_DA + CAT run costs ~22% extra hours).
        3. **Bandwidth bottleneck** - whether the total required weekly hours fit
           inside the student's actual ``weekly_hours``.
        4. **A ranked, drop-one recommendation** based on ROI per hour,
           time-to-deadline and strategic fit.

    Args:
        known_skills: The student's current, real skill inventory.
        target_paths: Two or more paths from ``PLACEMENTS``, ``GATE_CS``,
            ``GATE_DA``, ``CAT``, ``STUDY_ABROAD``. Aliases are accepted. Sending
            a single path is allowed and yields a single-goal analysis.
        weekly_hours: Hours the student can genuinely commit each week. Be
            honest and conservative here - optimistic inputs produce false
            confidence.

    Returns:
        A JSON string containing ``paths_evaluated``, ``skill_overlap``,
        ``conflict_penalty_percent``, ``per_path`` (hours, gap months,
        feasibility), ``total_required_weekly_hours``, ``weekly_hours_available``,
        ``bandwidth_bottleneck``, ``utilisation_percent``, ``elimination_matrix``,
        ``recommendation`` and ``weekly_split``.
    """
    hours_available = DEFAULT_WEEKLY_HOURS if weekly_hours is None else int(weekly_hours)
    hours_available = max(1, min(140, hours_available))

    requested = [p for p in (target_paths or []) if isinstance(p, str) and p.strip()]
    if not requested:
        return _tool_error(
            "calculate_path_elimination_matrix",
            "EMPTY_TARGET_PATHS",
            "target_paths was empty - at least one target path is required.",
        )

    resolved: List[str] = []
    unresolved: List[str] = []
    for raw in requested:
        path_id = resolve_path(raw)
        if not path_id:
            unresolved.append(raw)
        elif path_id not in resolved:
            resolved.append(path_id)

    if not resolved:
        return _tool_error(
            "calculate_path_elimination_matrix",
            "UNMAPPED_PATH",
            "None of the supplied target_paths could be mapped.",
            received_paths=requested,
        )
    if len(resolved) == 1 and unresolved:
        # Keep going with the resolvable one, but make the gap explicit.
        logger.warning("Unmapped paths during elimination matrix: %s", unresolved)

    _raw_skills, matched_map = clean_skills(known_skills)
    matched_topics = set(matched_map.values())
    multiplier = study_multiplier(
        float(PATH_PROFILES[resolved[0]]["target_months"])
    )

    # --------------------------------------------------------- overlap -----
    per_path: Dict[str, Dict[str, Any]] = {}
    topic_sets: Dict[str, set[str]] = {}
    for path_id in resolved:
        profile = PATH_PROFILES[path_id]
        topics = set(path_topic_ids(path_id))
        topic_sets[path_id] = topics
        gap = estimate_gap(topics, matched_topics, multiplier)
        mandatory_gap = estimate_gap(set(profile["mandatory"]), matched_topics, multiplier)
        feasibility = compute_feasibility(
            mandatory_gap["months_required"],
            float(profile["target_months"]),
            gap["weighted_coverage_percent"],
        )
        weeks_available = float(profile["target_months"]) * WEEKS_PER_MONTH
        required_weekly = gap["estimated_hours"] / weeks_available if weeks_available else 0.0
        per_path[path_id] = {
            "label": profile["label"],
            "target_months": profile["target_months"],
            "solo_weekly_hours": profile["weekly_hours"],
            "coverage_percent": gap["coverage_percent"],
            "weighted_coverage_percent": gap["weighted_coverage_percent"],
            "missing_topics": [TOPIC_CATALOG[t]["label"] for t in gap["missing_topics"]],
            "estimated_hours": gap["estimated_hours"],
            "required_weekly_hours": round(required_weekly, 1),
            "solo_months_required": gap["months_required"],
            "feasibility_alone": feasibility,
            "missing_hours_per_week": round(gap["estimated_hours"] / 52.0, 1),
        }

    # ------------------------------------------------------- conflict -----
    conflict_penalty = 0.0
    conflict_notes: List[str] = []
    if len(resolved) >= 2:
        signature = tuple(sorted(resolved))
        conflict_penalty = CONFLICT_LOOKUP.get(
            signature, 0.10 * (len(resolved) - 1)
        )
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                key = tuple(sorted((resolved[i], resolved[j])))
                note = SYNERGY_LOOKUP.get(key)
                if note:
                    conflict_notes.append(f"{key[0]} <-> {key[1]} SYNERGY: {note}")
                else:
                    conflict_notes.append(
                        f"{key[0]} <-> {key[1]} CONFLICT: "
                        "incompatible paper styles and question banks - little "
                        "skill transfer, so effort is duplicated."
                    )

    # -------------------------------------------------- bandwidth -----
    # Demand is the *gap* each path must close before its own deadline, not the
    # full-syllabus load - a student who already holds half the syllabus does
    # not need the same weekly hours as a zero-start student.
    base_weekly_hours = round(
        sum(per_path[p]["required_weekly_hours"] for p in resolved), 1
    )
    conflict_hours = round(base_weekly_hours * conflict_penalty, 1)
    total_required_weekly_hours = round(base_weekly_hours + conflict_hours, 1)
    utilisation = round(100.0 * total_required_weekly_hours / hours_available, 1)
    bottleneck = total_required_weekly_hours > hours_available

    # Overlap credits (shared topics only counted once).
    shared_all: Optional[set[str]] = None
    for path_id in resolved:
        shared_all = (
            topic_sets[path_id] if shared_all is None else (shared_all & topic_sets[path_id])
        )
    shared_ids = sorted(shared_all or set())
    overlap_hours = round(
        sum(TOPIC_CATALOG[t]["hours"] for t in shared_ids) * multiplier * (1 - conflict_penalty),
        1,
    )
    overlap_labels = [TOPIC_CATALOG[t]["label"] for t in shared_ids]

    # ------------------------------------------------ elimination -----
    elimination_matrix: List[Dict[str, Any]] = []
    ranked = rank_paths(
        {
            p: {
                "feasibility": per_path[p]["feasibility_alone"],
                "weighted_coverage_percent": per_path[p]["weighted_coverage_percent"],
            }
            for p in resolved
        }
    )
    for rank, path_id in enumerate(ranked, start=1):
        elimination_matrix.append(
            {
                "path": path_id,
                "label": PATH_PROFILES[path_id]["label"],
                "rank": rank,
                "verdict": "KEEP_AS_PRIMARY" if rank == 1 else "DROP_OR_DEFER",
                "reason": _elimination_reason(rank, per_path[path_id], shared_ids),
                "weighted_coverage_percent": per_path[path_id]["weighted_coverage_percent"],
                "feasibility_alone": per_path[path_id]["feasibility_alone"],
                "solo_weekly_hours": PATH_PROFILES[path_id]["weekly_hours"],
            }
        )

    primary = ranked[0]
    drop_list = [r["path"] for r in elimination_matrix if r["verdict"] != "KEEP_AS_PRIMARY"]

    if bottleneck:
        recommendation = (
            f"BANDWIDTH BOTTLENECK CONFIRMED: this plan needs "
            f"{total_required_weekly_hours} hrs/week (base {base_weekly_hours} + "
            f"{conflict_penalty * 100:.0f}% conflict overhead = {conflict_hours}) "
            f"but only {hours_available} are available - a "
            f"{utilisation}% utilisation. Drop {', '.join(drop_list)} from the "
            f"core plan. Keep {primary} as the single primary target and treat "
            f"the rest as optional/bonus only when the primary is on schedule."
        )
    else:
        recommendation = (
            f"Arithmetically this fits ({total_required_weekly_hours} of "
            f"{hours_available} hrs/week, {utilisation}% utilisation). Still, "
            f"attention is the real bottleneck, not hours: keep {primary} as the "
            f"non-negotiable primary and treat {', '.join(drop_list) or 'nothing else'} "
            f"as a stretch track. If a mock score slips for two consecutive weeks, "
            f"drop the stretch track immediately."
        )

    weekly_split = _weekly_split(resolved, primary, hours_available, bottleneck)

    payload = {
        "paths_evaluated": resolved,
        "unmapped_paths": unresolved,
        "skills_matched": sorted(matched_topics),
        "skill_overlap": {
            "shared_topics": overlap_labels,
            "shared_topic_count": len(shared_ids),
            "hours_saved_by_overlap": overlap_hours,
            "note": (
                "Shared topics are studied once and reused across paths - this is "
                "the only real bandwidth saving available."
            )
            if shared_ids
            else "No meaningful overlap: every path needs a disjoint skill set.",
        },
        "conflict_penalty_percent": round(conflict_penalty * 100, 1),
        "conflict_analysis": conflict_notes,
        "per_path": per_path,
        "total_required_weekly_hours": total_required_weekly_hours,
        "weekly_hours_available": hours_available,
        "utilisation_percent": utilisation,
        "bandwidth_bottleneck": bottleneck,
        "elimination_matrix": elimination_matrix,
        "recommendation": recommendation,
        "weekly_split": weekly_split,
    }
    return json.dumps(payload, indent=2)


def _elimination_reason(rank: int, stats: Dict[str, Any], shared: List[str]) -> str:
    """Compose a one-line justification for a keep/drop verdict.

    Args:
        rank: 1-based rank (1 = keep as primary).
        stats: The path's statistics block.
        shared: Topic ids shared across all evaluated paths.

    Returns:
        Human-readable justification string.
    """
    label = stats["label"]
    feasibility = stats["feasibility_alone"]
    overlap = len(shared)
    if rank == 1:
        if feasibility == "HIGH":
            return (
                f"Keep as primary: already {stats['weighted_coverage_percent']}% weighted "
                f"covered, and {label} is the target your timeline most supports."
            )
        if feasibility == "MODERATE":
            return (
                f"Keep as primary: {stats['weighted_coverage_percent']}% weighted "
                f"covered with a feasible timeline. Highest ROI per hour right now."
            )
        return (
            f"Keep as primary only because nothing else is better: even {label} is "
            f"CRITICAL_TIMELINE at {stats['solo_months_required']} months. "
            f"Audit whether the goal itself is realistic before committing."
        )
    if feasibility == "CRITICAL_TIMELINE":
        return (
            f"Drop/defer {label}: needs {stats['solo_months_required']} months and "
            f"cannot absorb a parallel syllabus."
        )
    return (
        f"Drop/defer {label}: only {overlap} topics are shared across your whole "
        f"target set, so most of its effort is duplicated - "
        f"{stats['required_weekly_hours']} hrs/week on a second, competing "
        f"deadline."
    )


def _weekly_split(
    paths: Sequence[str],
    primary: str,
    hours_available: int,
    bottleneck: bool,
) -> List[Dict[str, Any]]:
    """Allocate the student's weekly hours across the pursued paths.

    Args:
        paths: All resolved paths.
        primary: The retained primary path.
        hours_available: Total weekly hours the student committed.
        bottleneck: Whether the aggregate demand exceeds supply.

    Returns:
        List of ``{"path", "weekly_hours", "focus"}`` allocations.
    """
    if not paths:
        return []
    others = [p for p in paths if p != primary]
    primary_weight = 0.80 if others else 1.0
    primary_hours = round(hours_available * primary_weight, 1)
    if not others:
        return [
            {
                "path": primary,
                "weekly_hours": primary_hours,
                "focus": "Full-time single-goal focus on the primary target.",
            }
        ]

    remaining = round(hours_available - primary_hours, 1)
    per_other = round(remaining / len(others), 1)
    split = [
        {
            "path": primary,
            "weekly_hours": primary_hours,
            "focus": (
                "Protected block. Nothing else may touch this slot."
                if bottleneck
                else "Primary block; keep it stable even if the secondary slips."
            ),
        }
    ]
    for other in others:
        split.append(
            {
                "path": other,
                "weekly_hours": per_other,
                "focus": (
                    "Reduced 20% slot - treat as a bonus track and drop it the "
                    "moment the primary slips."
                    if bottleneck
                    else "Secondary slot - rotation only, no fragmented daily grind."
                ),
            }
        )
    return split


def _suggest_topics(raw: str) -> List[str]:
    """Return the closest catalog labels for an unresolved topic string.

    Args:
        raw: The unmatched topic text.

    Returns:
        Up to five human-readable catalog labels.
    """
    # Alias labels contain stopwords; word-level containment avoids the
    # "quantum contains quant" class of false positive.
    words = {w for w in token_words(raw) if w not in _STOPWORDS}
    scored: List[Tuple[float, str]] = []
    for topic_id, meta in TOPIC_CATALOG.items():
        alias_words: set[str] = set()
        for alias in list(meta["aliases"]) + [meta["label"]]:
            alias_words.update(w for w in token_words(alias) if w not in _STOPWORDS)
        if not alias_words:
            continue
        score = len(words & alias_words) / len(alias_words) if words else 0.0
        label_words = {w for w in token_words(meta["label"]) if w not in _STOPWORDS}
        if words and label_words and label_words <= words:
            score += 0.5
        if score > 0:
            scored.append((score, meta["label"]))
    scored.sort(reverse=True)
    return [label for score, label in scored[:5] if score > 0] or [
        TOPIC_CATALOG[t]["label"] for t in list(TOPIC_CATALOG)[:5]
    ]


NEXUS_TOOLS = [
    audit_skill_inventory,
    query_syllabus_knowledge_base,
    calculate_path_elimination_matrix,
]


# ===========================================================================
# SECTION 5 — AGENT (LangChain tool-calling ReAct loop)
# ===========================================================================

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are NEXUS AI - the career diagnostic and roadmap engine for Indian
    B.Tech engineering students. You are a calm, precise strategist, not a
    motivational speaker.

    ## Your mission
    A student arrives overwhelmed by four competing paths: Placements, GATE
    (CS/DA), CAT, and Higher Studies Abroad. Your job is to help them
    eliminate confusion with evidence, then hand them one executable roadmap.

    ## Non-negotiable tool protocol
    1. ALWAYS start with `audit_skill_inventory` for the student's stated
       target path and timeline. Never estimate coverage yourself.
    2. If the student names TWO OR MORE targets, you MUST call
       `calculate_path_elimination_matrix` before recommending anything.
       Deciding between paths without that tool is a protocol violation.
    3. Use `query_syllabus_knowledge_base` for every gap the audit reports, in
       gap-priority order. Do not invent study plans.
    4. All numbers you quote MUST come verbatim from tool output. If a tool
       returns `status: ERROR`, fix your arguments and retry once using the
       `supported_paths` / `closest_matches` lists - never fabricate a value.
    5. If a tool error is genuinely unresolvable, say plainly which inputs are
       missing and ask ONE specific clarifying question.

    ## Reasoning rules
    - Treat `CRITICAL_TIMELINE` as arithmetic, not opinion. Say so kindly but
      unambiguously; do not soften it into "it depends".
    - When the elimination matrix flags a bottleneck, name the path to drop and
      give the reason. One primary goal. Never endorse parallel GATE + CAT +
      Placements as "doable with discipline".
    - Prefer the highest-ROI path the timeline actually supports, even when it
      is not the student's first preference.
    - Use the student's own declared skills as evidence of capability, and frame
      every gap as a learnable, time-boxed task with hours attached.

    ## Tone (strict)
    - Encouraging but never cheerleading. No "you can do it!", no emojis, no
      exclamation marks.
    - Zero judgement about ability, family pressure, money, or "lazy" students.
      The only thing in question is sequencing, never the person.
    - Explicitly validate the feeling before fixing the logic:
      "Wanting all three is not a discipline problem - it is an arithmetic one."
    - Own your mistakes plainly if a tool contradicts the student's assumption.

    ## Required output structure (use these exact headings, in this order)
    1. **Reality Check** - the audit numbers in plain language (coverage,
       hours, feasibility flag).
    2. **What To Let Go Of** - explicit eliminations with trade-off reasoning,
       or an honest note that one goal is genuinely safe if kept in parallel.
    3. **Your Path (Basic -> Intermediate -> Advanced)** - the tiered sequence
       for the top 3-5 gaps, with weekly hour allocation.
    4. **First 14 Days** - a concrete, day-level starter plan.
    5. **Next Checkpoint** - the exact date/week and the metric that re-opens the
       decision (e.g. "if DSA accuracy is below 60% in week 4, we re-audit").

    ## Style
    - Markdown, short paragraphs, bold the verdicts and the numbers.
    - Be concise. No filler. No restating the student's question back to them.
    - Close by asking exactly ONE question that unblocks the next diagnosis.
    """
).strip()


def build_prompt_template() -> ChatPromptTemplate:
    """Build the agent prompt template.

    Returns:
        A :class:`ChatPromptTemplate` with the system message, optional chat
        history, human input and the agent scratchpad.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


def llm_configured() -> bool:
    """Report whether an API key is present for the OpenAI-compatible client.

    Every supported provider (OpenAI, Groq, OpenRouter, GitHub Models, Cerebras,
    Google AI Studio's OpenAI-compat endpoint, Ollama, ...) authenticates through
    the single ``OPENAI_API_KEY`` variable and is steered by ``OPENAI_BASE_URL``.

    Note: no provider-specific integrations exist by design. A native
    ``langchain-google-genai`` integration was evaluated and rejected, because
    every version of that package requires ``langchain-core>=1.0.0`` while
    ``create_tool_calling_agent`` - which this agent depends on - only exists in
    ``langchain<1.0``. Google's OpenAI-compatible endpoint is therefore used
    instead, keeping a single dependency stack.

    Returns:
        ``True`` when a key or non-offline provider is configured.
    """
    return key_manager.is_configured() and key_manager.provider != "offline"


def get_model(specific_key: Optional[str] = None) -> Any:
    """Instantiate the configured chat model via KeyRotationManager.

    Supports Google Gemini, Groq, OpenRouter, OpenAI, and custom endpoints.
    Allows passing a specific rotated key or defaults to the active key.

    Returns:
        A temperature-0 chat model instance.

    Raises:
        HTTPException: 503 when no LLM key is configured.
    """
    api_key = specific_key or key_manager.get_active_key()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No active LLM credentials configured. Set GEMINI_API_KEY(s), GROQ_API_KEY, "
                "or OPENAI_API_KEY, or switch to the Deterministic Offline Engine."
            ),
        )
    base_url = key_manager.get_base_url()
    model = key_manager.get_model()
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "api_key": api_key,
        "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "1")),
        "timeout": float(os.getenv("OPENAI_TIMEOUT", "60")),
    }
    if base_url and not base_url.startswith("local://"):
        kwargs["base_url"] = base_url
    logger.info(
        "Initialising ChatOpenAI provider=%s model=%s (key=%s)",
        key_manager.provider,
        model,
        mask_key(api_key),
    )
    return ChatOpenAI(**kwargs)


_agent_executor: Optional[AgentExecutor] = None


def get_agent_executor() -> AgentExecutor:
    """Return the process-wide singleton :class:`AgentExecutor`.

    The agent is built lazily on first use so that importing this module never
    requires credentials, and reused afterwards to avoid re-paying prompt
    construction cost.

    Returns:
        The shared agent executor with all NEXUS tools bound.

    Raises:
        HTTPException: 503 when the LLM cannot be configured.
    """
    global _agent_executor
    if _agent_executor is None:
        model = get_model()
        prompt = build_prompt_template()
        agent = create_tool_calling_agent(model, NEXUS_TOOLS, prompt)
        _agent_executor = AgentExecutor(
            agent=agent,
            tools=NEXUS_TOOLS,
            max_iterations=12,
            early_stopping_method="force",
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            verbose=os.getenv("NEXUS_VERBOSE", "false").lower() == "true",
        )
        logger.info("NEXUS agent executor ready with %d tools", len(NEXUS_TOOLS))
    return _agent_executor


#: Process-wide circuit breaker state for permanent LLM failures. Retrying an
#: exhausted billing balance or a revoked key only adds latency - every request
#: pays the full retry budget for an error that can never succeed.
LLM_BREAKER_COOLDOWN_SECONDS = float(os.getenv("LLM_CIRCUIT_COOLDOWN", "300"))
_llm_breaker: Dict[str, float] = {"open_until": 0.0, "reason": ""}

#: Error markers that indicate a permanent (non-retryable) LLM failure.
_PERMANENT_ERROR_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "invalid_api_key",
    "incorrect api key provided",
    "account_deactivated",
)


def is_permanent_llm_error(exc: BaseException) -> bool:
    """Classify an LLM exception as permanent (never worth retrying).

    Args:
        exc: Exception raised during agent invocation.

    Returns:
        True for auth / quota / account errors, False for transient ones.
    """
    if getattr(exc, "status_code", None) in (401, 403):
        return True
    if getattr(exc, "code", None) in {
        "insufficient_quota",
        "credit_balance_exhausted",
        "invalid_api_key",
    }:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _PERMANENT_ERROR_MARKERS)


def llm_circuit_state() -> Dict[str, Any]:
    """Inspect the LLM circuit breaker.

    Returns:
        Dict with ``open``, ``seconds_remaining`` and ``reason``.
    """
    remaining = max(0.0, _llm_breaker["open_until"] - time.monotonic())
    return {
        "open": remaining > 0,
        "seconds_remaining": round(remaining, 1),
        "reason": _llm_breaker["reason"],
    }


def _open_circuit(reason: str) -> None:
    """Trip the breaker so subsequent requests skip the LLM entirely.

    Args:
        reason: Human-readable cause stored for the health endpoint.
    """
    _llm_breaker["open_until"] = time.monotonic() + LLM_BREAKER_COOLDOWN_SECONDS
    _llm_breaker["reason"] = reason
    logger.warning(
        "LLM circuit OPEN for %ss: %s",
        LLM_BREAKER_COOLDOWN_SECONDS,
        reason,
    )


def _reset_circuit() -> None:
    """Close the breaker after a successful agent call."""
    if _llm_breaker["open_until"] or _llm_breaker["reason"]:
        logger.info("LLM circuit closed - agent recovered")
    _llm_breaker["open_until"] = 0.0
    _llm_breaker["reason"] = ""


#: Per-provider default models. Sending "gpt-4o-mini" to a Groq or Ollama
#: endpoint fails with model_not_found, so the default must follow the provider.
PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    # Groq free Developer plan. Note: llama-3.3-70b-versatile is Enterprise
    # ("Contact Sales") and is NOT available on the free tier. GPT-OSS is an
    # OpenAI open-weight model, so its tool-calling is the closest match to
    # what this agent's prompt was written against.
    "groq": "openai/gpt-oss-20b",
    "openrouter": "openai/gpt-oss-20b",
    # NOTE: Cerebras' free tier caps context near 8K, which an agent loop with
    # three JSON-returning tools will exceed. Use it for text, not for NEXUS.
    "cerebras": "gpt-oss-120b",
    # GitHub Models - free frontier-tier access via an Azure OpenAI shim.
    "github": "openai/gpt-4o",
    # Google AI Studio, reached through its OpenAI-compatible endpoint at
    # https://generativelanguage.googleapis.com/v1beta/openai/ . The most
    # generous free tier of any OpenAI-compatible provider, and tool-calling
    # support is verified by POST /api/v1/llm-check.
    "gemini": "gemini-2.5-flash",
    "local": "qwen2.5:14b",
    "custom": "gpt-4o-mini",
}


def resolve_provider() -> Dict[str, str]:
    """Describe the configured LLM provider for banners and health checks.

    Reflects the active state of KeyRotationManager (Gemini, Groq, OpenRouter,
    OpenAI or Offline Deterministic).

    Returns:
        Dict with provider name, base URL, model and configuration status.
    """
    st = key_manager.get_status()
    return {
        "name": st["provider"],
        "provider_name": st["provider_name"],
        "base_url": st["base_url"],
        "model": st["model"],
        "model_source": "key_manager",
        "api_key_configured": llm_configured(),
        "total_keys": st["total_keys"],
        "active_key_masked": st["active_key_masked"],
    }


def build_student_message(profile: "StudentProfilePayload") -> str:
    """Render a :class:`StudentProfilePayload` into the agent's human turn.

    Args:
        profile: Validated student profile.

    Returns:
        A structured natural-language brief for the agent.
    """
    months = profile.months_available
    deadline_note = (
        "not stated - assume 6 months and say so"
        if months is None
        else f"{months} months from today"
    )
    lines = [
        "Please run a full career diagnostic for this student.",
        "",
        f"Name: {profile.name or 'Anonymous'} (branch: {profile.branch or 'not stated'})",
        f"Year of study: {profile.year_of_study or 'not stated'}",
        f"CGPA: {profile.cgpa if profile.cgpa is not None else 'not stated'}",
        f"Weekly hours genuinely available: {profile.weekly_hours}",
        f"Target paths (in their priority order): {', '.join(profile.target_paths)}",
        f"Timeline: {deadline_note}",
        "",
        "Skills they claim (raw, unedited): "
        + (", ".join(profile.known_skills) if profile.known_skills else "NONE - zero baseline"),
    ]
    if profile.constraints:
        lines += ["", f"Hard constraints: {profile.constraints}"]
    if profile.additional_context:
        lines += ["", f"Additional context: {profile.additional_context}"]
    lines += [
        "",
        "Mandatory sequence:",
        "1. audit_skill_inventory on the primary target path.",
        "2. calculate_path_elimination_matrix across all named targets.",
        "3. query_syllabus_knowledge_base for the top gaps.",
        "4. Deliver the full report using the required headings.",
    ]
    return "\n".join(lines)


# ===========================================================================
# SECTION 6 — OFFLINE FALLBACK ENGINE (deterministic, no LLM required)
# ===========================================================================

FEASIBILITY_EMOJI_BANNER = {
    "HIGH": "Comfortably feasible",
    "MODERATE": "Feasible with discipline",
    "CRITICAL_TIMELINE": "Timeline is mathematically tight",
}


def _fmt_list(items: Sequence[str]) -> str:
    """Render a list as a markdown bullet list (or an em-dash if empty)."""
    if not items:
        return "- _none_"
    return "\n".join(f"- {item}" for item in items)


def _run_agent_sync(profile: "StudentProfilePayload") -> str:
    """Run the LangChain agent to completion with automatic API key rotation.

    Supports rotating multiple Gemini/Groq keys seamlessly on 429 rate limit
    or quota exhaustion without failing the student diagnosis.

    Args:
        profile: Validated student profile.

    Returns:
        The agent's markdown report.

    Raises:
        HTTPException: 503 when no LLM key is configured or all keys exhausted.
        ValueError: When the agent returns an empty report.
    """
    circuit = llm_circuit_state()
    if circuit["open"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"LLM temporarily disabled after a permanent failure "
                f"({circuit['reason']}). Retry in "
                f"{int(circuit['seconds_remaining'])}s or switch to Offline mode."
            ),
        )

    if not key_manager.keys and key_manager.provider != "offline":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No API keys configured. Using deterministic engine.",
        )

    max_attempts = max(1, len(key_manager.keys))
    last_exc: Optional[Exception] = None

    for attempt in range(max_attempts):
        active_key = key_manager.get_active_key()
        if not active_key:
            break

        try:
            model = get_model(specific_key=active_key)
            prompt = build_prompt_template()
            agent = create_tool_calling_agent(model, NEXUS_TOOLS, prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=NEXUS_TOOLS,
                max_iterations=12,
                early_stopping_method="force",
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                verbose=os.getenv("NEXUS_VERBOSE", "false").lower() == "true",
            )
            result = executor.invoke(
                {
                    "input": build_student_message(profile),
                    "chat_history": [],
                }
            )
            key_manager.record_success(active_key)
            _reset_circuit()
            report = str(result.get("output", "")).strip()
            if not report:
                raise ValueError("Agent returned an empty report.")
            return report

        except HTTPException:
            raise
        except Exception as exc:
            last_exc = exc
            has_next, next_key, reason = key_manager.record_error(active_key, exc)
            logger.warning(
                "Agent attempt %d/%d failed with key %s: %s",
                attempt + 1,
                max_attempts,
                mask_key(active_key),
                reason,
            )
            if not has_next or not next_key:
                if is_permanent_llm_error(exc) and len(key_manager.keys) <= 1:
                    _open_circuit(f"{exc.__class__.__name__}: {exc}")
                raise exc
            # Otherwise next iteration will pick up next_key!

    if last_exc:
        raise last_exc
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="All configured API keys exhausted.",
    )


def offline_diagnostic(profile: "StudentProfilePayload") -> Dict[str, Any]:
    """Produce a full deterministic diagnosis without calling any LLM.

    Used when the agent cannot be reached (missing key, provider outage,
    timeouts) so the endpoint always returns a useful, honest response instead
    of a 500.

    Args:
        profile: Validated student profile.

    Returns:
        Dict with ``mode``, ``narrative`` (markdown) and ``raw`` (tool payloads).
    """
    hours = profile.weekly_hours or DEFAULT_WEEKLY_HOURS
    months = (
        profile.months_available
        if profile.months_available is not None
        else int(PATH_PROFILES.get(resolve_path(profile.target_paths[0]) or "", {}).get("target_months", 6))
    )
    multiplier = study_multiplier(float(months))

    audits: Dict[str, Dict[str, Any]] = {}
    for raw_path in profile.target_paths:
        path_id = resolve_path(raw_path)
        if not path_id or path_id in audits:
            continue
        matched = clean_skills(profile.known_skills)[1]
        gap = estimate_gap(path_topic_ids(path_id), set(matched.values()), multiplier)
        mandatory_gap = estimate_gap(
            PATH_PROFILES[path_id]["mandatory"], set(matched.values()), multiplier
        )
        audits[path_id] = {
            "path": path_id,
            "label": PATH_PROFILES[path_id]["label"],
            "gap": gap,
            "mandatory_gap": mandatory_gap,
            "feasibility": compute_feasibility(
                mandatory_gap["months_required"],
                float(months),
                gap["weighted_coverage_percent"],
            ),
        }

    if not audits:
        return {
            "mode": "OFFLINE_DETERMINISTIC",
            "narrative": (
                "No supported target path was recognised. Supported paths: "
                + ", ".join(CANONICAL_PATHS)
                + "."
            ),
            "raw": {},
        }

    ranked_ids = rank_paths(
        {
            pid: {
                "feasibility": data["feasibility"],
                "weighted_coverage_percent": data["gap"]["weighted_coverage_percent"],
            }
            for pid, data in audits.items()
        }
    )
    primary_id = ranked_ids[0]
    primary = audits[primary_id]
    ordered_gaps = primary["gap"]["missing_topics"][:5]

    blueprints: Dict[str, Any] = {}
    for topic_id in ordered_gaps:
        blueprints[topic_id] = build_topic_blueprint(topic_id, primary_id)

    matrix: Dict[str, Any] = {}
    if len(profile.target_paths) > 1:
        matrix = json.loads(
            calculate_path_elimination_matrix.invoke(
                {
                    "known_skills": profile.known_skills,
                    "target_paths": profile.target_paths,
                    "weekly_hours": hours,
                }
            )
        )

    narrative_parts: List[str] = [
        f"## 1. Reality Check\n\n"
        f"**{primary['label']}** - weighted coverage "
        f"**{primary['gap']['weighted_coverage_percent']}%** "
        f"({primary['gap']['matched_topics'] and ', '.join(TOPIC_CATALOG[t]['label'] for t in primary['gap']['matched_topics']) or 'no mapped skills yet'}).",
        f"\n- Estimated remediation: **{primary['gap']['estimated_hours']} hrs** "
        f"({primary['gap']['weeks_required']} weeks at {DEFAULT_WEEKLY_HOURS} hrs/week)\n"
        f"- Timeline available: **{months} months**\n"
        f"- Buffer: **{round(months - primary['gap']['months_required'], 1)} months**\n"
        f"- Feasibility: **{primary['feasibility']}** "
        f"({FEASIBILITY_EMOJI_BANNER[primary['feasibility']]})",
        f"\n**Missing foundational prerequisites:**\n{_fmt_list([TOPIC_CATALOG[t]['label'] for t in primary['gap']['missing_prerequisites']])}",
        f"\n**All gaps for this path:**\n{_fmt_list([TOPIC_CATALOG[t]['label'] for t in primary['gap']['missing_topics']])}",
        "\n## 2. What To Let Go Of\n",
    ]

    if matrix and not matrix.get("status") == "ERROR":
        verdict_lines = []
        for row in matrix.get("elimination_matrix", []):
            verdict_lines.append(
                f"- **{row['path']}** -> {row['verdict']}: {row['reason']}"
            )
        narrative_parts.append("\n".join(verdict_lines))
        narrative_parts.append(
            f"\n\nBandwidth: **{matrix['total_required_weekly_hours']} hrs/week required** "
            f"vs **{matrix['weekly_hours_available']} available** "
            f"({matrix['utilisation_percent']}% utilisation). "
            f"{'Bottleneck confirmed.' if matrix['bandwidth_bottleneck'] else 'Fits, but attention is still the constraint.'}"
        )
    else:
        narrative_parts.append(
            f"- Single-goal plan for **{primary_id}**. Nothing to eliminate: "
            "one target, one timeline, one skill spine."
        )

    narrative_parts.append(
        "\n\n> Wanting several paths at once is not a discipline problem - it is "
        "an arithmetic one. Sequencing is the fix, not guilt."
    )

    narrative_parts.append("\n## 3. Your Path (Basic -> Intermediate -> Advanced)\n")
    tier_budget = {
        "BASIC": round(hours * 0.30, 1),
        "INTERMEDIATE": round(hours * 0.50, 1),
        "ADVANCED": round(hours * 0.20, 1),
    }
    narrative_parts.append(
        f"Weekly allocation at {hours} hrs: **{tier_budget['BASIC']} hrs BASIC**, "
        f"**{tier_budget['INTERMEDIATE']} hrs INTERMEDIATE**, "
        f"**{tier_budget['ADVANCED']} hrs ADVANCED**.\n"
    )
    for topic_id in ordered_gaps:
        bp = blueprints[topic_id]
        narrative_parts.append(f"\n### {bp['label']} ({bp['total_hours']} hrs)\n")
        for tier in bp["sequence"]:
            narrative_parts.append(
                f"- **{tier['tier_name']}** ({tier['estimated_hours']} hrs, ~{tier['weeks']} wks): "
                f"{tier['objective']}\n  - Exit criteria: {tier['exit_criteria']}"
            )

    narrative_parts.append(
        "\n## 4. First 14 Days\n"
        f"1. Days 1-2: finish the BASIC tier of {TOPIC_CATALOG[ordered_gaps[0]]['label']} and build its one-page cheat sheet.\n"
        f"2. Days 3-7: {TOPIC_CATALOG[ordered_gaps[0]]['label']} INTERMEDIATE problem set - 5 timed sets of 25.\n"
        f"3. Days 8-10: start BASIC of {TOPIC_CATALOG[ordered_gaps[1]]['label'] if len(ordered_gaps) > 1 else 'the next gap'}.\n"
        f"4. Days 11-14: one 180-minute timed mock in {primary['label']}; start the error log.\n"
        f"5. Day 14: re-audit. Continue only if the metric below is met."
    )
    narrative_parts.append(
        "\n## 5. Next Checkpoint\n"
        f"**Week 4.** Re-run the audit. Decision metric: at least "
        f"**70% accuracy on a timed 25-question set** in your top INTERMEDIATE "
        f"topic, and **one mock logged with an error log**. If either is missing, "
        f"the honest move is to defer {', '.join(r['path'] for r in (matrix or {}).get('elimination_matrix', []) if r['verdict'] != 'KEEP_AS_PRIMARY') or 'secondary goals'} "
        "and rebuild around the primary."
    )
    narrative_parts.append(
        "\n**One question to unblock the next diagnosis:** which single skill "
        "would you feel proud to be interviewed on in eight weeks?"
    )

    return {
        "mode": "OFFLINE_DETERMINISTIC",
        "narrative": "\n".join(narrative_parts),
        "raw": {
            "audits": {
                k: {
                    "feasibility": v["feasibility"],
                    "weighted_coverage_percent": v["gap"]["weighted_coverage_percent"],
                    "estimated_hours": v["gap"]["estimated_hours"],
                    "months_required": v["gap"]["months_required"],
                    "buffer_months": round(months - v["gap"]["months_required"], 1),
                    "missing_topics": [TOPIC_CATALOG[t]["label"] for t in v["gap"]["missing_topics"]],
                    "missing_prerequisites": [
                        TOPIC_CATALOG[t]["label"] for t in v["gap"]["missing_prerequisites"]
                    ],
                    "tier_hours": v["gap"]["tier_hours"],
                }
                for k, v in audits.items()
            },
            "elimination_matrix": matrix,
            "blueprints": {
                t: {
                    "label": b["label"],
                    "total_hours": b["total_hours"],
                    "sequence": [
                        {
                            "tier": s["tier"],
                            "estimated_hours": s["estimated_hours"],
                            "weeks": s["weeks"],
                            "objective": s["objective"],
                            "exit_criteria": s["exit_criteria"],
                        }
                        for s in b["sequence"]
                    ],
                }
                for t, b in blueprints.items()
            },
        },
    }


# ===========================================================================
# SECTION 7 — API SCHEMAS
# ===========================================================================


class StudentProfilePayload(BaseModel):
    """Validated inbound student profile for the diagnose endpoint."""

    name: Optional[str] = Field(
        default=None, max_length=80, description="Student's preferred name."
    )
    year_of_study: Optional[str] = Field(
        default=None,
        max_length=40,
        description="e.g. '3rd year / 6th sem', 'final year', '2nd year'.",
    )
    branch: Optional[str] = Field(
        default=None,
        max_length=80,
        description="e.g. 'CSE', 'IT', 'ECE', 'Mechanical'.",
    )
    cgpa: Optional[float] = Field(
        default=None,
        ge=0,
        le=10,
        description="CGPA on a 10-point scale (optional, used only as context).",
    )
    known_skills: List[str] = Field(
        default_factory=list,
        max_length=60,
        description=(
            "Self-declared skills in any wording, e.g. ['Python', 'OOPs', "
            "'LeetCode 150', 'a bit of SQL']."
        ),
    )
    target_paths: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=5,
        description=(
            "One or more target paths: PLACEMENTS, GATE_CS, GATE_DA, CAT, "
            "STUDY_ABROAD. Aliases such as 'GATE CSE' or 'iim cat' are accepted."
        ),
    )
    months_available: Optional[int] = Field(
        default=None,
        ge=1,
        le=120,
        description=(
            "Months of runway until the target date (1-120). Omit it to use "
            "each path's typical preparation window."
        ),
    )
    weekly_hours: int = Field(
        default=DEFAULT_WEEKLY_HOURS,
        ge=1,
        le=140,
        description="Hours genuinely available per week. Conservative input recommended.",
    )
    constraints: Optional[str] = Field(
        default=None,
        max_length=600,
        description="Non-negotiables, e.g. 'no unpaid internships', 'city constraint'.",
    )
    additional_context: Optional[str] = Field(
        default=None,
        max_length=1500,
        description="Anything else that changes the arithmetic: backlog, health, work.",
    )

    @field_validator("year_of_study", mode="before")
    @classmethod
    def _coerce_year_of_study(cls, value: Any) -> Optional[str]:
        """Accept integer or string for year_of_study and normalize."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            v = int(value)
            suffixes = {1: "1st Year", 2: "2nd Year", 3: "3rd Year", 4: "4th Year / Final Year"}
            return suffixes.get(v, f"{v}th Year")
        s = str(value).strip()
        return s or None

    @field_validator("known_skills", "target_paths", mode="before")
    @classmethod
    def _coerce_sequence(cls, value: Any) -> Any:
        """Accept comma-separated strings or ``None`` for list fields."""
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(v).strip() for v in value if str(v).strip()]
        raise ValueError("expected a list of strings or a comma-separated string")

    @field_validator("target_paths")
    @classmethod
    def _validate_target_paths(cls, value: List[str]) -> List[str]:
        """Ensure at least one requested path maps to the supported catalog."""
        if not value:
            raise ValueError(
                "target_paths must contain at least one of: " + ", ".join(CANONICAL_PATHS)
            )
        return value

    @model_validator(mode="after")
    def _validate_consistency(self) -> "StudentProfilePayload":
        """Derive sane defaults and normalise the path order."""
        if not self.target_paths:
            raise ValueError("target_paths is required to run a diagnosis.")
        # Preserve student priority order, drop duplicates and unmapped noise.
        seen: List[str] = []
        for raw in self.target_paths:
            resolved = resolve_path(raw)
            if resolved and resolved not in seen:
                seen.append(resolved)
        if not seen:
            raise ValueError(
                "None of the supplied target_paths is supported. Use one of: "
                + ", ".join(CANONICAL_PATHS)
            )
        object.__setattr__(self, "target_paths", seen)
        if self.months_available is None:
            object.__setattr__(
                self,
                "months_available",
                int(PATH_PROFILES[seen[0]]["target_months"]),
            )
        return self


class AuditResponse(BaseModel):
    """Structured response returned by ``POST /api/v1/diagnose``."""

    status: str = Field(description="'success', 'partial' or 'error'.")
    mode: str = Field(
        description="'AGENT' when the LangChain agent produced the report, "
        "'OFFLINE_DETERMINISTIC' when the deterministic engine answered."
    )
    agent: str = Field(default=APP_NAME, description="Generating agent identity.")
    version: str = Field(default=APP_VERSION)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    student: Dict[str, Any] = Field(description="Echo of the normalised profile.")
    primary_path: Optional[str] = Field(
        default=None, description="The recommended single primary path."
    )
    feasibility: Optional[Dict[str, str]] = Field(
        default=None, description="Path -> feasibility flag mapping."
    )
    coverage: Dict[str, Any] = Field(
        default_factory=dict, description="Per-path weighted coverage summary."
    )
    elimination_matrix: Dict[str, Any] = Field(
        default_factory=dict, description="Raw elimination-matrix tool output."
    )
    roadmap: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered per-topic 3-tier (Basic -> Intermediate -> Advanced) blueprints.",
    )
    report: str = Field(description="Full markdown diagnostic report.")
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "mode": "AGENT",
                "agent": "NEXUS AI",
                "version": "1.0.0",
                "generated_at": "2026-10-04T09:30:00+00:00",
                "student": {
                    "name": "Aarav",
                    "branch": "CSE",
                    "known_skills": ["Python", "OOPs", "LeetCode 120"],
                    "target_paths": ["GATE_CS", "CAT"],
                    "weekly_hours": 45,
                },
                "primary_path": "GATE_CS",
                "feasibility": {"GATE_CS": "MODERATE", "CAT": "CRITICAL_TIMELINE"},
                "coverage": {
                    "GATE_CS": {"weighted_coverage_percent": 38.5, "estimated_hours": 1420.0}
                },
                "report": "## 1. Reality Check\n...",
            }
        }
    }


# ===========================================================================
# SECTION 8 — FASTAPI APPLICATION
# ===========================================================================

app = FastAPI(
    title="NEXUS AI API",
    version=APP_VERSION,
    description=(
        "Autonomous B.Tech career diagnostic and roadmap system. Deterministic "
        "skill audit + multi-path elimination matrix + tiered learning roadmaps, "
        "orchestrated by a LangChain tool-calling agent."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler returning a structured, non-leaky error payload.

    Args:
        request: The incoming request.
        exc: The raised exception.

    Returns:
        JSONResponse with a 500 status and a trace-safe message.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "mode": "UNKNOWN",
            "agent": APP_NAME,
            "version": APP_VERSION,
            "report": "",
            "error": f"Internal error: {exc.__class__.__name__}. Check server logs.",
        },
    )


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LLMConfigPayload(BaseModel):
    """Payload to dynamically update provider and API keys at runtime."""
    provider: str = Field(description="'gemini', 'groq', 'openrouter', 'openai', 'offline'")
    keys: Any = Field(default="", description="Single key, comma-separated keys, or list of keys")
    model: Optional[str] = None
    base_url: Optional[str] = None


class CopilotRequest(BaseModel):
    """Payload for interactive student follow-up guidance."""
    message: str = Field(..., max_length=2000, description="Student question or prompt")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Current student diagnostic context")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous conversation turns")


@app.get("/", tags=["ui"], response_class=HTMLResponse)
async def root(request: Request) -> Any:
    """Serve the NEXUS AI interactive web dashboard (or JSON if requested)."""
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(content=await service_info())

    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content=(
            "<!DOCTYPE html><html><body style='font-family:sans-serif;background:#0d1224;color:#fff;padding:2rem;'>"
            "<h1>NEXUS AI Service Active</h1><p>Visit <a href='/docs' style='color:#06b6d4'>/docs</a> for API documentation.</p>"
            "</body></html>"
        )
    )


@app.get(f"{API_PREFIX}/info", tags=["meta"])
async def service_info() -> Dict[str, Any]:
    """Service banner with capabilities and configuration status."""
    st = key_manager.get_status()
    return {
        "agent": APP_NAME,
        "version": APP_VERSION,
        "tagline": "Navigated Execution & eXam Unified System",
        "status": "operational",
        "llm_configured": llm_configured(),
        "provider": st["provider"],
        "model": st["model"],
        "endpoints": {
            "ui": "GET /",
            "diagnose": f"POST {API_PREFIX}/diagnose",
            "copilot": f"POST {API_PREFIX}/copilot",
            "config_llm": f"GET/POST {API_PREFIX}/config/llm",
            "llm_check": f"POST {API_PREFIX}/llm-check",
            "health": "GET /health",
            "catalogue": f"GET {API_PREFIX}/catalogue",
            "docs": "GET /docs",
        },
        "supported_paths": list(CANONICAL_PATHS),
        "tools": [t.name for t in NEXUS_TOOLS],
    }


@app.get(f"{API_PREFIX}/config/llm", tags=["config"])
async def get_llm_config() -> Dict[str, Any]:
    """Inspect the LLM provider configuration and rotation status."""
    return key_manager.get_status()


@app.post(f"{API_PREFIX}/config/llm", tags=["config"])
async def update_llm_config(payload: LLMConfigPayload) -> Dict[str, Any]:
    """Dynamically update LLM provider and API keys at runtime."""
    raw_keys = payload.keys
    parsed_keys: List[str] = []
    if isinstance(raw_keys, str):
        for item in re.split(r"[\n,;]+", raw_keys):
            clean = item.strip().strip("'\"")
            if clean and clean not in parsed_keys:
                parsed_keys.append(clean)
    elif isinstance(raw_keys, (list, tuple)):
        for item in raw_keys:
            clean = str(item).strip().strip("'\"")
            if clean and clean not in parsed_keys:
                parsed_keys.append(clean)

    key_manager.set_configuration(
        provider=payload.provider,
        keys=parsed_keys,
        model=payload.model,
        base_url=payload.base_url,
    )
    _reset_circuit()
    logger.info("Updated LLM configuration via API: provider=%s, keys=%d", payload.provider, len(parsed_keys))
    return key_manager.get_status()


@app.post(f"{API_PREFIX}/copilot", tags=["copilot"])
async def copilot(req: CopilotRequest) -> Dict[str, Any]:
    """Interactive AI Advisor Copilot for student roadmap questions."""
    started = time.perf_counter()
    msg = req.message.strip()
    ctx = req.context or {}
    primary = ctx.get("primary_path", "your primary target")
    feas = (ctx.get("feasibility") or {}).get(primary, "MODERATE")

    # If LLM configured, invoke model
    if llm_configured():
        try:
            active_key = key_manager.get_active_key()
            model = get_model(specific_key=active_key)
            system_prompt = (
                "You are NEXUS AI Copilot, an empathetic, highly structured academic and career "
                "advisor for B.Tech engineering students. The student has run a diagnostic.\n"
                f"Diagnostic Context: Primary Path={primary}, Feasibility={feas}, Context={json.dumps(ctx)[:400]}.\n"
                "Give a direct, actionable, honest answer (max 3-4 short paragraphs or bullet points). "
                "Remind them to protect their bandwidth and eliminate conflicting goals."
            )
            chat_turns = [("system", system_prompt)]
            for turn in (req.chat_history or [])[-4:]:
                role = "human" if turn.get("role") == "user" else "ai"
                chat_turns.append((role, turn.get("content", "")))
            chat_turns.append(("human", msg))

            prompt = ChatPromptTemplate.from_messages(chat_turns)
            chain = prompt | model
            res = await run_in_threadpool(chain.invoke, {})
            reply_text = str(res.content).strip()
            key_manager.record_success(active_key)
            return {
                "reply": reply_text,
                "provider": key_manager.provider,
                "model": key_manager.get_model(),
                "mode": "AI_AGENT",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        except Exception as exc:
            logger.warning("Copilot LLM invocation failed, using deterministic guidance: %s", exc)

    # Deterministic contextual advisor response
    lower = msg.lower()
    if "time" in lower or "hour" in lower or "manage" in lower or "semester" in lower:
        advice = (
            f"**Time Allocation Advice for {primary}:**\n"
            "- Allocate **70%** of your weekly study bandwidth to your primary path.\n"
            "- Reserve **30%** for semester exams and buffer. Do not attempt two divergent exams (e.g. GATE CS + CAT) simultaneously if weekly hours are below 35.\n"
            "- Protect 1 day each week for revision of your blocking prerequisites."
        )
    elif "dsa" in lower or "placement" in lower or "leetcode" in lower:
        advice = (
            "**DSA & Placement Strategy:**\n"
            "- Focus on standard problem patterns (Two Pointers, Sliding Window, Fast/Slow Pointers, BFS/DFS, DP on Grids).\n"
            "- Prioritize depth on 100 core problems over breadth on 400 random problems.\n"
            "- Ensure your core CS subjects (OS, DBMS, Computer Networks) are reviewed concurrently, as technical interviewers weigh them heavily alongside coding."
        )
    elif "gate" in lower or "math" in lower:
        advice = (
            "**GATE Preparation Strategy:**\n"
            "- Engineering Mathematics and Discrete Mathematics carry 15-18% of the weight and are high-yield.\n"
            "- Practice previous year questions (PYQs) topic-by-topic immediately after finishing each subject module.\n"
            "- Avoid jumping into advanced mock tests until your foundational coverage is at least 65%."
        )
    else:
        advice = (
            f"**NEXUS Diagnostic Recommendation for {primary}:**\n"
            f"- Your current feasibility is flagged as **{feas}**.\n"
            "- Eliminate secondary low-overlap commitments to avoid timeline saturation.\n"
            "- Tackle missing prerequisite topics first before advancing to Tier 2 or Tier 3 problem sets."
        )

    return {
        "reply": advice,
        "provider": "offline",
        "model": "deterministic-v1",
        "mode": "DETERMINISTIC_ASSISTANT",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.get("/health", tags=["meta"])
async def health() -> Dict[str, Any]:
    """Liveness/readiness probe.

    Returns:
        Dict with process health and LLM availability.
    """
    return {
        "status": "healthy",
        "agent": APP_NAME,
        "version": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_configured": llm_configured(),
        "llm_provider": resolve_provider(),
        "llm_circuit": llm_circuit_state(),
        "tools_registered": len(NEXUS_TOOLS),
        "paths_supported": len(CANONICAL_PATHS),
        "topics_indexed": len(TOPIC_CATALOG),
    }


@app.get(f"{API_PREFIX}/catalogue", tags=["meta"])
async def catalogue() -> Dict[str, Any]:
    """Expose the deterministic knowledge base used by the tools.

    Returns:
        Dict with canonical paths, their topic spines and the indexed topics.
    """
    return {
        "paths": {
            path_id: {
                "label": profile["label"],
                "summary": profile["summary"],
                "typical_timeline_months": profile["target_months"],
                "recommended_weekly_hours": profile["weekly_hours"],
                "cutoff_percentile": profile["cutoff_percentile"],
                "mandatory": [TOPIC_CATALOG[t]["label"] for t in profile["mandatory"]],
                "recommended": [TOPIC_CATALOG[t]["label"] for t in profile["recommended"]],
            }
            for path_id, profile in PATH_PROFILES.items()
        },
        "topics": {
            topic_id: {
                "label": meta["label"],
                "tier": meta["tier"],
                "hours": meta["hours"],
                "importance": meta["importance"],
                "prerequisites": meta["prereqs"],
            }
            for topic_id, meta in TOPIC_CATALOG.items()
        },
        "conflicts": [
            {"paths": list(k), "overhead_fraction": v}
            for k, v in PATH_CONFLICTS.items()
        ],
    }


@app.post(f"{API_PREFIX}/llm-check", tags=["meta"])
async def llm_check() -> Dict[str, Any]:
    """Preflight the configured LLM: credentials, model reachability and
    tool-calling support.

    NEXUS is an agent, so a model that chats but cannot emit tool calls is
    useless here - and that is the most common failure when swapping OpenAI for
    a free provider. Run this before trusting a real diagnosis.

    Returns:
        Dict with ``ok``, ``provider``, ``text_roundtrip``, ``tool_calling``,
        ``tools_exposed``, ``recommended_models`` and ``remedy``.
    """
    provider = resolve_provider()
    started = time.perf_counter()

    result: Dict[str, Any] = {
        "ok": False,
        "provider": provider,
        "api_key_present": llm_configured(),
        "text_roundtrip": None,
        "tool_calling": None,
        "tools_exposed": [t.name for t in NEXUS_TOOLS],
        "recommended_models": PROVIDER_DEFAULT_MODELS,
        "remedy": None,
    }

    if not result["api_key_present"]:
        result["remedy"] = (
            f"OPENAI_API_KEY is not set. Provider is '{provider['name']}' at "
            f"{provider['base_url']}. Set OPENAI_API_KEY in your environment (or a "
            ".env file) to that provider's key and restart - every provider NEXUS "
            "supports is reached through this single variable."
        )
        return result

    try:
        model = get_model()

        # 1. Plain text round-trip.
        text_reply = model.invoke("Reply with exactly: NEXUS_OK")
        text_ok = "NEXUS_OK" in str(text_reply.content).upper()
        result["text_roundtrip"] = {
            "ok": text_ok,
            "sample": str(text_reply.content)[:120],
        }

        # 2. Tool-calling round-trip: the capability NEXUS actually depends on.
        probe = model.bind_tools(NEXUS_TOOLS)
        tool_reply = probe.invoke(
            "Call the audit_skill_inventory tool. known_skills=['Python'], "
            "target_path='GATE_CS', months_available=6. Do not answer in text."
        )
        tool_calls = getattr(tool_reply, "tool_calls", None) or []
        result["tool_calling"] = {
            "ok": bool(tool_calls),
            "emitted_tool": tool_calls[0]["name"] if tool_calls else None,
            "raw_content": str(getattr(tool_reply, "content", ""))[:160],
        }

        result["ok"] = bool(text_ok and tool_calls)
        if result["ok"]:
            _reset_circuit()
            result["remedy"] = None
        else:
            result["remedy"] = (
                "The endpoint answered but did not emit a tool call. Try a "
                f"tool-capable model, e.g. {PROVIDER_DEFAULT_MODELS.get(provider['name'])}."
            )
    except Exception as exc:
        # Classify the failure into an actionable hint. These are the four
        # errors that actually occur when swapping providers, and each one has a
        # different fix, so guessing in the message wastes the user's time.
        detail = f"{exc.__class__.__name__}: {exc}"
        hint = {
            "model_not_found": (
                "The endpoint does not know this model. Set OPENAI_MODEL to a model "
                f"that '{provider['name']}' actually serves - the default NEXUS "
                f"picked was '{provider['model']}'."
            ),
            "insufficient_quota": (
                "The account authenticated but has no usable balance. This provider "
                "requires a payment method or a different key."
            ),
            "invalid_api_key": "The key was rejected. Re-copy it and restart.",
            "context_length": (
                "The prompt exceeded this endpoint's context window. This provider's "
                "free tier caps context low; pick one with a larger limit."
            ),
        }
        for needle, advice in hint.items():
            if needle in detail.lower().replace(" ", "_"):
                result["remedy"] = f"{detail} - {advice}"
                break
        else:
            result["remedy"] = detail
        logger.warning("LLM preflight failed: %s", detail)

    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


@app.post(f"{API_PREFIX}/diagnose", response_model=AuditResponse, tags=["diagnosis"])
async def diagnose(profile: StudentProfilePayload) -> AuditResponse:
    """Run a full career diagnosis for one student.

    Execution flow:
        1. Validate and normalise the profile (Pydantic v2).
        2. Run the deterministic audit for every requested path.
        3. Invoke the LangChain agent with all NEXUS tools bound.
        4. If the agent is unavailable for any reason, fall back to the
           deterministic offline engine so the caller still gets a roadmap.
        5. Return a structured :class:`AuditResponse`.

    Args:
        profile: The validated student profile.

    Returns:
        The :class:`AuditResponse` payload.

    Raises:
        HTTPException: 422 on invalid input (raised by FastAPI), 500 on an
            unrecoverable internal failure.
    """
    started = time.perf_counter()
    logger.info(
        "Diagnosis requested: paths=%s weekly_hours=%s months=%s skills=%d",
        profile.target_paths,
        profile.weekly_hours,
        profile.months_available,
        len(profile.known_skills),
    )

    warnings: List[str] = []
    months = int(profile.months_available or 0)
    multiplier = study_multiplier(float(months))

    # ---------------------------------------------- deterministic audit ----
    audits: Dict[str, Dict[str, Any]] = {}
    _clean, matched_map = clean_skills(profile.known_skills)
    matched_topics = set(matched_map.values())

    for path_id in profile.target_paths:
        gap = estimate_gap(path_topic_ids(path_id), matched_topics, multiplier)
        mandatory_gap = estimate_gap(
            PATH_PROFILES[path_id]["mandatory"], matched_topics, multiplier
        )
        audits[path_id] = {
            "path": path_id,
            "label": PATH_PROFILES[path_id]["label"],
            "coverage_percent": gap["coverage_percent"],
            "weighted_coverage_percent": gap["weighted_coverage_percent"],
            "matched_topics": gap["matched_topics"],
            "missing_topics": gap["missing_topics"],
            "missing_prerequisites": gap["missing_prerequisites"],
            "blocking_topics": gap["blocking_topics"],
            "estimated_hours": gap["estimated_hours"],
            "weeks_required": gap["weeks_required"],
            "months_required": gap["months_required"],
            "tier_hours": gap["tier_hours"],
            "feasibility": compute_feasibility(
                mandatory_gap["months_required"],
                float(months),
                gap["weighted_coverage_percent"],
            ),
        }

    # ------------------------------------------ elimination matrix (tool) ---
    matrix: Dict[str, Any] = {}
    try:
        matrix = json.loads(
            calculate_path_elimination_matrix.invoke(
                {
                    "known_skills": profile.known_skills,
                    "target_paths": profile.target_paths,
                    "weekly_hours": int(profile.weekly_hours),
                }
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Elimination matrix tool failed")
        warnings.append(f"Elimination matrix unavailable: {exc.__class__.__name__}")

    # ---------------------------------------------- blueprints (tool) ------
    primary_path: str = profile.target_paths[0]
    ranked = rank_paths(
        {
            pid: {
                "feasibility": data["feasibility"],
                "weighted_coverage_percent": data["weighted_coverage_percent"],
            }
            for pid, data in audits.items()
        }
    )
    primary_path = ranked[0]
    primary = audits[primary_path]

    roadmap: List[Dict[str, Any]] = []
    for topic_id in primary["missing_topics"][:5]:
        try:
            blueprint = json.loads(
                query_syllabus_knowledge_base.invoke(
                    {"missing_topic": topic_id, "path_type": primary_path}
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Syllabus tool failed for topic %s", topic_id)
            warnings.append(f"Blueprint unavailable for {topic_id}: {exc.__class__.__name__}")
            continue
        if blueprint.get("status") == "OK":
            roadmap.append(
                {
                    "topic": blueprint["topic"],
                    "label": blueprint["label"],
                    "path_type": blueprint["path_type"],
                    "total_hours": blueprint["total_hours"],
                    "prerequisites": blueprint["prerequisites"],
                    "sequence": blueprint["sequence"],
                }
            )

    coverage_summary = {
        path_id: {
            "weighted_coverage_percent": data["weighted_coverage_percent"],
            "coverage_percent": data["coverage_percent"],
            "estimated_hours": data["estimated_hours"],
            "months_required": data["months_required"],
            "buffer_months": round(months - data["months_required"], 1),
            "feasibility": data["feasibility"],
            "missing_topics": [TOPIC_CATALOG[t]["label"] for t in data["missing_topics"]],
            "missing_prerequisites": [
                TOPIC_CATALOG[t]["label"] for t in data["missing_prerequisites"]
            ],
            "blocking_topics": [TOPIC_CATALOG[t]["label"] for t in data["blocking_topics"]],
            "tier_hours": data["tier_hours"],
        }
        for path_id, data in audits.items()
    }

    # ------------------------------------------------------- agent pass ----
    report: str
    mode: str
    api_status = "success"

    agent_error: Optional[str] = None
    try:
        # AgentExecutor.invoke() is blocking I/O, so it runs in the threadpool
        # to keep the event loop free for concurrent diagnoses.
        report = await run_in_threadpool(_run_agent_sync, profile)
        mode = "AGENT"
    except HTTPException as exc:
        agent_error = str(exc.detail)
        logger.warning("Agent unavailable (HTTPException): %s", agent_error)
        mode = "OFFLINE_DETERMINISTIC"
        api_status = "partial"
        warnings.append(agent_error)
    except Exception as exc:
        agent_error = f"{exc.__class__.__name__}: {exc}"
        logger.exception("Agent invocation failed")
        mode = "OFFLINE_DETERMINISTIC"
        api_status = "partial"
        warnings.append(
            "LLM agent failed; served the deterministic engine instead "
            f"({exc.__class__.__name__})."
        )

    if mode == "OFFLINE_DETERMINISTIC":
        fallback = offline_diagnostic(profile)
        report = fallback["narrative"]

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "Diagnosis complete in %s ms | primary=%s mode=%s feasibility=%s",
        elapsed_ms,
        primary_path,
        mode,
        {p: d["feasibility"] for p, d in audits.items()},
    )

    return AuditResponse(
        status=api_status,
        mode=mode,
        agent=APP_NAME,
        version=APP_VERSION,
        student=profile.model_dump(exclude={"additional_context"}),
        primary_path=primary_path,
        feasibility={p: d["feasibility"] for p, d in audits.items()},
        coverage=coverage_summary,
        elimination_matrix=matrix if matrix.get("status") != "ERROR" else {},
        roadmap=roadmap,
        report=report,
        warnings=warnings,
        error=agent_error,
    )


# ===========================================================================
# SECTION 9 — ENTRYPOINT
# ===========================================================================

def main() -> None:
    """Launch the NEXUS AI service with Uvicorn.

    Configuration comes from the environment (or ``.env``):
        ``HOST``, ``PORT``, ``RELOAD``, ``WORKERS``, ``LOG_LEVEL``.

    Raises:
        SystemExit: If ``uvicorn`` is not installed.
    """
    if not _UVICORN_AVAILABLE:
        raise SystemExit(
            "uvicorn is not installed. Install dependencies with:\n"
            "    pip install fastapi uvicorn[standard] langchain langchain-openai "
            "langchain-community pydantic python-dotenv"
        )

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"}
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    provider = resolve_provider()

    print("=" * 72)
    print(f"  {APP_NAME} v{APP_VERSION}")
    print("  Navigated Execution & eXam Unified System")
    print("=" * 72)
    print(f"  Docs        : http://{host}:{port}/docs")
    print(f"  Diagnose    : POST http://{host}:{port}{API_PREFIX}/diagnose")
    print(f"  LLM check   : POST http://{host}:{port}{API_PREFIX}/llm-check")
    print(f"  Health      : http://{host}:{port}/health")
    print(f"  Provider    : {provider['name']} ({provider['base_url']})")
    print(
        f"  Model       : {provider['model']} (from {provider['model_source']})"
    )
    print(
        "  LLM status  : "
        + ("configured" if llm_configured() else "NOT configured - offline mode only")
    )
    print(f"  Tools       : {', '.join(t.name for t in NEXUS_TOOLS)}")
    print("=" * 72)

    uvicorn.run(
        "main:app" if reload_flag or workers > 1 else app,
        host=host,
        port=port,
        reload=reload_flag,
        workers=workers if not reload_flag else None,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
