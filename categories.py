# categories.py
"""
Central definition of the classification categories and the prompts used by
both classification strategies (multi-class and per-category binary).

Adding a new category = adding one entry to CATEGORIES. Everything downstream
(benchmark building, classification, evaluation) reads from here, so MEDICAL,
CYBERSECURITY and CLIMATE are all first-class and treated identically.
"""

# The "negative" / catch-all label.
OTHER = "OTHER"

# Each category gets:
#   - definition:  what counts as a positive (used in both prompts)
#   - exclude:     what does NOT count, to suppress false positives
CATEGORIES = {
    "MEDICAL": {
        "definition": (
            "Clinical medicine (symptoms, diagnosis, treatment, diseases, surgery), "
            "drugs / pharmaceuticals (medication names, dosages, side effects), "
            "medical research or clinical studies, public health / epidemiology / "
            "health policy, or healthcare services (hospitals, doctors, insurance "
            "coverage of treatments)."
        ),
        "exclude": (
            "A document that merely mentions a medical term in passing, or whose "
            "primary topic is travel, food, law, tech, politics, sports, "
            "entertainment, automotive or finance."
        ),
    },
    "CYBERSECURITY": {
        "definition": (
            "Information security and cybersecurity: malware, ransomware, phishing, "
            "vulnerabilities and exploits (CVEs), hacking and attacks, encryption / "
            "cryptography, network and endpoint security, data breaches, security "
            "tooling, incident response, or security policy and compliance."
        ),
        "exclude": (
            "General IT, software development or consumer tech that does not center "
            "on security; a document that merely mentions a password or 'hacker' "
            "in passing."
        ),
    },
    "CLIMATE": {
        "definition": (
            "Climate and climate change: global warming, greenhouse gas emissions, "
            "carbon footprint, climate policy and agreements, renewable energy in a "
            "climate context, climate science / modelling, extreme weather attributed "
            "to climate change, or environmental sustainability framed around climate."
        ),
        "exclude": (
            "Ordinary daily weather reports, general nature / gardening content, or a "
            "document that merely mentions the word 'climate' or 'weather' in passing "
            "without a climate-change focus."
        ),
    },
}

CATEGORY_NAMES = list(CATEGORIES.keys())
# All labels the multi-class classifier may emit.
MULTICLASS_LABELS = CATEGORY_NAMES + [OTHER]


def multiclass_system_prompt() -> str:
    """System prompt for the single multi-class classifier."""
    lines = [
        "You are a strict topical classifier for German web documents.",
        "Assign the document to exactly ONE of the following categories, based on "
        "its PRIMARY topic:",
        "",
    ]
    for name, spec in CATEGORIES.items():
        lines.append(f"{name}: {spec['definition']}")
    lines.append(
        f"{OTHER}: anything whose primary topic does not clearly fit one of the "
        "categories above (e.g. travel, food, law, politics, sports, "
        "entertainment, automotive, finance, general tech, navigation/product pages)."
    )
    lines += [
        "",
        "A document only qualifies for a category if that is its PRIMARY topic, not "
        "a passing mention.",
        "",
        f"Respond with exactly one word: {', '.join(MULTICLASS_LABELS)}. "
        "No explanation.",
    ]
    return "\n".join(lines)


def binary_system_prompt(category: str) -> str:
    """System prompt for a single-category binary classifier (POSITIVE/NEGATIVE)."""
    spec = CATEGORIES[category]
    pos = category          # e.g. MEDICAL
    neg = f"NON_{category}"  # e.g. NON_MEDICAL
    return (
        f"You are a strict binary classifier for German web documents.\n"
        f"Label the document {pos} only if its PRIMARY topic is:\n"
        f"{spec['definition']}\n\n"
        f"Label it {neg} if:\n"
        f"- {spec['exclude']}\n"
        f"- The medical/security/climate term is only mentioned in passing.\n"
        f"- It is a product listing, forum post, or navigation page unrelated to "
        f"the topic.\n\n"
        f"Respond with exactly one word: {pos} or {neg}. No explanation."
    )


def message_text(msg) -> str:
    """Best label-bearing text from a chat message.

    Reasoning models (e.g. Qwen3.x) return content=None and put everything in
    `reasoning_content`; the final label is at the END of that reasoning. So we
    prefer content, else fall back to the TAIL of the reasoning trace.
    """
    content = (getattr(msg, "content", None) or "").strip()
    if content:
        return content
    reasoning = getattr(msg, "reasoning_content", None) or ""
    if not reasoning:
        psf = getattr(msg, "provider_specific_fields", None)
        if isinstance(psf, dict):
            reasoning = (
                psf.get("reasoning_content")
                or psf.get("reasoning")
                or psf.get("thinking")
                or ""
            )
    # The decision is at the end of the reasoning; return the tail.
    return reasoning[-600:]


def parse_multiclass_label(content: str) -> str:
    """Map raw model output to one of MULTICLASS_LABELS, defaulting to OTHER.

    For reasoning traces the label is at the end, so scan from the tail and take
    the LAST category mentioned rather than the first.
    """
    up = (content or "").strip().upper()
    # Whichever label appears LAST wins (the conclusion of a reasoning trace).
    # OTHER competes on position too, so a trace that weighs the real categories
    # and then concludes OTHER resolves correctly.
    last_pos, last_name = -1, OTHER
    for name in MULTICLASS_LABELS:           # includes OTHER
        idx = up.rfind(name)
        if idx > last_pos:
            last_pos, last_name = idx, name
    return last_name


def parse_binary_label(content: str, category: str) -> str:
    """Map raw binary output to {category} or NON_{category}, defaulting negative.

    Scan from the tail: NON_<cat> contains <cat>, so we compare the last position
    of each and let whichever appears later win.
    """
    up = (content or "").strip().upper()
    neg = f"NON_{category}"
    neg_idx = max(up.rfind(neg), up.rfind(f"NON {category}"))
    # Position of a *standalone* positive: find <category> not preceded by "NON".
    pos_idx = -1
    start = 0
    while True:
        i = up.find(category, start)
        if i == -1:
            break
        preceding = up[max(0, i - 4):i]
        if "NON" not in preceding:
            pos_idx = i
        start = i + 1
    if neg_idx > pos_idx:
        return neg
    if pos_idx >= 0:
        return category
    return neg


# ---------------------------------------------------------------------------
# Multi-LABEL strategy: a document may belong to several categories at once
# (e.g. a cyberattack on a hospital -> MEDICAL + CYBERSECURITY). OTHER is
# treated as EXCLUSIVE: it applies only when no real category does.
# ---------------------------------------------------------------------------

def multilabel_system_prompt() -> str:
    """System prompt for the multi-label classifier (zero or more categories)."""
    lines = [
        "You are a strict topical classifier for German web documents.",
        "A document may belong to MULTIPLE of the following categories at once. "
        "List EVERY category that is a substantial topic of the document "
        "(not just a passing mention):",
        "",
    ]
    for name, spec in CATEGORIES.items():
        lines.append(f"{name}: {spec['definition']}")
    lines += [
        "",
        f"If none of these categories substantially apply, respond with {OTHER}.",
        "Do NOT combine OTHER with a real category — use OTHER only on its own.",
        "",
        "Respond with the applicable category names separated by commas, e.g. "
        f"'MEDICAL, CYBERSECURITY' or a single '{OTHER}'. No explanation.",
    ]
    return "\n".join(lines)


def parse_multilabel(content: str) -> frozenset:
    """Map raw output to a set of category labels.

    Returns a frozenset of real CATEGORY_NAMES, or frozenset({OTHER}) if none
    apply. Substring-scans for each category name so it is robust to reasoning
    traces and varied formatting. OTHER is exclusive: dropped if any real
    category is present.
    """
    up = (content or "").upper()
    found = {name for name in CATEGORY_NAMES if name in up}
    if found:
        return frozenset(found)
    return frozenset({OTHER})
