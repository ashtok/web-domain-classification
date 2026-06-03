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


def parse_multiclass_label(content: str) -> str:
    """Map raw model output to one of MULTICLASS_LABELS, defaulting to OTHER."""
    up = (content or "").strip().upper()
    # Check the specific categories before OTHER so 'OTHER' isn't masked.
    for name in CATEGORY_NAMES:
        if name in up:
            return name
    return OTHER


def parse_binary_label(content: str, category: str) -> str:
    """Map raw binary output to {category} or NON_{category}, defaulting negative."""
    up = (content or "").strip().upper()
    neg = f"NON_{category}"
    neg_spaced = f"NON {category}"
    if neg in up or neg_spaced in up:
        return neg
    if category in up:
        return category
    return neg
