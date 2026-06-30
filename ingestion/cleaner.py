# ingestion/cleaner.py

import re
from bs4 import BeautifulSoup

# SEC boilerplate that appears at the top of every filing — strip it
SEC_BOILERPLATE_PATTERNS = [
    r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION.*?Washington.*?D\.C\.",
    r"FORM\s+10-[KQ]",
    r"For the (?:fiscal|quarterly|annual) (?:year|period).*?\d{4}",
    r"Commission file number.*?\n",
    r"(?:Indicate by check mark|Check the appropriate box).*?\n",
]

# MD&A and Risk Factors section headers — these are the only sections we want
TARGET_SECTION_PATTERNS = [
    r"item\s*7[\.\s]+management.{0,30}discussion",   # MD&A
    r"item\s*1a[\.\s]+risk\s+factors",               # Risk Factors
    r"item\s*2[\.\s]+management.{0,30}discussion",   # 10-Q MD&A (different numbering)
]

# Where the next irrelevant section starts — stop extracting here
STOP_SECTION_PATTERNS = [
    r"item\s*7a[\.\s]+quantitative",
    r"item\s*8[\.\s]+financial\s+statements",
    r"item\s*2[\.\s]+unregistered",
    r"item\s*1b[\.\s]+unresolved",
]


def clean_html(raw_html: str) -> str:
    """
    Full pipeline: raw 10-K/10-Q HTML → clean narrative text.
    Steps:
      1. Parse with BeautifulSoup
      2. Strip non-text tags (script, style, table)
      3. Get plain text
      4. Strip SEC boilerplate
      5. Extract only MD&A + Risk Factors sections
      6. Normalize whitespace
    """
    # Step 1: parse
    soup = BeautifulSoup(raw_html, "lxml")

    # Step 2: remove noise tags entirely
    for tag in soup(["script", "style", "table", "thead",
                     "tbody", "tr", "td", "th", "img",
                     "footer", "header", "nav"]):
        tag.decompose()

    # Step 3: get plain text
    text = soup.get_text(separator=" ")

    # Step 4: strip SEC boilerplate
    for pattern in SEC_BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)

    # Step 5: extract target sections (MD&A + Risk Factors)
    extracted = extract_target_sections(text)

    # Fall back to full text if section extraction finds nothing
    # (some older filings use non-standard headers)
    if len(extracted.strip()) < 500:
        extracted = text

    # Step 6: normalize whitespace
    extracted = re.sub(r"\s+", " ", extracted).strip()
    extracted = re.sub(r" \. ", ". ", extracted)   # fix spaced periods

    return extracted


def extract_target_sections(text: str) -> str:
    """
    Finds MD&A and Risk Factors sections by header pattern,
    extracts text until the next irrelevant section starts.
    Returns combined extracted text from both sections.
    """
    text_lower = text.lower()
    extracted_parts = []

    for start_pattern in TARGET_SECTION_PATTERNS:
        start_match = re.search(start_pattern, text_lower)
        if not start_match:
            continue

        start_pos = start_match.start()

        # Find where this section ends (next major section header)
        end_pos = len(text)
        for stop_pattern in STOP_SECTION_PATTERNS:
            stop_match = re.search(stop_pattern, text_lower[start_pos + 100:])
            if stop_match:
                candidate = start_pos + 100 + stop_match.start()
                end_pos   = min(end_pos, candidate)

        section_text = text[start_pos:end_pos]
        extracted_parts.append(section_text)

    return " ".join(extracted_parts)


def is_meaningful(text: str, min_chars: int = 500) -> bool:
    """
    Quick sanity check — returns False if cleaning produced
    too little text (likely a failed parse or empty section).
    """
    return len(text.strip()) >= min_chars