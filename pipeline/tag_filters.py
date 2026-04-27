"""
tag_filters.py
Shared junk-tag filtering and normalization for Last.fm tags.

Used by both fetch_lastfm.py (at ingest time) and clean_data.py (at merge
time, which re-applies filters to cached Last.fm JSON so stale data written
before these rules existed is cleaned up without a forced re-fetch).

Keeping these rules in one place means adding a new junk tag to JUNK_EXACT
or JUNK_SUBSTRINGS takes effect in both scripts on the next run. Previously
the two filter sets had to be kept in sync manually.
"""

import re


# Only these characters are allowed in a clean tag. Anything else (typos
# like "synth fumk", stray punctuation, emoji, non-ASCII) gets rejected.
_VALID_TAG_RE = re.compile(r"^[a-zA-Z0-9 \-&]+$")


# Substring checks applied to tag names (case-insensitive). Catches patterns
# like "best of 2020", "i love singing along", "albums i own".
JUNK_SUBSTRINGS = [
    "seen live",
    "favourite",
    "favorites",
    "under 2000 listeners",
    "best of",
    "album of the year",
    "i love",
    "i like",
    "i can",
    "i own",
    "where is",
    "my top",
    "fav ",          # trailing space avoids matching "favourite"
    "cum",
    "goodness",      # "shoegazing goodness" and similar praise-noise tags
]


# Exact-match tags (case-insensitive) to discard. Grouped for readability.
JUNK_EXACT = {
    # Vague genre labels (too broad to add useful signal)
    "indie", "alternative", "progressive",
    # Generic meta / collection tags
    "aoty", "albums", "all", "classic albums", "vinyl", "hiphop",
    # Quality / rating / hype tags
    "peak", "goated", "goat", "masterpiece", "hot", "hype",
    "fresh", "beautiful", "classic", "special", "good", "great",
    "good stuff", "full", "long", "new", "slaps", "drumless",
    # Mood / descriptor tags (not genre descriptors)
    "atmospheric", "melancholic", "melancholy", "melodic", "energetic",
    "aggressive", "sexual", "romantic", "hedonistic", "playful", "powerful",
    "positive", "warm", "complex", "yearning", "sweet", "soothing", "dreamy",
    "intense", "bombastic", "witty", "heartbreak", "drugs", "nocturnal",
    "emotional", "moody", "boisterous", "rhythmic", "bass", "heavy",
    "funky", "introspective", "sensual", "sexy", "fun", "rage", "slow jams",
    # Geographic tags
    "usa", "american", "texas", "california", "london", "los angeles",
    "toronto", "new jersey", "sweden", "swedish", "british", "canadian",
    "southern", "urban", "united states", "america", "colombian", "alabama",
    "maryland", "ohio", "georgia", "florida", "michigan", "chicago",
    "new york", "detroit", "atlanta", "houston", "brooklyn", "queens",
    "uk", "england", "united kingdom", "australian", "irish", "german", "french", "japanese",
    # Decade tags
    "2020s", "2010s", "90s", "80s", "70s", "60s",
    # Demographic tags
    "male vocalists", "female vocalists", "male vocals", "female vocalist", "diva",
    # Label / industry tags
    "domino", "domino records", "columbia", "interscope", "def jam",
    "atlantic", "republic", "sub pop", "matador", "4ad",
    # Community / random noise
    "happy", "short", "chill", "wikipedia", "fob", "mixtaperoom",
    "xotwod", "windmill scene", "post-darling-core", "elevator music",
    "science", "cover", "bbc", "deposito", "prosty", "synth fumk",
    "pussy music",
}


def is_junk(tag_name: str) -> bool:
    """Return True if a Last.fm tag should be discarded.

    Rejection rules, in order:
      1. Length <= 2 (country abbrevs, stray chars)
      2. Invalid character set (typos, punctuation, emoji)
      3. Starts with a digit (decade tags like "90s", "2020s")
      4. Contains a 4-digit year (community tags like "ch2023")
      5. Exact match in JUNK_EXACT
      6. Substring match in JUNK_SUBSTRINGS
    """
    name = tag_name.strip()
    if len(name) <= 2:
        return True
    if not _VALID_TAG_RE.match(name):
        return True
    if name[0].isdigit():
        return True
    if re.search(r"\d{4}", name):
        return True
    lower = name.lower()
    if lower in JUNK_EXACT:
        return True
    if any(sub in lower for sub in JUNK_SUBSTRINGS):
        return True
    return False


def normalize_tag(name: str) -> str:
    """Lowercase and apply canonical normalizations.

    Uses word-boundary substitution for "rnb" so compound tags like
    "alternative rnb" correctly normalize to "alternative r&b".
    """
    t = name.lower().strip()
    t = t.replace("hip hop", "hip-hop")
    t = re.sub(r"\brnb\b", "r&b", t)
    return t
