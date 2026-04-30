"""
tag_filters.py
Filtering and normalization for Last.fm tags.
"""

import re

# Allowed characters in a clean tag
_VALID_TAG_RE = re.compile(r"^[a-zA-Z0-9 \-&]+$")

# Substring checks (case-insensitive)
JUNK_SUBSTRINGS = [
    "seen live", "favourite", "favorites", "under 2000 listeners", "best of",
    "album of the year", "i love", "i like", "i can", "i own", "where is",
    "my top", "fav ", "cum", "goodness"
]

# Exact-match tags to discard
JUNK_EXACT = {
    "indie", "alternative", "progressive", "aoty", "albums", "all",
    "classic albums", "vinyl", "hiphop", "peak", "goated", "goat",
    "masterpiece", "hot", "hype", "fresh", "beautiful", "classic",
    "special", "good", "great", "good stuff", "full", "long", "new",
    "slaps", "drumless", "atmospheric", "melancholic", "melancholy",
    "melodic", "energetic", "aggressive", "sexual", "romantic",
    "hedonistic", "playful", "powerful", "positive", "warm", "complex",
    "yearning", "sweet", "soothing", "dreamy", "intense", "bombastic",
    "witty", "heartbreak", "drugs", "nocturnal", "emotional", "moody",
    "boisterous", "rhythmic", "bass", "heavy", "funky", "introspective",
    "sensual", "sexy", "fun", "rage", "slow jams", "usa", "american",
    "texas", "california", "london", "los angeles", "toronto", "new jersey",
    "sweden", "swedish", "british", "canadian", "southern", "urban",
    "united states", "america", "colombian", "alabama", "maryland", "ohio",
    "georgia", "florida", "michigan", "chicago", "new york", "detroit",
    "atlanta", "houston", "brooklyn", "queens", "uk", "england",
    "united kingdom", "australian", "irish", "german", "french", "japanese",
    "2020s", "2010s", "90s", "80s", "70s", "60s", "male vocalists",
    "female vocalists", "male vocals", "female vocalist", "diva", "domino",
    "domino records", "columbia", "interscope", "def jam", "atlantic",
    "republic", "sub pop", "matador", "4ad", "happy", "short", "chill",
    "wikipedia", "fob", "mixtaperoom", "xotwod", "windmill scene",
    "post-darling-core", "elevator music", "science", "cover", "bbc",
    "deposito", "prosty", "synth fumk", "pussy music"
}

def is_junk(tag_name: str) -> bool:
    """Return True if a tag should be discarded."""
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
    """Lowercase and apply standard normalizations."""
    t = name.lower().strip()
    t = t.replace("hip hop", "hip-hop")
    t = re.sub(r"\brnb\b", "r&b", t)
    return t
