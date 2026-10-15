"""Small, versioned French spoken-text normalizer shared with future inference."""
from datetime import date
from decimal import Decimal
import re
import unicodedata

from num2words import num2words

VERSION = "fr-baseline-1"
MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
          "août", "septembre", "octobre", "novembre", "décembre")
UNITS = {
    "km/h": ("kilomètre par heure", "kilomètres par heure"),
    "km²": ("kilomètre carré", "kilomètres carrés"),
    "m²": ("mètre carré", "mètres carrés"),
    "km": ("kilomètre", "kilomètres"), "cm": ("centimètre", "centimètres"),
    "mm": ("millimètre", "millimètres"), "m": ("mètre", "mètres"),
    "kg": ("kilogramme", "kilogrammes"), "g": ("gramme", "grammes"),
    "ml": ("millilitre", "millilitres"), "l": ("litre", "litres"),
    "°C": ("degré Celsius", "degrés Celsius"),
    "Hz": ("hertz", "hertz"), "kHz": ("kilohertz", "kilohertz"),
    "MHz": ("mégahertz", "mégahertz"), "%": ("pour cent", "pour cent"),
    "min": ("minute", "minutes"), "s": ("seconde", "secondes"),
    "ms": ("milliseconde", "millisecondes"),
}


def cardinal(number):
    return num2words(int(number), lang="fr")


def normalize_sentence(text):
    """Return spoken text and applied rules. Preserve original text in metadata.

    This normalizes TEXT, never IPA: nasal combining marks must stay intact.
    Dates, hours, euro amounts, common units/abbreviations and digits are bounded
    rules, not a complete French NLP system. Ambiguous formulas/URLs are rejected
    by the corpus builder before reaching this function.
    """
    changes = []
    original = text
    text = unicodedata.normalize("NFC", text).replace("’", "'").replace("−", "-")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b([A-Za-zÀ-ÿ])'\s+(?=\w)", r"\1'", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    if text != original:
        changes.append("unicode_whitespace_punctuation")

    def apply(pattern, replacement, name, flags=0):
        nonlocal text
        value, count = re.subn(pattern, replacement, text, flags=flags)
        if count and value != text:
            changes.append(name)
        text = value

    apply(r"\[(?:\d+(?:[,–-]\d+)*|réf\.? nécessaire|citation needed)\]", "", "references", re.I)
    # Join French thousands separators before reading numbers (not date slashes).
    apply(r"(?<=\d) (?=\d{3}(?:\D|$))", "", "thousands_separator")

    def spoken_date(match):
        day, month, year = map(int, match.groups())
        date(year, month, day)  # invalid calendar dates must not silently survive
        return f"{'premier' if day == 1 else cardinal(day)} {MONTHS[month - 1]} {cardinal(year)}"
    apply(r"(?<!\w)(\d{1,2})/(\d{1,2})/(\d{4})(?!\w)", spoken_date, "date")

    def spoken_hour(match):
        hour = int(match[1])
        minute = int(match[2]) if match[2] is not None else 0
        if hour > 23 or minute > 59:
            raise ValueError("invalid clock time")
        return ("une heure" if hour == 1 else "vingt et une heures" if hour == 21 else f"{cardinal(hour)} heures") + (
            f" {cardinal(minute)}" if minute else "")
    apply(r"(?<!\w)(\d{1,2})\s*h(?:\s*(\d{1,2}))?(?!\w)", spoken_hour, "hour")
    apply(r"(?<!\w)(\d{1,2}):(\d{2})(?!\w)", spoken_hour, "hour")
    apply(r"(?<!\w)-(?=\d)", "moins ", "negative_number")
    apply(r"\b1 (?=(?:" + "|".join(MONTHS) + r")\b)", "premier ", "date", re.I)
    apply(r"\b1 (?=(?:minute|seconde|heure|année|personne|femme|fille|fois|place)\b)", "une ", "feminine_one")

    def euro(match):
        value = Decimal(match[1].replace(",", "."))
        euros = int(value)
        cents = int((value - euros) * 100)
        result = f"{cardinal(euros)} {'euro' if euros == 1 else 'euros'}"
        if cents:
            result += f" et {cardinal(cents)} {'centime' if cents == 1 else 'centimes'}"
        return result
    apply(r"(?<![\w.,])(\d+(?:[,.]\d{1,2})?)\s*(?:€|euros?\b)", euro, "euro")

    unit_choices = "|".join(re.escape(k) for k in sorted(UNITS, key=len, reverse=True))
    def unit(match):
        value, abbreviation = match.groups()
        singular = Decimal(value.replace(",", ".")) == 1
        if abbreviation in {"min", "s", "ms"} and singular:
            value = "une"
        return f"{value} {UNITS[abbreviation][0 if singular else 1]}"
    apply(r"(?<![\w.,])(\d+(?:[,.]\d+)?)\s*(" + unit_choices + r")(?!\w)", unit, "unit")

    for pattern, replacement in (
        (r"\bM\.(?=\s+[A-ZÀ-Ý])", "Monsieur"),
        (r"\bMme\.?(?!\w)", "Madame"), (r"\bMmes\.?(?!\w)", "Mesdames"),
        (r"\bDr\.?(?!\w)", "Docteur"), (r"\bPr\.?(?!\w)", "Professeur"),
        (r"\bSt\.(?=\s|-)" , "Saint"), (r"\bSte\.(?=\s|-)" , "Sainte"),
        (r"\bn[°º]\s*(?=\d)", "numéro "),
    ):
        apply(pattern, replacement, "abbreviation")

    apply(r"(?<!\w)(\d+)(er|re|e|ème|eme|ᵉ)(?!\w)",
          lambda m: "première" if m[1] == "1" and m[2] == "re" else num2words(int(m[1]), to="ordinal", lang="fr"), "ordinal")

    def number(match):
        value = match[0]
        if "," in value or "." in value:
            whole, fraction = re.split(r"[,.]", value)
            # Each decimal digit is explicit: keeps leading/trailing zeros.
            return cardinal(whole) + " virgule " + " ".join(cardinal(c) for c in fraction)
        return cardinal(value)
    apply(r"(?<!\w)\d+(?:[,.]\d+)?(?!\w)", number, "number")
    apply(r"\bde (un|une)\b", lambda m: "d'" + m[1], "numeric_elision")
    return re.sub(r"\s+", " ", text).strip(), list(dict.fromkeys(changes))
