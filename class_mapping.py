"""
class_mapping.py
================
Mapping definitions for the 71 Tamil Palm Leaf Vatteluttu character classes.
Handles dataset label IDs ("1" to "71"), Keras model index sorting,
Tamil Unicode character representations, and phonetic transliterations.
"""

from typing import Dict, List, Any

# The 71 classes in alphabetical order as trained by Keras image_dataset_from_directory
KERAS_CLASS_NAMES: List[str] = [
    "1", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "2", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "3", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
    "4", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
    "5", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "6", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
    "7", "70", "71", "8", "9"
]

# Comprehensive metadata mapping for each class label 1..71
# Contains:
# - tamil_char: Modern Tamil Unicode character
# - transliteration: Romanized phonetic form
# - category: Vowel, Consonant, Uyirmei (Syllabic), Diacritic, etc.
TAMIL_VATTELUTTU_MAP: Dict[str, Dict[str, str]] = {
    "1":  {"tamil": "அ",   "translit": "a",    "category": "Uyir (Vowel)"},
    "2":  {"tamil": "ஆ",   "translit": "ā",    "category": "Uyir (Vowel)"},
    "3":  {"tamil": "ம",   "translit": "ma",   "category": "Mei/Uyirmei"},
    "4":  {"tamil": "ஈ",   "translit": "ī",    "category": "Uyir (Vowel)"},
    "5":  {"tamil": "உ",   "translit": "u",    "category": "Uyir (Vowel)"},
    "6":  {"tamil": "ஊ",   "translit": "ū",    "category": "Uyir (Vowel)"},
    "7":  {"tamil": "எ",   "translit": "e",    "category": "Uyir (Vowel)"},
    "8":  {"tamil": "ஏ",   "translit": "ē",    "category": "Uyir (Vowel)"},
    "9":  {"tamil": "ஐ",   "translit": "ai",   "category": "Uyir (Vowel)"},
    "10": {"tamil": "ஒ",   "translit": "o",    "category": "Uyir (Vowel)"},
    "11": {"tamil": "ஓ",   "translit": "ō",    "category": "Uyir (Vowel)"},
    "12": {"tamil": "ஔ",  "translit": "au",   "category": "Uyir (Vowel)"},
    "13": {"tamil": "ஃ",   "translit": "ak",   "category": "Ayutham"},
    "14": {"tamil": "க்",  "translit": "k",    "category": "Mei (Consonant)"},
    "15": {"tamil": "ங்",  "translit": "ṅ",    "category": "Mei (Consonant)"},
    "16": {"tamil": "ச்",  "translit": "c",    "category": "Mei (Consonant)"},
    "17": {"tamil": "ஞ்",  "translit": "ñ",    "category": "Mei (Consonant)"},
    "18": {"tamil": "ட்",  "translit": "ṭ",    "category": "Mei (Consonant)"},
    "19": {"tamil": "ண்",  "translit": "ṇ",    "category": "Mei (Consonant)"},
    "20": {"tamil": "த்",  "translit": "t",    "category": "Mei (Consonant)"},
    "21": {"tamil": "ந்",  "translit": "n",    "category": "Mei (Consonant)"},
    "22": {"tamil": "ப்",  "translit": "p",    "category": "Mei (Consonant)"},
    "23": {"tamil": "ம்",  "translit": "m",    "category": "Mei (Consonant)"},
    "24": {"tamil": "ய்",  "translit": "y",    "category": "Mei (Consonant)"},
    "25": {"tamil": "ர்",  "translit": "r",    "category": "Mei (Consonant)"},
    "26": {"tamil": "ல்",  "translit": "l",    "category": "Mei (Consonant)"},
    "27": {"tamil": "வ்",  "translit": "v",    "category": "Mei (Consonant)"},
    "28": {"tamil": "ழ்",  "translit": "ḻ",    "category": "Mei (Consonant)"},
    "29": {"tamil": "ள்",  "translit": "ḷ",    "category": "Mei (Consonant)"},
    "30": {"tamil": "ற்",  "translit": "ṟ",    "category": "Mei (Consonant)"},
    "31": {"tamil": "ன்",  "translit": "ṉ",    "category": "Mei (Consonant)"},
    "32": {"tamil": "க",   "translit": "ka",   "category": "Uyirmei"},
    "33": {"tamil": "கா",  "translit": "kā",   "category": "Uyirmei"},
    "34": {"tamil": "கி",  "translit": "ki",   "category": "Uyirmei"},
    "35": {"tamil": "கீ",  "translit": "kī",   "category": "Uyirmei"},
    "36": {"tamil": "கு",  "translit": "ku",   "category": "Uyirmei"},
    "37": {"tamil": "கூ",  "translit": "kū",   "category": "Uyirmei"},
    "38": {"tamil": "ச",   "translit": "ca",   "category": "Uyirmei"},
    "39": {"tamil": "சா",  "translit": "cā",   "category": "Uyirmei"},
    "40": {"tamil": "சி",  "translit": "ci",   "category": "Uyirmei"},
    "41": {"tamil": "சீ",  "translit": "cī",   "category": "Uyirmei"},
    "42": {"tamil": "சு",  "translit": "cu",   "category": "Uyirmei"},
    "43": {"tamil": "சூ",  "translit": "cū",   "category": "Uyirmei"},
    "44": {"tamil": "த",   "translit": "ta",   "category": "Uyirmei"},
    "45": {"tamil": "தா",  "translit": "tā",   "category": "Uyirmei"},
    "46": {"tamil": "தி",  "translit": "ti",   "category": "Uyirmei"},
    "47": {"tamil": "தீ",  "translit": "tī",   "category": "Uyirmei"},
    "48": {"tamil": "து",  "translit": "tu",   "category": "Uyirmei"},
    "49": {"tamil": "தூ",  "translit": "tū",   "category": "Uyirmei"},
    "50": {"tamil": "ந",   "translit": "na",   "category": "Uyirmei"},
    "51": {"tamil": "நா",  "translit": "nā",   "category": "Uyirmei"},
    "52": {"tamil": "நி",  "translit": "ni",   "category": "Uyirmei"},
    "53": {"tamil": "நீ",  "translit": "nī",   "category": "Uyirmei"},
    "54": {"tamil": "நு",  "translit": "nu",   "category": "Uyirmei"},
    "55": {"tamil": "நூ",  "translit": "nū",   "category": "Uyirmei"},
    "56": {"tamil": "ப",   "translit": "pa",   "category": "Uyirmei"},
    "57": {"tamil": "பா",  "translit": "pā",   "category": "Uyirmei"},
    "58": {"tamil": "பி",  "translit": "pi",   "category": "Uyirmei"},
    "59": {"tamil": "பீ",  "translit": "pī",   "category": "Uyirmei"},
    "60": {"tamil": "பு",  "translit": "pu",   "category": "Uyirmei"},
    "61": {"tamil": "பூ",  "translit": "pū",   "category": "Uyirmei"},
    "62": {"tamil": "ம",   "translit": "ma",   "category": "Uyirmei"},
    "63": {"tamil": "மா",  "translit": "mā",   "category": "Uyirmei"},
    "64": {"tamil": "மி",  "translit": "mi",   "category": "Uyirmei"},
    "65": {"tamil": "மீ",  "translit": "mī",   "category": "Uyirmei"},
    "66": {"tamil": "மு",  "translit": "mu",   "category": "Uyirmei"},
    "67": {"tamil": "மூ",  "translit": "mū",   "category": "Uyirmei"},
    "68": {"tamil": "ய",   "translit": "ya",   "category": "Uyirmei"},
    "69": {"tamil": "யா",  "translit": "yā",   "category": "Uyirmei"},
    "70": {"tamil": "ர",   "translit": "ra",   "category": "Uyirmei"},
    "71": {"tamil": "ரா",  "translit": "rā",   "category": "Uyirmei"},
}


def get_character_info(class_label: str) -> Dict[str, str]:
    """Retrieve character information for a given class label ('1' to '71')."""
    return TAMIL_VATTELUTTU_MAP.get(
        str(class_label),
        {"tamil": f"[{class_label}]", "translit": f"c_{class_label}", "category": "Unknown"}
    )


def class_index_to_label(index: int) -> str:
    """Map Keras output index (0..70) to dataset class label ('1'..'71')."""
    if 0 <= index < len(KERAS_CLASS_NAMES):
        return KERAS_CLASS_NAMES[index]
    return str(index)
