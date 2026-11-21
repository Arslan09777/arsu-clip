"""Romanize Hindi captions while leaving surrounding English and timing intact."""
import re
import unicodedata
from functools import lru_cache


_WORD = re.compile(r"[\u0900-\u0963\u0970-\u097f]+")
_MARKS = str.maketrans("०१२३४५६७८९।॥", "0123456789..")
_CONSONANTS = dict(zip(
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ",
    "k kh g gh ng ch chh j jh ny t th d dh n t th d dh n p ph b bh m y r l v sh sh s h l".split(),
))
_NUKTA = dict(zip("कखगजडढफय", ("q", "kh", "gh", "z", "r", "rh", "f", "y")))
_VOWELS = dict(zip("अआइईउऊऋऌऍऎएऐऑऒओऔ", "a aa i ee u oo ri li e e e ai o o o au".split()))
_MATRAS = dict(zip("ािीुूृॢॅॆेैॉॊोौ", "aa i ee u oo ri li e e e ai o o o au".split()))
_SOUNDS = {"ँ": "n", "ं": "n", "ः": "h", "ऽ": "'", "़": "", "्": ""}
_COMMON = {
    "और": "aur", "अब": "ab", "इधर": "idhar", "उनको": "unko",
    "क्या": "kya", "क्यों": "kyun", "क्योंकि": "kyunki", "कुछ": "kuch",
    "कर": "kar", "करना": "karna", "कहना": "kehna", "कैसा": "kaisa", "कैसे": "kaise",
    "गया": "gaya", "गई": "gayi", "गेम": "game", "गेमिंग": "gaming",
    "चाहिए": "chahiye", "ठीक": "theek", "तुम्हारा": "tumhara", "नहीं": "nahi",
    "पहले": "pehle", "प्लीज": "please", "फिर": "phir", "फाइनल": "final",
    "बहुत": "bahut", "भाई": "bhai", "भैया": "bhaiya", "मैं": "main", "में": "mein",
    "मुझे": "mujhe", "ये": "ye", "यह": "yeh", "यहां": "yahan", "यहाँ": "yahan",
    "यार": "yaar", "रहा": "raha", "रही": "rahi", "रहे": "rahe",
    "लाइव": "live", "वीडियो": "video", "वो": "woh", "सही": "sahi",
    "है": "hai", "हैं": "hain", "हूं": "hoon", "हूँ": "hoon", "हो": "ho",
    "ग्रैंड": "Grand", "थेफ्ट": "Theft", "ऑटो": "Auto", "जीटीए": "GTA",
    "सब्सक्राइब": "subscribe", "लाइक": "like", "चैट": "chat", "स्ट्रीम": "stream",
}


@lru_cache(maxsize=8_192)
def _romanize_word(word: str) -> str:
    if word in _COMMON:
        return _COMMON[word]

    units: list[tuple[str, str, bool]] = []
    pos = 0
    while pos < len(word):
        char = word[pos]
        pos += 1
        if char not in _CONSONANTS:
            units.append((_VOWELS.get(char, _SOUNDS.get(char, char)), "", False))
            continue

        consonant = _CONSONANTS[char]
        if word[pos:pos + 1] == "़":
            consonant = _NUKTA.get(char, consonant)
            pos += 1
        following = word[pos:pos + 1]
        vowel = "a"
        inherent = True
        if following in _MATRAS or following == "्":
            vowel = _MATRAS.get(following, "")
            inherent = False
            pos += 1
        units.append((consonant, vowel, inherent))

    last = len(units) - 1
    if last > 0 and units[last][2]:
        units[last] = (units[last][0], "", False)
    if last >= 2 and units[last][1] and not units[last][2] and units[last - 1][2]:
        # Hindi often suppresses the schwa before a final marked vowel: खेलना → khelna.
        units[last - 1] = (units[last - 1][0], "", False)
    if units[last][1] == "aa":
        units[last] = (units[last][0], "a", False)
    return "".join(syllable + vowel for syllable, vowel, _ in units)


def romanize_hindi(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_MARKS)
    text = _WORD.sub(lambda match: _romanize_word(match.group()), text)
    return text.replace("\u200c", "").replace("\u200d", "")
