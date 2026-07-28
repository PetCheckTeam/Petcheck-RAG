import re
from typing import Optional

START_PATTERNS = [
    r"사용한\s*원료의\s*명칭",
    r"원료의\s*명칭",
    r"\bINGREDIENTS\b",
]

SOFT_END_PATTERNS = [
    r"주의\s*사항",
    r"등록\s*성분\s*량",
    r"보관\s*방법",
    r"유통\s*기한",
    r"제조\s*년월일",
    r"수입\s*업자",
    r"판매\s*원",
    r"\bGUARANTEED\s+ANALYSIS\b",
    r"\bFEEDING\s+GUIDE\b",
    r"\bDIRECTIONS\b",
    r"\bCAUTION\b",
    r"\b성분량\b",
    r"등록\s*성분\s*량",
    r"\n\s*인\s*\n?\s*성분량",
    r"\s인\s성분량",
]

SENTENCE_TAIL_STARTERS = [
    "주의사항",
    "주의 사항",
    "개봉 후",
    "보관",
    "직사광선",
    "유통기한",
    "제조년월일",
    "수입업자",
    "판매원",
    "전화번호",
    "www",
    "http",
    "Guaranteed Analysis",
    "Feeding Guide",
    "Directions",
    "Caution",
    "Store ",
    "Keep ",
    "Feed ",
    "Use ",
    "Refrigerate ",
    "성분량",
    " 인 성분량",
    "등록성분량",
    "등록 성분량",
]


def read_text_file(path: str) -> str:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        "파일 인코딩을 utf-8, cp949, euc-kr 중 어떤 것으로도 읽을 수 없습니다.",
    )


def skip_english_instruction_noise(section: str) -> str:
    lines = [line.strip() for line in section.split("\n") if line.strip()]
    for index, line in enumerate(lines):
        cleaned_line = clean_ingredient_name(line)
        if looks_like_english_ingredient_list_start(cleaned_line):
            return " ".join(lines[index:])
    return section


def looks_like_english_ingredient_list_start(line: str) -> bool:
    comma_count = line.count(",")
    if comma_count >= 3:
        return True

    common_ingredient_words = [
        "beef",
        "chicken",
        "salmon",
        "whitefish",
        "liver",
        "rice",
        "corn",
        "flaxseed",
        "sweet potato",
        "egg",
        "pea flour",
        "apple",
        "blueberry",
        "carrot",
        "pumpkin",
        "spinach",
        "kelp",
    ]
    lower_line = line.lower()
    matched_count = sum(1 for word in common_ingredient_words if word in lower_line)
    return matched_count >= 2


def extract_ingredients(ocr_text: str) -> list[str]:
    section, _ = extract_ingredient_section(ocr_text)
    return split_ingredients(section)


def extract_ingredient_details(ocr_text: str) -> dict[str, object]:
    section, method = extract_ingredient_section(ocr_text)
    ingredients = split_ingredients(section)
    return {
        "method": method,
        "section": section,
        "ingredients": ingredients,
        "confidence": calculate_extraction_confidence(method, ingredients),
    }

def extract_ingredient_section(
    ocr_text: str, max_chars: int = 1200
) -> tuple[str, str]:
    text = normalize_ocr_text(ocr_text)
    
    # 💡 안전하게 예외 처리를 감싸서 500 에러를 원천 차단합니다.
    try:
        start_match = find_start(text)
    except Exception:
        start_match = None

    if not start_match:
        section = text.strip()
        method = "FULL_TEXT_FALLBACK"
    else:
        section = text[start_match.end() :].strip()
        method = "START_FOUND"

        try:
            if start_match.group(0).lower() == "ingredients":
                section = skip_english_instruction_noise(section)
        except Exception:
            pass

    soft_end = find_soft_end(section)
    if soft_end is not None:
        return section[:soft_end].strip(), "SOFT_END_FOUND"

    return section[:max_chars].strip(), method


def find_soft_end(text: str) -> Optional[int]:
    positions = []
    for pattern in SOFT_END_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            positions.append(match.start())

    if not positions:
        return None

    return min(positions)


def split_ingredients(section_text: str) -> list[str]:
    if not section_text:
        return []

    text = normalize_separators(section_text)
    candidates = split_by_comma_outside_parentheses(text)
    ingredients = []

    for candidate in candidates:
        item = clean_ingredient_name(candidate)
        item = trim_sentence_tail(item)
        item = clean_ingredient_name(item)

        if not item:
            continue
        if is_sentence_like(item):
            break
        ingredients.append(item)

    return ingredients


def split_by_comma_outside_parentheses(text: str) -> list[str]:
    result = []
    current = []
    depth = 0

    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]" and depth > 0:
            depth -= 1

        if char == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        result.append("".join(current).strip())

    return result


def trim_sentence_tail(text: str) -> str:
    lower_text = text.lower()
    positions = []

    for starter in SENTENCE_TAIL_STARTERS:
        index = lower_text.find(starter.lower())
        if index > 0:
            positions.append(index)

    if positions:
        return text[: min(positions)].strip(" ,.;:：-")

    return text


def is_sentence_like(text: str) -> bool:
    text = text.strip()
    lower_text = text.lower()

    if not text or len(text) >= 70:
        return True

    korean_sentence_endings = [
        "합니다",
        "하십시오",
        "바랍니다",
        "됩니다",
        "주세요",
        "있습니다",
        "없습니다",
    ]
    for ending in korean_sentence_endings:
        if ending in text:
            return True

    english_sentence_starts = [
        "store ",
        "keep ",
        "feed ",
        "use ",
        "refrigerate ",
        "serve ",
        "provide ",
        "consult ",
        "not for ",
        "for animal ",
        "this product ",
    ]
    for start in english_sentence_starts:
        if lower_text.startswith(start):
            return True

    word_count = len(text.split())
    if "." in text and word_count >= 6:
        return True

    korean_sentence_markers = [
        " 후 ",
        " 시 ",
        " 또는 ",
        " 및 ",
        "하며",
        "하고",
        "하여",
        "되어",
        "있는",
        "없는",
    ]
    marker_count = sum(1 for marker in korean_sentence_markers if marker in text)
    if marker_count >= 2 and len(text) >= 35:
        return True

    return False


def clean_ingredient_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"^[\s\]\[\):：,.;\-]+", "", name)
    name = re.sub(r"\s*인\s*성분량\s*$", "", name)
    name = re.sub(r"\s*성분량\s*$", "", name)
    name = re.sub(r"[\s,.;:：-]+$", "", name)
    return name


def normalize_ocr_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def normalize_separators(text: str) -> str:
    text = text.replace("ㆍ", ",")
    text = text.replace("·", ",")
    text = text.replace("•", ",")
    text = text.replace(";", ",")
    text = text.replace("\n", " ")
    return text


def calculate_extraction_confidence(method: str, ingredients: list[str]) -> float:
    if not ingredients:
        return 0.0
    score = 0.7
    if method == "SOFT_END_FOUND":
        score += 0.2
    if method == "MAX_CHARS_FALLBACK":
        score -= 0.1
    if len(ingredients) >= 3:
        score += 0.1
    if len(ingredients) >= 8:
        score += 0.05
    return min(score, 1.0)