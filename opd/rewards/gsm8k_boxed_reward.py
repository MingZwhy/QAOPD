"""Format-tolerant GSM8K reward for Qwen reasoning responses."""

import re

_TAIL_CHARS = 400
_HASH_ANSWER_RE = re.compile(r"#### (-?[0-9][0-9,.]*)")


def _num_from(text):
    if not text:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def _extract_boxed(text):
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    start = text.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos]
    return None


def _extract_answer(solution_str):
    if not solution_str:
        return None
    strict = re.findall(r"####\s*\$?\s*(-?[0-9][0-9,.]*)", solution_str)
    if strict:
        return _num_from(strict[-1])
    boxed = _extract_boxed(solution_str)
    if boxed is not None:
        answer = _num_from(boxed)
        if answer is not None:
            return answer
    return _num_from(solution_str[-_TAIL_CHARS:])


def _normalize(value):
    if value is None:
        return None
    value = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except (TypeError, ValueError):
        return value


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    score=1.0,
    format_score=0.0,
    require_hash=False,
    correct_without_hash_score=None,
    hash_format_score=None,
    **kwargs,
):
    prediction = _normalize(_extract_answer(solution_str))
    if prediction is None:
        return 0.0
    is_correct = prediction == _normalize(ground_truth)
    if not require_hash:
        return score if is_correct else format_score

    has_hash_answer = bool(_HASH_ANSWER_RE.search(solution_str))
    if is_correct and has_hash_answer:
        return score
    if is_correct and correct_without_hash_score is not None:
        return correct_without_hash_score
    if has_hash_answer and hash_format_score is not None:
        return hash_format_score
    return format_score
