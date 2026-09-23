"""Unified reward for mixed GSM8K + MATH OPD training.

GSM8K answers are plain integers extracted from `#### N`; MATH answers are
symbolic (`\\boxed{\\frac{14}{3}}`, `3\\sqrt{13}`, ...). This reward handles
both:

  1. try `#### N` (GSM8K strict) -> numeric compare
  2. else take the last `\\boxed{...}` -> numeric compare if possible,
     otherwise symbolic/string equivalence (the MATH path)

The symbolic comparison is the same sympy-based `is_equiv` validated on
MATH-500 (no antlr4/math_verify dependency). GSM8K's numeric case is a
special case of it, so the existing GSM8K recipe is unaffected: a `#### N`
answer still goes through the exact numeric path.
"""
import re
import signal

_HASH_RE = re.compile(r"####\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_TAIL = 100

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    _HAVE_SYMPY = True
except Exception:
    _HAVE_SYMPY = False


class _timeout:
    def __init__(self, seconds=3):
        self.seconds = seconds

    def _h(self, *_):
        raise TimeoutError

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._h)
        signal.alarm(self.seconds)

    def __exit__(self, *_):
        signal.alarm(0)


def _last_boxed(text):
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
                return text[start + 1:pos]
    return None


def _num(value):
    if value is None:
        return None
    v = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return None


def _clean_latex(s):
    if s is None:
        return None
    s = s.strip().strip("$").strip()
    s = s.replace("\\!", "").replace("\\,", "").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\boxed", "")
    return s


def _to_expr(s):
    if not _HAVE_SYMPY:
        return None
    import re as _re
    t = s
    # \frac{a}{b} -> ((a)/(b))
    for _ in range(3):
        t = _re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", t)
    t = t.replace("\\pi", "pi").replace("\\cdot", "*").replace("^", "**")
    t = t.replace("\\sqrt", "sqrt").replace("{", "(").replace("}", ")").replace("\\", "")
    try:
        return parse_expr(t, evaluate=True)
    except Exception:
        return None


def _equiv(pred, gold):
    if pred is None or gold is None:
        return False
    # numeric path (covers GSM8K exactly)
    np, ng = _num(pred), _num(gold)
    if np is not None and ng is not None:
        return np == ng
    # symbolic / string path (MATH)
    cp, cg = _clean_latex(pred), _clean_latex(gold)
    if cp == cg:
        return True
    if _HAVE_SYMPY:
        try:
            with _timeout(3):
                ep, eg = _to_expr(cp), _to_expr(cg)
                if ep is not None and eg is not None:
                    try:
                        return bool(sympy.simplify(ep - eg) == 0)
                    except Exception:
                        return False
        except Exception:
            return False
    return False


def _extract(solution_str):
    if not solution_str:
        return None
    m = _HASH_RE.findall(solution_str)
    if m:
        return m[-1]
    boxed = _last_boxed(solution_str)
    if boxed is not None:
        return boxed
    # last resort: a trailing number
    tail = re.findall(r"(-?[0-9][0-9,]*(?:\.[0-9]+)?)", solution_str[-_TAIL:])
    return tail[-1] if tail else None


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
    pred = _extract(solution_str)
    gold = ground_truth if not isinstance(ground_truth, dict) else ground_truth.get("answer", ground_truth)
    is_correct = _equiv(pred, gold)

    # Length control: RL on mixed GSM8K+MATH drifts toward ever-longer answers
    # until every rollout hits the response cap (observed: mean length climbing
    # 250 -> 1024 with clip_ratio 0.69 and reward collapsing to ~0). A wrong
    # short answer and a truncated one both score 0, so GRPO gets no signal
    # that length is the problem. Penalise responses that show no answer
    # delimiter AND are long -- the signature of a truncated rollout.
    overlong_penalty = float(kwargs.get("overlong_penalty", 0.0) or 0.0)
    if overlong_penalty and not is_correct:
        has_end = bool(_HASH_RE.search(solution_str)) or (_last_boxed(solution_str) is not None)
        if not has_end and len(solution_str) > int(kwargs.get("overlong_chars", 2000)):
            return -abs(overlong_penalty)

    if not require_hash:
        return score if is_correct else format_score

    # strict-format path, mirrors gsm8k_boxed_reward: reward only when the
    # answer is both correct AND presented in an extractable delimiter
    has_delim = bool(_HASH_RE.search(solution_str)) or (_last_boxed(solution_str) is not None)
    if is_correct and has_delim:
        return score
    if is_correct and correct_without_hash_score is not None:
        return correct_without_hash_score
    if has_delim and hash_format_score is not None:
        return hash_format_score
    return format_score
