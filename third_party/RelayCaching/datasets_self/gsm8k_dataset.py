import re
import asyncio
import torch
import contextlib
from typing import Optional

from RelayCaching.llm import LLMChat
from RelayCaching.utils.log import logger

def universal_data_process(dataset):
    """
    Universal math dataset preprocessing function.
    Compatible with:
    1. GSM8K style (Answer field contains "####" separator)
    2. MATH style (Solution field contains \boxed{...} marker)
    """
    list_data_dict = []
    
    for data in dataset:
        task = data.get("problem") or data.get("question")
        raw_solution = data.get("answer") or data.get("solution")
        item = {"task": task}
        
        # Case A: GSM8K style (split by ####)
        if raw_solution and "####" in raw_solution:
            parts = raw_solution.split("####")
            item["step"] = parts[0].strip()
            # GSM8K GT is typically a number (e.g. 1,000), remove commas for comparison
            item["answer"] = parts[-1].replace(",", "").strip()
        
        # Case B: MATH style (answer in \boxed{})
        elif raw_solution and "\\boxed" in raw_solution:
            item["step"] = raw_solution
            gt_content = _extract_boxed_nested(raw_solution)
            if gt_content:
                # Normalize GT: replace \dfrac with \frac, remove commas
                item["answer"] = gt_content.replace(r"\dfrac", r"\frac").replace(",", "").strip()
            else:
                item["answer"] = raw_solution  # fallback
        
        # Case C: Fallback
        else:
            item["step"] = raw_solution
            item["answer"] = str(raw_solution).strip()

    return list_data_dict

try:
    import sympy
    from sympy.parsing import sympy_parser
    from pylatexenc import latex2text
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    logger.warning("sympy or pylatexenc not found. Sympy grading will be skipped (Accuracy may drop).")


def _extract_boxed_nested(text: str) -> str:
    """ 
    Deal with nested brackets in boxed, e.g. \boxed{\dfrac{14}{3}} -> \dfrac{14}{3}
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    
    idx += len("\\boxed{")
    balance = 1
    content = []
    
    while idx < len(text) and balance > 0:
        char = text[idx]
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
        
        if balance > 0:
            content.append(char)
        idx += 1
        
    return "".join(content)

def extract_answer_regex(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    boxed_content = _extract_boxed_nested(text)
    if boxed_content:
        return boxed_content.replace(r"\dfrac", r"\frac").strip()
    
    if "The answer is" in text:
        text = text.split("The answer is")[-1]
    
    # Regex fallback: extract last number/fraction/LaTeX
    pattern = r'-?[\d,]+(?:\.\d+)?(?:/\d+)?|\\frac\{\d+\}\{\d+\}|\\dfrac\{\d+\}\{\d+\}'
    matches = re.findall(pattern, text)
    if matches:
        res = matches[-1].replace(',', '')
        return res.replace(r"\dfrac", r"\frac")
    
    return ""

async def get_pred_value(pred_str):
    """
    Smart extraction entry: Regex first -> LLM fallback
    """
    if isinstance(pred_str, list):
        pred_str = pred_str[0] if pred_str else ""
    
    extracted = extract_answer_regex(pred_str)
    if extracted and len(extracted) < 100: 
        return extracted

    logger.info("Regex extraction failed, using LLM fallback...")
    
    tokenizer = LLMChat._shared_tokenizer
    model = LLMChat._shared_model
    if tokenizer is None or model is None:
        raise RuntimeError("Shared LLM resources not initialised.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in evaluating math answers. "
                "Extract the final numerical answer or expression from the text. "
                "Output ONLY the value (e.g., 140, 3/4, \\frac{1/2}). "
                "Do not format it as a sentence."
            ),
        },
        {"role": "user", "content": f"Text: {pred_str}\n\nFinal Answer:"}, 
    ]

    token_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    
    inputs = {
        "input_ids": token_inputs.to(model.device),
        "attention_mask": torch.ones_like(token_inputs).to(model.device),
    }

    generate_kwargs = {
        "max_new_tokens": 64, 
        "do_sample": False,
    }
    
    outputs = await asyncio.to_thread(model.generate, **inputs, **generate_kwargs)
    generated = outputs.sequences[:, inputs["input_ids"].shape[-1]:]
    response_message = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    
    final_clean = extract_answer_regex(response_message)
    if not final_clean:
        final_clean = response_message 

    logger.opt(colors=True).debug(f"<blue>[LLM Extract]</blue> {final_clean}")
    return final_clean


# -------------------------------------------------------------------------
# Grading Logic (Verification) - DeepScaler / Verl Style
#    Includes Normalization and SymPy Equivalence Check
# -------------------------------------------------------------------------

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"

def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr

def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False

def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False

def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False

def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)

def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))

def _inject_implicit_mixed_number(step: str):
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step) 
    return step

def _sympy_parse(expr: str):
    if not HAS_SYMPY: return None
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )

def _parse_latex(expr: str) -> str:
    if not HAS_SYMPY: return expr
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    try:
        expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    except:
        pass 
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()

def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)

def should_allow_eval(expr: str):
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True

def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    if not HAS_SYMPY: return False
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal

def split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems

def _normalize(expr: str) -> str:
    """Normalize answer expressions (DeepScaler Version)."""
    if expr is None: return None

    # Helper functions defined locally to ensure closure
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        return new_str

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2: return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            return "\\frac{" + str(a) + "}{" + str(b) + "}"
        except Exception:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string: return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None: expr = m.group("text")

    expr = expr.replace("\\%", "%").replace("\\$", "$").replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")

    for unit in ["degree", "cm", "centimeter", "meter", "mile", "second", "minute", "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard"]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}": expr = expr[1:-1]
    expr = re.sub(",\\\\! *", "", expr)
    
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    
    if "\\" in expr:
        try: expr = _parse_latex(expr)
        except: pass

    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    
    expr = _fix_sqrt(expr)
    expr = _fix_fracs(expr)
    if expr == "0.5": expr = "\\frac{1}{2}"
    expr = _fix_a_slash_b(expr)

    return expr

def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    """Uses SymPy to check for mathematical equivalence."""
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None: return False
    if ground_truth_normalized == given_normalized: return True
    if len(given_normalized) == 0: return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) != len(given_elems):
        return False
    
    for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
        if _is_frac(ground_truth_elem) and _is_frac(given_elem):
            is_correct = ground_truth_elem == given_elem
        elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
            is_correct = False
        else:
            is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
        if not is_correct:
            return False

    return True

def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    """Uses simple string normalization (Math Dataset style)."""
    gt_norm = _normalize(ground_truth)
    given_norm = _normalize(given_answer)
    return gt_norm == given_norm

# -------------------------------------------------------------------------
# Main Entry
# -------------------------------------------------------------------------

def check_correctness(model_output: str, ground_truth: str) -> bool:
    """
    Unified correctness verification entry.

    Args:
        model_output: Clean string extracted via get_pred_value (e.g. "14/3")
        ground_truth: Ground truth answer from dataset (e.g. "14/3" or "4.666")
    """
    
    if grade_answer_mathd(model_output, ground_truth):
        return True
    
    if HAS_SYMPY and grade_answer_sympy(model_output, ground_truth):
        return True
    
    try:
        val_pred = float(eval(_normalize(model_output)))
        val_gt = float(eval(_normalize(ground_truth)))
        if abs(val_pred - val_gt) < 1e-6:
            return True
    except:
        pass
        
    return False