#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    import os

    from langchain_core.messages import SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Create a .env file in the repo root "
            "containing one line: DEEPSEEK_API_KEY=sk-your-key"
        )

    # Stage 1 model: deterministic decoding, bounded retries so a transient
    # network error can never hang or crash the whole run.
    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_retries=2,
        timeout=120,
    )

    extraction_system = (
        "You are a precise Hong Kong supermarket receipt parser. Read the "
        "receipt image and return ONLY one JSON object. No prose, no markdown "
        "code fences, no explanation.\n\n"
        "Rules:\n"
        "1. Copy numbers exactly as printed; use plain numbers with 2 decimals.\n"
        '2. "items": every product line as {"name": <str>, "price": <number>}, '
        "where price is the ORIGINAL price of that line as printed.\n"
        '3. "discounts": every promotion / coupon / member / app / '
        'packaging-damage / percentage discount line as {"label": <str>, '
        '"amount": <number>}. Keep the sign exactly as printed (a reduction is '
        "normally negative).\n"
        '4. "subtotal": the printed SUBTOTAL (or \u5c0f\u8a08 / \u5408\u8a08) line. It is the '
        "amount AFTER discounts but BEFORE any rounding adjustment. Use 0 if absent.\n"
        '5. "rounding": the printed ROUNDING (or \u8abf\u6574 / \u56db\u820d\u4e94\u5165) line, e.g. '
        "-0.01. Use 0 if absent.\n"
        '6. "amount_paid": the FINAL amount actually paid, taken from the '
        "payment line (OCTOPUS / CASH / CARD / EPS / \u5be6\u4ed8 / \u4ed8\u6b3e). If several "
        "payment lines are shown, give their sum. If there is no payment line, "
        "use SUBTOTAL plus ROUNDING.\n"
        "7. Do NOT compute, sum, or invent any value; only report what is printed.\n"
        "8. If a field is missing, use 0.\n\n"
        "Exact output shape:\n"
        '{"items": [{"name": "MILK", "price": 10.0}], "discounts": '
        '[{"label": "5% OFF", "amount": -5.39}], "subtotal": 102.31, '
        '"rounding": -0.01, "amount_paid": 102.3}'
    )

    # Stage 1: one vision call per receipt -> raw text (parsed deterministically
    # in answer_queries so that malformed output can still be repaired).
    # The system text contains a literal JSON shape, so pass it as a fixed
    # SystemMessage: ChatPromptTemplate would otherwise read "{" as a variable.
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=extraction_system),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Parse this receipt image and return the JSON object.",
                    },
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                ],
            ),
        ]
    )

    # Stage 2: conditional repair branch, used only when Stage 1 output is not
    # valid JSON. Keeps the expensive vision pass from being wasted.
    repair_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You repair malformed JSON. Output ONLY a valid JSON object "
                "with the keys items, discounts, subtotal, rounding and "
                "amount_paid. No prose and no markdown code fences.",
            ),
            (
                "human",
                "The text below should be a receipt JSON record but is "
                "malformed. Fix it and return valid JSON only.\n\n{raw}",
            ),
        ]
    )

    # Stage 2b: a second, deliberately narrow reading of the same image. It
    # ignores products and asks only for the three numbers that drive the two
    # answers, so it fails independently of the full extraction.
    totals_system = (
        "You are a precise Hong Kong supermarket receipt reader. Read the "
        "receipt image and return ONLY one JSON object. No prose, no markdown "
        "code fences.\n\n"
        "Steps, in this order:\n"
        '1. Copy out EVERY promotion / coupon / member / app / packaging-damage '
        '/ percentage discount line you can see, one by one, into '
        '"discount_lines". Read each amount digit by digit; receipts often '
        "carry several separate discount lines. Give each amount as a POSITIVE "
        "number (drop the minus sign). Use an empty list if there is none.\n"
        '2. "subtotal": the printed SUBTOTAL (or \u5c0f\u8a08 / \u5408\u8a08) line. It is the '
        "amount AFTER discounts but BEFORE any rounding adjustment. Use 0 if absent.\n"
        '3. "amount_paid": the FINAL amount actually paid, from the payment '
        "line (OCTOPUS / CASH / CARD / EPS / \u5be6\u4ed8 / \u4ed8\u6b3e). Sum several payment "
        "lines if shown; if there is no payment line use SUBTOTAL plus ROUNDING.\n\n"
        "Do not list products and do not report any other number.\n"
        'Output shape: {"discount_lines": [{"label": "5% OFF", "amount": 5.39}], '
        '"discount_total": 5.39, "subtotal": 102.31, "amount_paid": 102.3}'
    )
    totals_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=totals_system),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Read the three numbers from this receipt image.",
                    },
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                ],
            ),
        ]
    )

    # Stage 2c: adjudication, used only when the two readings disagree.
    adjudicate_system = (
        "You are re-checking a supermarket receipt because two independent "
        "readings of it disagreed. Re-read the image carefully, line by line, "
        "and return the correct values as ONE JSON object. No prose, no "
        "markdown code fences.\n\n"
        '"subtotal" is the printed SUBTOTAL line (after discounts, before '
        "rounding). \"discount_total\" is the sum of every promotion / coupon / "
        "member / app / packaging-damage / percentage discount line, as a "
        'POSITIVE number. "amount_paid" is the final payment line.'
    )
    adjudicate_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=adjudicate_system),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            "Reading A: subtotal={a_subtotal}, "
                            "discount_total={a_discount}, amount_paid={a_paid}.\n"
                            "Reading B: subtotal={b_subtotal}, "
                            "discount_total={b_discount}, amount_paid={b_paid}.\n"
                            "At most one of them is right. Re-read the receipt "
                            'and answer: {{"subtotal": <number>, '
                            '"discount_total": <positive number>, '
                            '"amount_paid": <number>}}'
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                ],
            ),
        ]
    )

    return {
        "extract": extract_prompt | model | StrOutputParser(),
        "totals": totals_prompt | model | StrOutputParser(),
        "adjudicate": adjudicate_prompt | model | StrOutputParser(),
        "repair": repair_prompt | model | StrOutputParser(),
    }


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    import os
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from decimal import ROUND_HALF_UP

    debug = bool(os.environ.get("HW1_DEBUG"))
    cents = Decimal("0.01")
    zero = Decimal("0")

    extract = chain["extract"]
    totals = chain["totals"]
    adjudicate = chain["adjudicate"]
    repair = chain["repair"]
    tolerance = Decimal("0.02")

    def to_decimal(value: Any) -> Decimal:
        """Turn any model-provided amount into a Decimal, never raising."""
        if value is None or isinstance(value, bool):
            return zero
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            try:
                return Decimal(str(value))
            except InvalidOperation:
                return zero
        text = str(value).strip().replace(",", "")
        text = text.replace("HK$", "").replace("HKD", "").replace("$", "")
        text = text.replace("(", "-").replace(")", "")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return zero
        try:
            return Decimal(match.group(0))
        except InvalidOperation:
            return zero

    def json_object(text: Any) -> dict | None:
        """Best-effort JSON object extraction from a raw model string."""
        if not isinstance(text, str) or not text.strip():
            return None
        candidate = text.strip()
        candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
        candidate = re.sub(r"```$", "", candidate).strip()
        for snippet in (candidate,):
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, TypeError):
                pass
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def amounts(record: dict) -> dict:
        """Normalise one parsed record into the numbers the queries need."""
        raw_items = record.get("items") or []
        raw_discounts = record.get("discounts") or []
        items = [i for i in raw_items if isinstance(i, dict)]
        discounts = [d for d in raw_discounts if isinstance(d, dict)]

        items_sum = sum((to_decimal(i.get("price")) for i in items), zero)
        # Discounts keep their printed sign (negative); both views are needed.
        discounts_signed = sum((to_decimal(d.get("amount")) for d in discounts), zero)
        discounts_abs = sum((abs(to_decimal(d.get("amount"))) for d in discounts), zero)

        subtotal = to_decimal(record.get("subtotal"))
        rounding = to_decimal(record.get("rounding"))
        paid = to_decimal(record.get("amount_paid"))
        reported_discount = to_decimal(record.get("discount_total"))
        raw_lines = record.get("discount_lines") or []
        lines = [d for d in raw_lines if isinstance(d, dict)]

        # Prefer summing the enumerated discount lines ourselves: it keeps the
        # model out of the arithmetic. Fall back to the total it reported.
        if lines:
            discounts_abs = sum((abs(to_decimal(d.get("amount"))) for d in lines), zero)
            discounts_signed = -discounts_abs
        elif reported_discount != zero:
            discounts_abs = abs(reported_discount)
            discounts_signed = -discounts_abs

        # Fallbacks keep a partially parsed receipt useful instead of dropping it.
        if subtotal == zero and items_sum != zero:
            subtotal = items_sum + discounts_signed
        if paid == zero:
            paid = subtotal + rounding

        # Query 2 = SUBTOTAL + every discount added back as a positive number.
        # ROUNDING is deliberately NOT added back.
        without_discount = subtotal + discounts_abs
        if without_discount == zero:
            without_discount = items_sum

        return {
            "paid": paid,
            "without_discount": without_discount,
            "subtotal": subtotal,
            "discount_total": discounts_abs,
            # On any receipt, SUBTOTAL plus the discounts must equal the sum of
            # the original item prices, so this is a free independent estimate.
            "items_sum": items_sum,
        }

    def invoke_text(target, payload, label: str, name: str) -> str | None:
        """Call a chain and return raw text, retrying once, never raising."""
        for _attempt in range(2):
            try:
                return target.invoke(payload)
            except Exception as error:  # network / API / rate limit
                if debug:
                    print(f"[hw1] {name}: {label} failed: {error}", file=sys.stderr)
        return None

    def read_record(target, payload, label: str, name: str) -> dict | None:
        """One vision reading: invoke, repair malformed JSON, normalise."""
        raw = invoke_text(target, payload, label, name)
        if raw is None:
            return None
        record = json_object(raw)
        if record is None:
            fixed = invoke_text(
                repair, {"raw": raw if isinstance(raw, str) else str(raw)}, "repair", name
            )
            record = json_object(fixed)
        if record is None:
            return None
        return amounts(record)

    def median(values: list[Decimal]) -> Decimal:
        ordered = sorted(values)
        size = len(ordered)
        if size % 2:
            return ordered[size // 2]
        return (ordered[size // 2 - 1] + ordered[size // 2]) / 2

    def disagrees(values: list[Decimal]) -> bool:
        return len(values) == 2 and (max(values) - min(values)) > tolerance

    def parse_receipt(path: Path) -> tuple[Decimal, Decimal] | None:
        """Stage 1-2 for one receipt: two readings, adjudicate if they clash."""
        try:
            payload = {"image_url": image_data_url(path)}
        except Exception:
            return None

        reading_a = read_record(extract, payload, "extract", path.name)
        reading_b = read_record(totals, payload, "totals", path.name)
        readings = [r for r in (reading_a, reading_b) if r is not None]
        if not readings:
            return None

        paid_values = [r["paid"] for r in readings if r["paid"] != zero]
        nodisc_values = [
            r["without_discount"] for r in readings if r["without_discount"] != zero
        ]

        # Tie-break for query 2 without spending another call: the sum of the
        # original item prices must equal SUBTOTAL plus the discounts, so it
        # votes for whichever reading satisfies that identity.
        items_sum = reading_a["items_sum"] if reading_a is not None else zero
        if disagrees(nodisc_values) and items_sum != zero:
            matches = [v for v in nodisc_values if abs(items_sum - v) <= tolerance]
            if len(matches) == 1:
                if debug:
                    print(
                        f"[hw1] {path.name}: item prices ({items_sum}) settle "
                        f"the disagreement -> {matches[0]}",
                        file=sys.stderr,
                    )
                nodisc_values = matches

        # Reflection step: still unresolved (or the payment disagrees), so send
        # the image back with both candidates and ask for one careful re-read.
        if reading_a is not None and reading_b is not None and (
            disagrees(paid_values) or disagrees(nodisc_values)
        ):
            third = read_record(
                adjudicate,
                {
                    "image_url": payload["image_url"],
                    "a_subtotal": reading_a["subtotal"],
                    "a_discount": reading_a["discount_total"],
                    "a_paid": reading_a["paid"],
                    "b_subtotal": reading_b["subtotal"],
                    "b_discount": reading_b["discount_total"],
                    "b_paid": reading_b["paid"],
                },
                "adjudicate",
                path.name,
            )
            if third is not None:
                if third["paid"] != zero:
                    paid_values.append(third["paid"])
                if third["without_discount"] != zero:
                    nodisc_values.append(third["without_discount"])
                if debug:
                    print(
                        f"[hw1] {path.name}: readings disagreed, adjudicated "
                        f"paid={third['paid']} nodisc={third['without_discount']}",
                        file=sys.stderr,
                    )

        if not paid_values or not nodisc_values:
            return None

        paid = median(paid_values)
        without_discount = median(nodisc_values)
        if debug:
            print(
                f"[hw1] {path.name}: paid={paid} {[str(v) for v in paid_values]} "
                f"nodisc={without_discount} {[str(v) for v in nodisc_values]}",
                file=sys.stderr,
            )
        return paid, without_discount

    # Stage 1 runs in parallel: each receipt is an independent extraction.
    workers = max(1, min(4, len(images)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_receipt = list(pool.map(parse_receipt, images))

    # Stage 3: deterministic aggregation. Money is never summed by the model
    # and never summed with floats.
    total_paid = sum((r[0] for r in per_receipt if r is not None), zero)
    total_without_discount = sum((r[1] for r in per_receipt if r is not None), zero)

    def format_amount(value: Decimal) -> str:
        """Stage 4: exactly one number in the response, nothing else."""
        return f"HK${value.quantize(cents, rounding=ROUND_HALF_UP):.2f}"

    return {
        QUERY_1: format_amount(total_paid),
        QUERY_2: format_amount(total_without_discount),
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
