"""Dataset format converters: Alpaca, ShareGPT, Conversational, DPO."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# --- Format Detection ---


def detect_format(sample: dict[str, Any]) -> str:
    """Auto-detect dataset format from a sample row."""
    keys = set(sample.keys())

    if {"instruction", "output"}.issubset(keys):
        return "alpaca"
    if "conversations" in keys:
        return "sharegpt"
    if "messages" in keys:
        return "conversational"
    if {"prompt", "chosen", "rejected"}.issubset(keys):
        return "dpo"
    if {"question", "answer"}.issubset(keys):
        return "alpaca"  # Map Q&A to Alpaca
    if {"text"} == keys or "text" in keys:
        return "raw_text"

    logger.warning(f"Could not detect format from keys: {keys}. Defaulting to alpaca.")
    return "alpaca"


# --- Converters ---


def to_alpaca(row: dict[str, Any], source_format: str) -> dict[str, str]:
    """Convert any format row to Alpaca format."""
    if source_format == "alpaca":
        return {
            "instruction": str(row.get("instruction", row.get("question", ""))),
            "input": str(row.get("input", "")),
            "output": str(row.get("output", row.get("answer", ""))),
        }
    elif source_format == "sharegpt":
        convos = row.get("conversations", [])
        instruction = ""
        output = ""
        for turn in convos:
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", turn.get("content", ""))
            if role in ("human", "user"):
                instruction = value
            elif role in ("gpt", "assistant"):
                output = value
        return {"instruction": instruction, "input": "", "output": output}
    elif source_format == "conversational":
        messages = row.get("messages", [])
        instruction = ""
        output = ""
        for msg in messages:
            if msg.get("role") == "user":
                instruction = msg.get("content", "")
            elif msg.get("role") == "assistant":
                output = msg.get("content", "")
        return {"instruction": instruction, "input": "", "output": output}
    elif source_format == "raw_text":
        text = str(row.get("text", ""))
        return {"instruction": "Continue the following text:", "input": text[:500], "output": text[500:]}
    else:
        return {
            "instruction": str(row.get("instruction", row.get("question", row.get("text", "")))),
            "input": str(row.get("input", "")),
            "output": str(row.get("output", row.get("answer", row.get("response", "")))),
        }


def to_sharegpt(row: dict[str, Any], source_format: str) -> dict[str, Any]:
    """Convert any format row to ShareGPT format."""
    if source_format == "sharegpt":
        return row
    elif source_format == "alpaca":
        instruction = str(row.get("instruction", ""))
        inp = str(row.get("input", ""))
        output = str(row.get("output", ""))
        user_msg = f"{instruction}\n{inp}".strip() if inp else instruction
        return {
            "conversations": [
                {"from": "human", "value": user_msg},
                {"from": "gpt", "value": output},
            ]
        }
    elif source_format == "conversational":
        messages = row.get("messages", [])
        conversations = []
        for msg in messages:
            role_map = {"user": "human", "assistant": "gpt", "system": "system"}
            conversations.append({
                "from": role_map.get(msg.get("role", "user"), "human"),
                "value": msg.get("content", ""),
            })
        return {"conversations": conversations}
    else:
        alpaca = to_alpaca(row, source_format)
        return to_sharegpt(alpaca, "alpaca")


def to_conversational(row: dict[str, Any], source_format: str) -> dict[str, Any]:
    """Convert any format row to Conversational (messages) format."""
    if source_format == "conversational":
        return row
    elif source_format == "alpaca":
        instruction = str(row.get("instruction", ""))
        inp = str(row.get("input", ""))
        output = str(row.get("output", ""))
        user_msg = f"{instruction}\n{inp}".strip() if inp else instruction
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]
        }
    elif source_format == "sharegpt":
        convos = row.get("conversations", [])
        messages = []
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        for turn in convos:
            messages.append({
                "role": role_map.get(turn.get("from", "human"), "user"),
                "content": turn.get("value", ""),
            })
        return {"messages": messages}
    else:
        alpaca = to_alpaca(row, source_format)
        return to_conversational(alpaca, "alpaca")


def to_dpo(row: dict[str, Any], source_format: str) -> dict[str, Any]:
    """Convert row to DPO format. Only works if source already has chosen/rejected."""
    if source_format == "dpo":
        return {
            "prompt": str(row.get("prompt", "")),
            "chosen": str(row.get("chosen", "")),
            "rejected": str(row.get("rejected", "")),
        }
    raise ValueError(
        f"Cannot auto-convert '{source_format}' to DPO format. "
        "DPO requires explicit chosen/rejected pairs."
    )


# --- Dispatcher ---

CONVERTERS = {
    "alpaca": to_alpaca,
    "sharegpt": to_sharegpt,
    "conversational": to_conversational,
    "dpo": to_dpo,
}


def convert_format(
    row: dict[str, Any],
    source_format: str,
    target_format: str,
) -> dict[str, Any]:
    """Convert a single row from source_format to target_format."""
    if source_format == target_format:
        return row

    converter = CONVERTERS.get(target_format)
    if converter is None:
        raise ValueError(f"Unknown target format: {target_format}")

    return converter(row, source_format)


def get_text_for_filtering(row: dict[str, Any], fmt: str) -> str:
    """Extract the main text content from a row for quality filtering."""
    if fmt == "alpaca":
        parts = [
            str(row.get("instruction", "")),
            str(row.get("input", "")),
            str(row.get("output", "")),
        ]
        return " ".join(p for p in parts if p)
    elif fmt == "sharegpt":
        return " ".join(
            turn.get("value", "") for turn in row.get("conversations", [])
        )
    elif fmt == "conversational":
        return " ".join(
            msg.get("content", "") for msg in row.get("messages", [])
        )
    elif fmt == "dpo":
        return f"{row.get('prompt', '')} {row.get('chosen', '')} {row.get('rejected', '')}"
    else:
        return str(row.get("text", str(row)))
