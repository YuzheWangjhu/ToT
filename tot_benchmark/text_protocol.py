from __future__ import annotations

import re

_ROLE_PREFIX_RE = re.compile(r"^(assistant|human|user|response|answer)\s*:\s*", re.IGNORECASE)
_BULLET_PREFIX_RE = re.compile(r"^[-*•]\s+")


def normalize_first_line(raw_text: str) -> str:
    text = str(raw_text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    non_empty = [line for line in lines if line]
    if not non_empty:
        return ""

    line = non_empty[0]
    if line.startswith("```"):
        non_fence = [candidate for candidate in non_empty if not candidate.startswith("```")]
        if not non_fence:
            return ""
        line = non_fence[0]

    line = line.strip().strip("`").strip()
    line = _ROLE_PREFIX_RE.sub("", line).strip()
    line = _BULLET_PREFIX_RE.sub("", line).strip()
    line = line.strip().strip("`").strip()
    return line
