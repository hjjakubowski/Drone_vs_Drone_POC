from __future__ import annotations


def format_ms(milliseconds: int | float) -> str:
    total_seconds = max(0, int(milliseconds // 1000))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
