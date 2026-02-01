"""
Вспомогательные утилиты, используемые в разных модулях.
"""

import re
from datetime import timedelta


# Маппинг суффиксов периодов в timedelta
_PERIOD_PATTERN = re.compile(r"^(\d+)(h|d)$")

_UNIT_MAP: dict[str, str] = {
    "h": "hours",
    "d": "days",
}


def parse_period(period: str) -> timedelta:
    """
    Конвертирует строку периода ('1h', '6h', '24h', '3d', '7d') в timedelta.

    Raises:
        ValueError: если формат строки некорректен.
    """
    match = _PERIOD_PATTERN.match(period.strip().lower())
    if not match:
        raise ValueError(
            f"Некорректный формат периода: '{period}'. "
            f"Ожидается формат вида '1h', '24h', '3d', '7d'."
        )

    value = int(match.group(1))
    unit_key = match.group(2)
    unit_name = _UNIT_MAP[unit_key]

    return timedelta(**{unit_name: value})