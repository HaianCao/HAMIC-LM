"""Vietnamese probe strings for the tokenizer acceptance suite.

Each category targets a known failure mode of tokenizers on Vietnamese text. The probes
are short, hand-written and free of personal data; they complement, not replace, the
held-out corpus metrics.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["PROBES"]

PROBES: Mapping[str, tuple[str, ...]] = {
    "person_names": (
        "Nguyễn Thị Minh Khai",
        "Trần Hưng Đạo",
        "Võ Nguyên Giáp",
        "Hồ Xuân Hương",
    ),
    "place_names": (
        "Đắk Lắk",
        "Bà Rịa – Vũng Tàu",
        "Thừa Thiên Huế",
        "Thành phố Hồ Chí Minh",
        "Quảng Ngãi",
    ),
    "multi_syllable_terms": (
        "trí tuệ nhân tạo",
        "bảo hiểm xã hội",
        "cơ sở dữ liệu",
        "biến đổi khí hậu",
    ),
    "tone_placement": ("hoà bình", "hòa bình", "thuỷ điện", "thủy điện"),
    "no_diacritics": ("toi di hoc o truong dai hoc", "ban co khoe khong"),
    "teencode": ("ko bít j hết á", "đc r nha", "iu nhìu lắm"),
    "code_switch": ("mai check mail rồi gửi file nhé", "deadline tuần này gấp lắm"),
    "numbers_money_dates": ("1.500.000đ", "3,14", "25/12/2026", "15%", "0,5 m²"),
    "urls_email": ("https://vi.wikipedia.org/wiki/Việt_Nam", "lien.he@vi-du.vn"),
    "emoji": ("đẹp quá 😍😍", "cảm ơn 🙏"),
    "math": ("x² + y² = z²", "∑ aᵢ ≤ 10"),
    "code": ("def tinh_tong(a, b):\n    return a + b",),
    "elongation": ("đẹpppp quáaaa", "hayyyy"),
}
