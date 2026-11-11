"""Unit handling.  All internal lengths are INCHES; conversion happens only at
the user-interface / report boundary."""

MM_PER_IN = 25.4
FR_PER_MM = 3.0          # 1 Fr = 1/3 mm of outer diameter

UNITS = ("in", "mm")


def to_disp(v_in, units: str):
    """inches -> display units."""
    return v_in * MM_PER_IN if units == "mm" else v_in


def from_disp(v, units: str):
    """display units -> inches."""
    return v / MM_PER_IN if units == "mm" else v


def unit_suffix(units: str) -> str:
    return " mm" if units == "mm" else '"'


def unit_name(units: str) -> str:
    return "mm" if units == "mm" else "in"


def french(od_in: float) -> float:
    return od_in * MM_PER_IN * FR_PER_MM


def od_from_french(fr: float) -> float:
    return fr / FR_PER_MM / MM_PER_IN


def fmt_len(v_in: float, units: str, signed: bool = False) -> str:
    """Compact numeric string in display units (no suffix)."""
    v = to_disp(v_in, units)
    dec = 4 if units == "mm" else 5
    s = f"{v:+.{dec}f}" if signed else f"{v:.{dec}f}"
    return s


def fmt_len_u(v_in: float, units: str, signed: bool = False) -> str:
    return fmt_len(v_in, units, signed) + unit_suffix(units)


def fmt_dia_both(v_in: float, units: str) -> str:
    """Ø value in primary units with the secondary unit in brackets."""
    if units == "mm":
        return f"Ø{v_in * MM_PER_IN:.3f} mm ({v_in:.4f}\")"
    return f"Ø{v_in:.4f}\" ({v_in * MM_PER_IN:.3f} mm)"
