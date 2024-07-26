from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

UNITS_MULT = {
    "ns": 1,
    "us": 1000,
    "ms": 1000,
    "s": 1000,
    "m": 60,
    "h": 60,
    "d": 24,
    "w": 7,
    "mo": 30 / 7,
    "y": 12,
}


def readable_twin(ax: Axes, unit: str):
    reached_unit = False
    sub_d = {}
    for k, v in UNITS_MULT.items():
        if reached_unit:
            sub_d[k] = v
        else:
            reached_unit = k == unit
    if len(sub_d) == 0:
        raise ValueError(f"unit should be one of {list(UNITS_MULT.keys())}")

    t_min, t_max = ax.get_xlim()
    twin = ax.twiny()
    xscale = ax.get_xscale()
    twin.set_xscale(xscale)
    if xscale != "linear":
        twin.minorticks_off()

    list_units = list(sub_d.keys())
    if t_min <= 1:
        mult = 1
        tick_unit = unit
    else:
        tick_unit = list_units[0]
        mult = sub_d[tick_unit]
        list_units.pop(0)
    tick_labels = [f"1{tick_unit}"]
    tick_values = [mult]
    for u in list_units:
        new_mult = sub_d[u]
        mult *= new_mult
        if mult > t_max:
            break
        # Avoid label overlap on linear scale by removing first ones:
        if xscale != "linear" or (t_max - t_min) / mult < 10:
            tick_values.append(mult)
            tick_labels.append(f"1{u}")
    twin.set_xlim(t_min, t_max)
    twin.set_xticks(tick_values, labels=tick_labels)
    return twin
