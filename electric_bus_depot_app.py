# Depot Power Planner
# Copyright (c) 2026 Maria Xylia and Stockholm Environment Institute
# Licensed under the MIT License.

from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


# ==========================================================
# PAGE / STYLE SETTINGS
# ==========================================================
st.set_page_config(page_title="Depot Power Planner", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stDataFrame"] div {
        font-size: 15px;
    }
    .stMarkdown p {
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# INPUT DATA STRUCTURE
# ==========================================================
@dataclass
class DepotInputs:
    n_12m: int
    n_18m: int
    enable_priority: bool
    priority_12m: int
    priority_18m: int
    priority_arrival_time: str
    battery_12m_kwh: float
    battery_18m_kwh: float
    charger_power_kw: float
    baseload_kw: float
    general_arrival_time: str
    charging_end_time: str
    session_length_h: float
    timestep_min: int
    efficiency: float
    min_soc: float
    target_soc: float
    charging_strategy: str
    mode: str
    grid_cap_kw: Optional[float]


# ==========================================================
# UI HELPERS
# ==========================================================
def kpi_box(title: str, value: str, highlight: bool = False) -> None:
    if highlight:
        bg = "#eef6ff"
        border = "#cce0ff"
        title_color = "#4a6fa5"
        value_color = "#1f4e79"
    else:
        bg = "#f8f9fa"
        border = "#e6e6e6"
        title_color = "#666666"
        value_color = "#222222"

    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            border:1px solid {border};
            border-radius:10px;
            padding:10px 8px;
            text-align:center;
            min-height:76px;
            display:flex;
            flex-direction:column;
            justify-content:center;
        ">
            <div style="font-size:12px; color:{title_color}; margin-bottom:4px;">
                {title}
            </div>
            <div style="font-size:18px; font-weight:700; color:{value_color}; line-height:1.15;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_hhmm(value: str, label: str) -> None:
    try:
        datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise ValueError(f"{label} must use HH:MM format, for example 23:00.") from exc


# ==========================================================
# TIME HELPERS
# ==========================================================
def resolve_time(reference_dt: datetime, time_str: str, spans_midnight: bool) -> datetime:
    same_day = datetime.strptime(time_str, "%H:%M").replace(
        year=reference_dt.year,
        month=reference_dt.month,
        day=reference_dt.day,
    )

    if not spans_midnight:
        return same_day

    candidates = [
        same_day - timedelta(days=1),
        same_day,
        same_day + timedelta(days=1),
    ]
    return min(candidates, key=lambda x: abs((x - reference_dt).total_seconds()))


def build_time_context(
    general_arrival_time: str,
    charging_end_time: str,
    enable_priority: bool,
    priority_arrival_time: str,
):
    parse_hhmm(general_arrival_time, "General fleet arrival time")
    parse_hhmm(charging_end_time, "Charging end time")
    parse_hhmm(priority_arrival_time, "Priority bus arrival time")

    base_date = datetime(2026, 3, 20)

    general_arrival_dt = datetime.strptime(general_arrival_time, "%H:%M").replace(
        year=base_date.year,
        month=base_date.month,
        day=base_date.day,
    )

    raw_end_dt = datetime.strptime(charging_end_time, "%H:%M").replace(
        year=base_date.year,
        month=base_date.month,
        day=base_date.day,
    )

    spans_midnight = raw_end_dt <= general_arrival_dt

    charging_end_dt = raw_end_dt + timedelta(days=1) if spans_midnight else raw_end_dt

    if enable_priority:
        priority_arrival_dt = resolve_time(general_arrival_dt, priority_arrival_time, spans_midnight)
    else:
        priority_arrival_dt = general_arrival_dt

    effective_start_dt = min(general_arrival_dt, priority_arrival_dt)

    while charging_end_dt <= effective_start_dt:
        charging_end_dt += timedelta(days=1)

    return general_arrival_dt, priority_arrival_dt, charging_end_dt, effective_start_dt


# ==========================================================
# GENERAL MODEL HELPERS
# ==========================================================
def get_status_category(
    mode: str,
    capacity_gap_kw: float,
    required_kw: float,
    available_kw: Optional[float],
) -> str:
    if mode == "estimate_capacity":
        return "info"
    if available_kw is None:
        return "info"
    if capacity_gap_kw > 1e-9:
        return "red"

    margin_ratio = (available_kw - required_kw) / required_kw if required_kw > 0 else 0.0
    if margin_ratio <= 0.10:
        return "yellow"
    return "green"


def split_evenly(total_items: int, n_bins: int) -> List[int]:
    if n_bins <= 0:
        return []
    base = total_items // n_bins
    remainder = total_items % n_bins
    result = [base] * n_bins
    for i in range(remainder):
        result[i] += 1
    return result


def calculate_energy_per_bus(inputs: DepotInputs):
    soc_fraction = max(0.0, inputs.target_soc - inputs.min_soc)
    e_12m_grid_kwh = soc_fraction * inputs.battery_12m_kwh / inputs.efficiency
    e_18m_grid_kwh = soc_fraction * inputs.battery_18m_kwh / inputs.efficiency
    return e_12m_grid_kwh, e_18m_grid_kwh


def validate_common_inputs(inputs: DepotInputs) -> None:
    total_buses = inputs.n_12m + inputs.n_18m
    if total_buses == 0:
        raise ValueError("Please enter at least one bus.")
    if inputs.target_soc < inputs.min_soc:
        raise ValueError("Battery level before buses leave must be higher than battery level when buses return.")
    if inputs.efficiency <= 0:
        raise ValueError("Charging efficiency must be greater than zero.")
    if inputs.priority_12m > inputs.n_12m:
        raise ValueError("Priority 12 m buses cannot exceed total 12 m buses.")
    if inputs.priority_18m > inputs.n_18m:
        raise ValueError("Priority 18 m buses cannot exceed total 18 m buses.")


# ==========================================================
# PLOT HELPERS
# ==========================================================
def get_24h_display_window(effective_start_dt: datetime):
    display_start = effective_start_dt.replace(hour=12, minute=0)
    if display_start > effective_start_dt:
        display_start -= timedelta(days=1)
    display_end = display_start + timedelta(hours=24)
    return display_start, display_end


def build_step_series(
    intervals: List[Dict],
    display_start: datetime,
    display_end: datetime,
    base_kw: float = 0.0,
):
    clipped = []
    for item in intervals:
        s = max(item["start"], display_start)
        e = min(item["end"], display_end)
        if e > s:
            clipped.append(
                {
                    "start": s,
                    "end": e,
                    "charging_kw": item["charging_kw"],
                }
            )

    clipped.sort(key=lambda x: x["start"])

    x = [display_start]
    y = [base_kw]

    current_time = display_start
    current_load = base_kw

    for i, item in enumerate(clipped):
        s = item["start"]
        e = item["end"]
        target_load = base_kw + item["charging_kw"]

        if s > current_time:
            x.append(s)
            y.append(current_load)

        if current_load != target_load:
            x.append(s)
            y.append(target_load)

        x.append(e)
        y.append(target_load)

        current_time = e
        current_load = target_load

        next_starts_now = i < len(clipped) - 1 and clipped[i + 1]["start"] == e

        if not next_starts_now:
            if current_load != base_kw:
                x.append(e)
                y.append(base_kw)
            current_load = base_kw

    if current_time < display_end:
        x.append(display_end)
        y.append(current_load)

    return x, y


# ==========================================================
# PDF HELPERS
# ==========================================================
def wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]

    lines = []
    current_line = words[0]

    for word in words[1:]:
        trial_line = current_line + " " + word
        if stringWidth(trial_line, font_name, font_size) <= max_width:
            current_line = trial_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines


def draw_key_value_section(
    c: canvas.Canvas,
    title: str,
    items: List[tuple],
    x_left: float,
    y_top: float,
    page_width: float,
    page_height: float,
    bottom_margin: float,
):
    y = y_top

    if y < bottom_margin + 40:
        c.showPage()
        y = page_height - 50

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_left, y, title)
    y -= 18

    label_width = 220
    value_width = page_width - x_left * 2 - label_width - 10

    for key, value in items:
        value_lines = wrap_text(str(value), "Helvetica", 9, value_width)
        row_height = max(14, 12 * len(value_lines))

        if y - row_height < bottom_margin:
            c.showPage()
            y = page_height - 50
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_left, y, title + " (cont.)")
            y -= 18

        c.setFont("Helvetica-Bold", 9)
        c.drawString(x_left, y, str(key))

        c.setFont("Helvetica", 9)
        value_y = y
        for line in value_lines:
            c.drawString(x_left + label_width, value_y, line)
            value_y -= 11

        y -= row_height

    return y - 10


def draw_dataframe_section(
    c: canvas.Canvas,
    title: str,
    df: pd.DataFrame,
    x_left: float,
    y_top: float,
    page_width: float,
    page_height: float,
    bottom_margin: float,
):
    y = y_top
    max_table_width = page_width - 2 * x_left

    if y < bottom_margin + 50:
        c.showPage()
        y = page_height - 50

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_left, y, title)
    y -= 18

    cols = list(df.columns)
    n_cols = max(1, len(cols))
    col_width = max_table_width / n_cols

    if y < bottom_margin + 30:
        c.showPage()
        y = page_height - 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x_left, y, title + " (cont.)")
        y -= 18

    c.setFont("Helvetica-Bold", 8)
    for i, col in enumerate(cols):
        c.drawString(x_left + i * col_width, y, str(col)[:22])
    y -= 14

    c.setFont("Helvetica", 8)
    for _, row in df.iterrows():
        if y < bottom_margin + 20:
            c.showPage()
            y = page_height - 50
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x_left, y, title + " (cont.)")
            y -= 18
            c.setFont("Helvetica-Bold", 8)
            for i, col in enumerate(cols):
                c.drawString(x_left + i * col_width, y, str(col)[:22])
            y -= 14
            c.setFont("Helvetica", 8)

        for i, col in enumerate(cols):
            text = str(row[col])
            c.drawString(x_left + i * col_width, y, text[:22])
        y -= 12

    return y - 10


def create_summary_pdf(
    inputs: DepotInputs,
    main_results_df: pd.DataFrame,
    advanced_df: pd.DataFrame,
    table_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    table_title: str,
) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    page_width, page_height = A4
    x_left = 40
    y = page_height - 50
    bottom_margin = 40

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_left, y, "Depot Power Planner")
    y -= 18

    c.setFont("Helvetica", 10)
    c.drawString(x_left, y, "Summary export")
    y -= 24

    inputs_items = [
        ("Charging strategy", "Fixed charging sessions" if inputs.charging_strategy == "fixed_sessions" else "Flexible smart charging"),
        ("Mode", "Estimate minimum depot capacity" if inputs.mode == "estimate_capacity" else "Check against known depot power limit"),
        ("12 m buses", inputs.n_12m),
        ("18 m buses", inputs.n_18m),
        ("Priority charging enabled", "Yes" if inputs.enable_priority else "No"),
        ("Priority 12 m buses", inputs.priority_12m),
        ("Priority 18 m buses", inputs.priority_18m),
        ("Priority arrival time", inputs.priority_arrival_time if inputs.enable_priority else "Not used"),
        ("Battery size of 12 m bus (kWh)", inputs.battery_12m_kwh),
        ("Battery size of 18 m bus (kWh)", inputs.battery_18m_kwh),
        ("Charger power per bus (kW)", inputs.charger_power_kw),
        ("Depot baseload (kW)", inputs.baseload_kw),
        ("Depot power limit (kW)", inputs.grid_cap_kw if inputs.grid_cap_kw is not None else "Not entered"),
        ("General fleet arrival time", inputs.general_arrival_time),
        ("Charging end time", inputs.charging_end_time),
        ("Fixed session length (hours)", inputs.session_length_h if inputs.charging_strategy == "fixed_sessions" else "Not used"),
        ("Charging efficiency", inputs.efficiency),
        ("Battery level when buses return", inputs.min_soc),
        ("Battery level before buses leave", inputs.target_soc),
    ]

    y = draw_key_value_section(c, "Inputs", inputs_items, x_left, y, page_width, page_height, bottom_margin)

    main_items = list(zip(main_results_df["Metric"], main_results_df["Value"]))
    y = draw_key_value_section(c, "Main results", main_items, x_left, y, page_width, page_height, bottom_margin)

    if not advanced_df.empty:
        advanced_items = list(zip(advanced_df["Metric"], advanced_df["Value"]))
        y = draw_key_value_section(c, "Advanced details", advanced_items, x_left, y, page_width, page_height, bottom_margin)

    y = draw_dataframe_section(c, table_title, table_df, x_left, y, page_width, page_height, bottom_margin)
    y = draw_dataframe_section(c, "Quick scenario comparison", scenario_df, x_left, y, page_width, page_height, bottom_margin)

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ==========================================================
# MODEL: FIXED SESSIONS
# ==========================================================
def run_fixed_sessions(inputs: DepotInputs):
    validate_common_inputs(inputs)

    general_arrival_dt, priority_arrival_dt, charging_end_dt, effective_start_dt = build_time_context(
        inputs.general_arrival_time,
        inputs.charging_end_time,
        inputs.enable_priority,
        inputs.priority_arrival_time,
    )

    total_window_h = (charging_end_dt - effective_start_dt).total_seconds() / 3600

    if total_window_h <= 0:
        raise ValueError("Charging window must be greater than zero.")
    if inputs.session_length_h <= 0:
        raise ValueError("Session length must be greater than zero.")

    n_sessions = int(total_window_h // inputs.session_length_h)
    if n_sessions < 1:
        raise ValueError("The charging window is shorter than one session. Please reduce session length or increase the charging window.")

    session_starts = [
        effective_start_dt + timedelta(hours=i * inputs.session_length_h)
        for i in range(n_sessions)
    ]
    session_ends = [s + timedelta(hours=inputs.session_length_h) for s in session_starts]

    used_session_time_h = n_sessions * inputs.session_length_h
    unused_time_h = total_window_h - used_session_time_h

    priority_12m = inputs.priority_12m if inputs.enable_priority else 0
    priority_18m = inputs.priority_18m if inputs.enable_priority else 0

    e_12m_grid_kwh, e_18m_grid_kwh = calculate_energy_per_bus(inputs)

    p_12m_required_kw = e_12m_grid_kwh / inputs.session_length_h
    p_18m_required_kw = e_18m_grid_kwh / inputs.session_length_h

    if p_12m_required_kw > inputs.charger_power_kw + 1e-9:
        raise ValueError(f"12 m buses require {p_12m_required_kw:.1f} kW per bus within one session, above the charger power of {inputs.charger_power_kw:.1f} kW.")
    if p_18m_required_kw > inputs.charger_power_kw + 1e-9:
        raise ValueError(f"18 m buses require {p_18m_required_kw:.1f} kW per bus within one session, above the charger power of {inputs.charger_power_kw:.1f} kW.")

    def eligible_session_indices(arrival_dt: datetime) -> List[int]:
        return [
            i for i, s in enumerate(session_starts)
            if s >= arrival_dt and session_ends[i] <= charging_end_dt
        ]

    priority_eligible = eligible_session_indices(priority_arrival_dt)
    general_eligible = eligible_session_indices(general_arrival_dt)

    if len(general_eligible) == 0:
        raise ValueError("No session starts at or after the general fleet arrival time.")

    session_12m = [0] * n_sessions
    session_18m = [0] * n_sessions
    session_12m_priority = [0] * n_sessions
    session_18m_priority = [0] * n_sessions
    reserved_priority_sessions = set()

    if inputs.enable_priority and (priority_12m + priority_18m) > 0:
        if len(priority_eligible) == 0:
            raise ValueError("No session starts at or after the priority arrival time.")

        first_priority_session = priority_eligible[0]
        session_12m[first_priority_session] += priority_12m
        session_18m[first_priority_session] += priority_18m
        session_12m_priority[first_priority_session] += priority_12m
        session_18m_priority[first_priority_session] += priority_18m
        reserved_priority_sessions.add(first_priority_session)

    remaining_12m = inputs.n_12m - priority_12m
    remaining_18m = inputs.n_18m - priority_18m

    nonpriority_eligible = [i for i in general_eligible if i not in reserved_priority_sessions]

    if len(nonpriority_eligible) == 0 and (remaining_12m > 0 or remaining_18m > 0):
        raise ValueError("No session remains for non-priority buses after assigning the priority buses first.")

    split_remaining_12m = split_evenly(remaining_12m, len(nonpriority_eligible))
    split_remaining_18m = split_evenly(remaining_18m, len(nonpriority_eligible))

    for idx, session_idx in enumerate(nonpriority_eligible):
        session_12m[session_idx] += split_remaining_12m[idx]
        session_18m[session_idx] += split_remaining_18m[idx]

    session_rows = []
    intervals = []

    for i in range(n_sessions):
        n12 = session_12m[i]
        n18 = session_18m[i]
        session_power_kw = n12 * p_12m_required_kw + n18 * p_18m_required_kw
        session_energy_kwh = n12 * e_12m_grid_kwh + n18 * e_18m_grid_kwh

        session_rows.append(
            {
                "Session": i + 1,
                "Start": session_starts[i],
                "End": session_ends[i],
                "Priority 12 m buses": session_12m_priority[i],
                "Priority 18 m buses": session_18m_priority[i],
                "12 m buses": n12,
                "18 m buses": n18,
                "Total buses": n12 + n18,
                "Charging load (MW)": round(session_power_kw / 1000, 3),
                "Total depot load incl. baseload (MW)": round((session_power_kw + inputs.baseload_kw) / 1000, 3),
                "Energy delivered (MWh)": round(session_energy_kwh / 1000, 3),
            }
        )

        intervals.append(
            {
                "start": session_starts[i],
                "end": session_ends[i],
                "charging_kw": session_power_kw,
            }
        )

    sessions_df = pd.DataFrame(session_rows)

    charging_peak_kw = sessions_df["Charging load (MW)"].max() * 1000 if len(sessions_df) > 0 else 0.0
    total_peak_kw = charging_peak_kw + inputs.baseload_kw

    available_limit_kw = None
    if inputs.mode == "check_known_limit":
        if inputs.grid_cap_kw is None:
            raise ValueError("Please enter the depot power limit.")
        available_limit_kw = inputs.grid_cap_kw

        if total_peak_kw <= inputs.grid_cap_kw + 1e-9:
            status_label = "Within depot power limit"
            status_text = "The charging schedule fits within the depot power limit you entered."
            capacity_gap_kw = 0.0
            feasible_against_cap = True
        else:
            status_label = "Above depot power limit"
            status_text = "The charging schedule exceeds the depot power limit you entered."
            capacity_gap_kw = total_peak_kw - inputs.grid_cap_kw
            feasible_against_cap = False
    else:
        status_label = "Minimum capacity estimated"
        status_text = "The tool has estimated the minimum depot power needed under the fixed-session setup."
        capacity_gap_kw = 0.0
        feasible_against_cap = True

    status_category = get_status_category(inputs.mode, capacity_gap_kw, total_peak_kw, available_limit_kw)

    total_grid_energy_kwh = inputs.n_12m * e_12m_grid_kwh + inputs.n_18m * e_18m_grid_kwh
    total_depot_energy_kwh = total_grid_energy_kwh + inputs.baseload_kw * total_window_h

    summary = {
        "Charging strategy": "Fixed charging sessions",
        "Mode": "Estimate minimum depot capacity" if inputs.mode == "estimate_capacity" else "Check against known depot power limit",
        "Number of sessions": n_sessions,
        "Session length (h)": inputs.session_length_h,
        "Total charging horizon (h)": round(total_window_h, 2),
        "Session time used (h)": round(used_session_time_h, 2),
        "Unused time in charging horizon (h)": round(unused_time_h, 2),
        "12 m buses": inputs.n_12m,
        "18 m buses": inputs.n_18m,
        "Priority charging enabled": "Yes" if inputs.enable_priority else "No",
        "Priority 12 m buses": priority_12m,
        "Priority 18 m buses": priority_18m,
        "General fleet arrival time": general_arrival_dt.strftime("%H:%M"),
        "Priority bus arrival time": priority_arrival_dt.strftime("%H:%M") if inputs.enable_priority else "Not used",
        "Charging end time": charging_end_dt.strftime("%H:%M"),
        "Depot baseload (kW)": round(inputs.baseload_kw, 1),
        "Battery level when buses return": inputs.min_soc,
        "Battery level before buses leave": inputs.target_soc,
        "Electricity needed per 12 m bus (kWh)": round(e_12m_grid_kwh, 1),
        "Electricity needed per 18 m bus (kWh)": round(e_18m_grid_kwh, 1),
        "Charging-only peak load (MW)": round(charging_peak_kw / 1000, 3),
        "Total depot peak load incl. baseload (MW)": round(total_peak_kw / 1000, 3),
        "Charging electricity needed (MWh)": round(total_grid_energy_kwh / 1000, 3),
        "Total depot electricity during charging horizon (MWh)": round(total_depot_energy_kwh / 1000, 3),
        "Status": status_label,
    }

    if inputs.mode == "check_known_limit":
        summary["Depot power limit entered by user (MW)"] = round(inputs.grid_cap_kw / 1000, 3)
        summary["Capacity gap above limit (MW)"] = round(capacity_gap_kw / 1000, 3)
        summary["Can the fleet be charged overnight?"] = "Yes" if feasible_against_cap else "No"

    return {
        "table_df": sessions_df,
        "table_title": "Session allocation table",
        "table_note": "How the sessions are built: priority buses are assigned to the earliest session that starts at or after their arrival time. The remaining buses are then spread as evenly as possible across the remaining eligible sessions to keep the depot peak as low as possible.",
        "intervals": intervals,
        "summary": summary,
        "status_label": status_label,
        "status_text": status_text,
        "capacity_gap_kw": capacity_gap_kw,
        "status_category": status_category,
        "required_kw": total_peak_kw,
        "charging_peak_kw": charging_peak_kw,
        "effective_start_dt": effective_start_dt,
        "charging_end_dt": charging_end_dt,
    }


# ==========================================================
# MODEL: SMART CHARGING
# ==========================================================
def run_smart_charging(inputs: DepotInputs):
    validate_common_inputs(inputs)

    general_arrival_dt, priority_arrival_dt, charging_end_dt, effective_start_dt = build_time_context(
        inputs.general_arrival_time,
        inputs.charging_end_time,
        inputs.enable_priority,
        inputs.priority_arrival_time,
    )

    total_window_h = (charging_end_dt - effective_start_dt).total_seconds() / 3600
    if total_window_h <= 0:
        raise ValueError("Charging window must be greater than zero.")

    priority_12m = inputs.priority_12m if inputs.enable_priority else 0
    priority_18m = inputs.priority_18m if inputs.enable_priority else 0

    e_12m_grid_kwh, e_18m_grid_kwh = calculate_energy_per_bus(inputs)

    priority_energy_kwh = priority_12m * e_12m_grid_kwh + priority_18m * e_18m_grid_kwh
    priority_bus_count = priority_12m + priority_18m

    rem_12m = inputs.n_12m - priority_12m
    rem_18m = inputs.n_18m - priority_18m
    remaining_energy_kwh = rem_12m * e_12m_grid_kwh + rem_18m * e_18m_grid_kwh

    if priority_bus_count > 0:
        priority_charging_power_kw = priority_bus_count * inputs.charger_power_kw
        priority_duration_h = priority_energy_kwh / priority_charging_power_kw
    else:
        priority_charging_power_kw = 0.0
        priority_duration_h = 0.0

    priority_finish_dt = priority_arrival_dt + timedelta(hours=priority_duration_h)

    remaining_start_dt = max(general_arrival_dt, priority_finish_dt)
    remaining_window_h = (charging_end_dt - remaining_start_dt).total_seconds() / 3600

    if priority_duration_h > total_window_h + 1e-9:
        raise ValueError("Priority buses alone cannot be fully charged within the charging horizon.")

    if remaining_energy_kwh > 1e-9 and remaining_window_h <= 1e-9:
        raise ValueError("No time remains for the non-priority buses after charging the priority buses first.")

    remaining_required_kw = remaining_energy_kwh / remaining_window_h if remaining_window_h > 0 else 0.0
    charging_peak_kw = max(priority_charging_power_kw, remaining_required_kw)
    total_peak_kw = charging_peak_kw + inputs.baseload_kw

    available_limit_kw = None
    if inputs.mode == "check_known_limit":
        if inputs.grid_cap_kw is None:
            raise ValueError("Please enter the depot power limit.")
        available_limit_kw = inputs.grid_cap_kw

        if total_peak_kw <= inputs.grid_cap_kw + 1e-9:
            status_label = "Within depot power limit"
            status_text = "The smart charging profile fits within the depot power limit you entered."
            capacity_gap_kw = 0.0
            feasible_against_cap = True
        else:
            status_label = "Above depot power limit"
            status_text = "The smart charging profile exceeds the depot power limit you entered."
            capacity_gap_kw = total_peak_kw - inputs.grid_cap_kw
            feasible_against_cap = False
    else:
        status_label = "Minimum capacity estimated"
        status_text = "The tool has estimated the minimum depot power needed under the flexible smart charging setup."
        capacity_gap_kw = 0.0
        feasible_against_cap = True

    status_category = get_status_category(inputs.mode, capacity_gap_kw, total_peak_kw, available_limit_kw)

    block_rows = []
    intervals = []

    if priority_bus_count > 0 and priority_duration_h > 1e-9:
        block_rows.append(
            {
                "Charging block": 1,
                "Start": priority_arrival_dt,
                "End": priority_finish_dt,
                "Block type": "Priority buses charge first",
                "Priority 12 m buses": priority_12m,
                "Priority 18 m buses": priority_18m,
                "12 m buses": priority_12m,
                "18 m buses": priority_18m,
                "Total buses": priority_12m + priority_18m,
                "Charging load (MW)": round(priority_charging_power_kw / 1000, 3),
                "Total depot load incl. baseload (MW)": round((priority_charging_power_kw + inputs.baseload_kw) / 1000, 3),
                "Energy delivered (MWh)": round(priority_energy_kwh / 1000, 3),
            }
        )
        intervals.append(
            {
                "start": priority_arrival_dt,
                "end": priority_finish_dt,
                "charging_kw": priority_charging_power_kw,
            }
        )

    if remaining_energy_kwh > 1e-9 and remaining_window_h > 1e-9:
        block_rows.append(
            {
                "Charging block": len(block_rows) + 1,
                "Start": remaining_start_dt,
                "End": charging_end_dt,
                "Block type": "Flexible smart charging for remaining fleet",
                "Priority 12 m buses": 0,
                "Priority 18 m buses": 0,
                "12 m buses": rem_12m,
                "18 m buses": rem_18m,
                "Total buses": rem_12m + rem_18m,
                "Charging load (MW)": round(remaining_required_kw / 1000, 3),
                "Total depot load incl. baseload (MW)": round((remaining_required_kw + inputs.baseload_kw) / 1000, 3),
                "Energy delivered (MWh)": round(remaining_energy_kwh / 1000, 3),
            }
        )
        intervals.append(
            {
                "start": remaining_start_dt,
                "end": charging_end_dt,
                "charging_kw": remaining_required_kw,
            }
        )

    smart_df = pd.DataFrame(block_rows)

    total_grid_energy_kwh = inputs.n_12m * e_12m_grid_kwh + inputs.n_18m * e_18m_grid_kwh
    total_depot_energy_kwh = total_grid_energy_kwh + inputs.baseload_kw * total_window_h

    summary = {
        "Charging strategy": "Flexible smart charging",
        "Mode": "Estimate minimum depot capacity" if inputs.mode == "estimate_capacity" else "Check against known depot power limit",
        "Total charging horizon (h)": round(total_window_h, 2),
        "Priority charging time used (h)": round(priority_duration_h, 2),
        "Remaining flexible charging time (h)": round(max(remaining_window_h, 0.0), 2),
        "12 m buses": inputs.n_12m,
        "18 m buses": inputs.n_18m,
        "Priority charging enabled": "Yes" if inputs.enable_priority else "No",
        "Priority 12 m buses": priority_12m,
        "Priority 18 m buses": priority_18m,
        "General fleet arrival time": general_arrival_dt.strftime("%H:%M"),
        "Priority bus arrival time": priority_arrival_dt.strftime("%H:%M") if inputs.enable_priority else "Not used",
        "Charging end time": charging_end_dt.strftime("%H:%M"),
        "Depot baseload (kW)": round(inputs.baseload_kw, 1),
        "Battery level when buses return": inputs.min_soc,
        "Battery level before buses leave": inputs.target_soc,
        "Electricity needed per 12 m bus (kWh)": round(e_12m_grid_kwh, 1),
        "Electricity needed per 18 m bus (kWh)": round(e_18m_grid_kwh, 1),
        "Charging-only peak load (MW)": round(charging_peak_kw / 1000, 3),
        "Total depot peak load incl. baseload (MW)": round(total_peak_kw / 1000, 3),
        "Charging electricity needed (MWh)": round(total_grid_energy_kwh / 1000, 3),
        "Total depot electricity during charging horizon (MWh)": round(total_depot_energy_kwh / 1000, 3),
        "Status": status_label,
    }

    if inputs.mode == "check_known_limit":
        summary["Depot power limit entered by user (MW)"] = round(inputs.grid_cap_kw / 1000, 3)
        summary["Capacity gap above limit (MW)"] = round(capacity_gap_kw / 1000, 3)
        summary["Can the fleet be charged overnight?"] = "Yes" if feasible_against_cap else "No"

    return {
        "table_df": smart_df,
        "table_title": "Smart charging blocks",
        "table_note": "How the smart charging profile is built: priority buses start charging immediately when they arrive. The remaining fleet is then charged as flexibly as possible across the rest of the charging horizon to keep the depot peak as low as possible.",
        "intervals": intervals,
        "summary": summary,
        "status_label": status_label,
        "status_text": status_text,
        "capacity_gap_kw": capacity_gap_kw,
        "status_category": status_category,
        "required_kw": total_peak_kw,
        "charging_peak_kw": charging_peak_kw,
        "effective_start_dt": effective_start_dt,
        "charging_end_dt": charging_end_dt,
    }


def run_model(inputs: DepotInputs):
    if inputs.charging_strategy == "fixed_sessions":
        return run_fixed_sessions(inputs)
    return run_smart_charging(inputs)


# ==========================================================
# STREAMLIT HEADER
# ==========================================================
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image("SEI-Master-Logo-Extended-Black-RGB.jpg", width=190)

with col_title:
    st.title("Depot Power Planner")
    st.caption("A planning tool for estimating charging power requirements for battery-electric bus depots.")

st.info("Beta version – feedback is welcome and will help improve future releases.")

st.caption(
    "Getting started: fill in the fleet and depot inputs in the left panel. "
    "The tool estimates required depot power, visualizes the 24-hour load profile, "
    "and summarizes fixed charging sessions or flexible smart charging blocks."
)


# ==========================================================
# SIDEBAR INPUTS
# ==========================================================
with st.sidebar:
    st.header("Inputs")

    charging_strategy_display = st.radio(
        "Charging approach",
        options=["Fixed charging sessions", "Flexible smart charging"],
        index=0,
    )
    charging_strategy = "fixed_sessions" if charging_strategy_display == "Fixed charging sessions" else "smart_charging"

    mode_display = st.radio(
        "What do you want the tool to do?",
        options=["Estimate minimum depot capacity", "Check against a known depot power limit"],
        index=0,
    )
    mode = "estimate_capacity" if mode_display == "Estimate minimum depot capacity" else "check_known_limit"

    st.subheader("Fleet")
    n_12m = st.number_input("Number of 12 m buses", min_value=0, max_value=1000, value=20, step=1)
    n_18m = st.number_input("Number of 18 m buses", min_value=0, max_value=1000, value=10, step=1)

    st.subheader("Priority charging")
    enable_priority = st.toggle(
        "Enable priority charging",
        value=False,
        help="Priority buses must charge as soon as they arrive at the depot.",
    )

    if enable_priority:
        priority_12m = st.number_input("Priority 12 m buses", min_value=0, max_value=1000, value=0, step=1)
        priority_18m = st.number_input("Priority 18 m buses", min_value=0, max_value=1000, value=0, step=1)
        priority_arrival_time = st.text_input("Priority bus arrival time (HH:MM)", "23:00")
    else:
        priority_12m = 0
        priority_18m = 0
        priority_arrival_time = "23:00"

    st.subheader("Vehicle and depot assumptions")
    battery_12m = st.number_input("Battery size of 12 m bus (kWh)", min_value=100.0, max_value=2000.0, value=350.0, step=10.0)
    battery_18m = st.number_input("Battery size of 18 m bus (kWh)", min_value=100.0, max_value=2000.0, value=500.0, step=10.0)
    charger_power_kw = st.number_input("Charger power per bus (kW)", min_value=10.0, max_value=1000.0, value=150.0, step=10.0)

    baseload_kw = st.number_input(
        "Depot baseload (kW)",
        min_value=0.0,
        max_value=10000.0,
        value=0.0,
        step=50.0,
        help="Constant non-charging depot demand such as buildings, heating, or other equipment.",
    )

    if mode == "check_known_limit":
        grid_cap_kw = st.number_input("Depot power limit (kW)", min_value=100.0, max_value=20000.0, value=1800.0, step=50.0)
    else:
        grid_cap_kw = None

    st.subheader("Time assumptions")
    general_arrival_time = st.text_input("General fleet arrival time (HH:MM)", "23:00")
    charging_end_time = st.text_input("Charging end time (HH:MM)", "05:00")

    if charging_strategy == "fixed_sessions":
        session_length_h = st.number_input("Fixed session length (hours)", min_value=0.5, max_value=12.0, value=3.0, step=0.5)
    else:
        session_length_h = 3.0

    st.subheader("Charging assumptions")
    efficiency = st.slider("Charging efficiency", min_value=0.70, max_value=1.00, value=0.92, step=0.01)
    min_soc = st.slider("Battery level when buses return", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    target_soc = st.slider("Battery level before buses leave", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

    timestep_min = st.selectbox(
        "Timestep for chart (minutes)",
        options=[5, 10, 15, 30, 60],
        index=2,
        help="Only affects how smooth the chart looks.",
    )


inputs = DepotInputs(
    n_12m=n_12m,
    n_18m=n_18m,
    enable_priority=enable_priority,
    priority_12m=priority_12m,
    priority_18m=priority_18m,
    priority_arrival_time=priority_arrival_time,
    battery_12m_kwh=battery_12m,
    battery_18m_kwh=battery_18m,
    charger_power_kw=charger_power_kw,
    baseload_kw=baseload_kw,
    general_arrival_time=general_arrival_time,
    charging_end_time=charging_end_time,
    session_length_h=session_length_h,
    timestep_min=timestep_min,
    efficiency=efficiency,
    min_soc=min_soc,
    target_soc=target_soc,
    charging_strategy=charging_strategy,
    mode=mode,
    grid_cap_kw=grid_cap_kw,
)


# ==========================================================
# RUN MODEL AND DISPLAY RESULTS
# ==========================================================
try:
    results = run_model(inputs)

    table_df = results["table_df"]
    table_title = results["table_title"]
    table_note = results["table_note"]
    intervals = results["intervals"]
    summary = results["summary"]
    status_text = results["status_text"]
    capacity_gap_kw = results["capacity_gap_kw"]
    status_category = results["status_category"]
    required_kw = results["required_kw"]
    charging_peak_kw = results["charging_peak_kw"]
    effective_start_dt = results["effective_start_dt"]
    charging_end_dt = results["charging_end_dt"]

    required_mw = required_kw / 1000
    charging_peak_mw = charging_peak_kw / 1000
    total_mwh = summary["Charging electricity needed (MWh)"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_box("Required total depot power", f"{required_mw:.3f} MW", highlight=True)

    with c2:
        kpi_box("Charging-only peak", f"{charging_peak_mw:.3f} MW")

    with c3:
        if mode == "check_known_limit":
            kpi_box("Available depot power", f"{inputs.grid_cap_kw / 1000:.3f} MW")
        else:
            if charging_strategy == "fixed_sessions":
                kpi_box("Number of sessions", f"{summary['Number of sessions']}")
            else:
                kpi_box("Priority buses", f"{inputs.priority_12m + inputs.priority_18m}")

    with c4:
        if mode == "check_known_limit":
            if capacity_gap_kw > 1e-9:
                kpi_box("Capacity gap", f"{capacity_gap_kw / 1000:.3f} MW")
            else:
                kpi_box("Capacity gap", "0.000 MW")
        else:
            kpi_box("Charging electricity", f"{total_mwh:.3f} MWh")

    st.subheader("Capacity check")

    comp1, comp2, comp3, comp4 = st.columns(4)

    with comp1:
        kpi_box("Required total depot power", f"{required_mw:.3f} MW", highlight=True)

    with comp2:
        if mode == "check_known_limit":
            kpi_box("Available depot power", f"{inputs.grid_cap_kw / 1000:.3f} MW")
        else:
            kpi_box("Available depot power", "Not entered")

    with comp3:
        if mode == "check_known_limit":
            if capacity_gap_kw > 1e-9:
                kpi_box("Gap", f"+{capacity_gap_kw / 1000:.3f} MW")
            else:
                spare_kw = inputs.grid_cap_kw - required_kw
                kpi_box("Spare margin", f"{spare_kw / 1000:.3f} MW")
        else:
            kpi_box("Gap", "—")

    with comp4:
        if mode == "estimate_capacity":
            kpi_box("Status", "Estimated")
        elif status_category == "green":
            kpi_box("Status", "Feasible")
        elif status_category == "yellow":
            kpi_box("Status", "Tight")
        else:
            kpi_box("Status", "Not feasible")

    if status_category == "red":
        st.error(f"{status_text} The charging setup exceeds the entered depot limit by {capacity_gap_kw / 1000:.3f} MW.")
    elif status_category == "yellow":
        st.warning("The fleet can be charged overnight, but the depot power limit is tight with little spare margin.")
    elif status_category == "green":
        st.success(status_text)
    else:
        st.info(status_text)

    left_col, right_col = st.columns([1.1, 1.9])

    with left_col:
        st.subheader("Main results")

        main_results = {
            "Charging strategy": summary["Charging strategy"],
            "Required total depot power (MW)": summary["Total depot peak load incl. baseload (MW)"],
            "Charging-only peak load (MW)": summary["Charging-only peak load (MW)"],
            "Depot baseload (kW)": summary["Depot baseload (kW)"],
            "Charging electricity needed (MWh)": summary["Charging electricity needed (MWh)"],
            "12 m buses": summary["12 m buses"],
            "18 m buses": summary["18 m buses"],
            "Priority charging enabled": summary["Priority charging enabled"],
            "Priority 12 m buses": summary["Priority 12 m buses"],
            "Priority 18 m buses": summary["Priority 18 m buses"],
            "Status": summary["Status"],
        }

        if charging_strategy == "fixed_sessions":
            main_results["Number of sessions"] = summary["Number of sessions"]
            main_results["Session length (h)"] = summary["Session length (h)"]
            main_results["Total charging horizon (h)"] = summary["Total charging horizon (h)"]
            main_results["Session time used (h)"] = summary["Session time used (h)"]
            main_results["Unused time in charging horizon (h)"] = summary["Unused time in charging horizon (h)"]
        else:
            main_results["Total charging horizon (h)"] = summary["Total charging horizon (h)"]
            main_results["Priority charging time used (h)"] = summary["Priority charging time used (h)"]
            main_results["Remaining flexible charging time (h)"] = summary["Remaining flexible charging time (h)"]

        if mode == "check_known_limit":
            main_results["Available depot power (MW)"] = summary["Depot power limit entered by user (MW)"]
            main_results["Capacity gap above limit (MW)"] = summary["Capacity gap above limit (MW)"]
            main_results["Can the fleet be charged overnight?"] = summary["Can the fleet be charged overnight?"]

        main_results_df = pd.DataFrame(
            [{"Metric": key, "Value": value} for key, value in main_results.items()]
        )
        st.dataframe(main_results_df, use_container_width=True, hide_index=True)

        with st.expander("Advanced details"):
            advanced_items = dict(summary)
            for key in list(main_results.keys()):
                advanced_items.pop(key, None)

            advanced_df = pd.DataFrame(
                [{"Metric": key, "Value": value} for key, value in advanced_items.items()]
            )
            st.dataframe(advanced_df, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("24-hour load curve")

        display_start, display_end = get_24h_display_window(effective_start_dt)

        x_total, y_total_kw = build_step_series(
            intervals=intervals,
            display_start=display_start,
            display_end=display_end,
            base_kw=inputs.baseload_kw,
        )
        x_charging, y_charging_kw = build_step_series(
            intervals=intervals,
            display_start=display_start,
            display_end=display_end,
            base_kw=0.0,
        )

        y_total_mw = [v / 1000 for v in y_total_kw]
        y_charging_mw = [v / 1000 for v in y_charging_kw]

        fig = go.Figure()

        shade_start = max(effective_start_dt, display_start)
        shade_end = min(charging_end_dt, display_end)

        if shade_end > shade_start:
            fig.add_vrect(
                x0=shade_start,
                x1=shade_end,
                fillcolor="lightgray",
                opacity=0.15,
                line_width=0,
                layer="below",
            )

        fig.add_trace(
            go.Scatter(
                x=x_total,
                y=y_total_mw,
                mode="lines",
                line=dict(width=3, shape="hv"),
                name="Total depot load",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_charging,
                y=y_charging_mw,
                mode="lines",
                line=dict(width=2, dash="dash", shape="hv"),
                name="Charging load",
            )
        )

        fig.update_layout(
            title="Depot load curve over 24 hours",
            xaxis_title="Time of day",
            yaxis_title="Load (MW)",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=80, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        fig.update_xaxes(tickformat="%H:%M", type="date")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(table_title)
    st.caption(table_note)

    display_table_df = table_df.copy()
    for col in ["Start", "End"]:
        if col in display_table_df.columns:
            display_table_df[col] = pd.to_datetime(display_table_df[col]).dt.strftime("%H:%M")

    st.dataframe(display_table_df, use_container_width=True, hide_index=True)

    st.subheader("Quick scenario comparison")

    scenario_fleet_increases = [0, 10, 20]
    scenario_rows = []

    for extra_buses in scenario_fleet_increases:
        scenario_inputs = DepotInputs(
            n_12m=inputs.n_12m + extra_buses,
            n_18m=inputs.n_18m,
            enable_priority=inputs.enable_priority,
            priority_12m=inputs.priority_12m,
            priority_18m=inputs.priority_18m,
            priority_arrival_time=inputs.priority_arrival_time,
            battery_12m_kwh=inputs.battery_12m_kwh,
            battery_18m_kwh=inputs.battery_18m_kwh,
            charger_power_kw=inputs.charger_power_kw,
            baseload_kw=inputs.baseload_kw,
            general_arrival_time=inputs.general_arrival_time,
            charging_end_time=inputs.charging_end_time,
            session_length_h=inputs.session_length_h,
            timestep_min=inputs.timestep_min,
            efficiency=inputs.efficiency,
            min_soc=inputs.min_soc,
            target_soc=inputs.target_soc,
            charging_strategy=inputs.charging_strategy,
            mode="estimate_capacity",
            grid_cap_kw=None,
        )

        try:
            scenario_results = run_model(scenario_inputs)
            scenario_summary = scenario_results["summary"]
            scenario_rows.append(
                {
                    "Scenario": f"+{extra_buses} extra 12 m buses" if extra_buses > 0 else "Current fleet",
                    "12 m buses": scenario_inputs.n_12m,
                    "18 m buses": scenario_inputs.n_18m,
                    "Required depot power (MW)": scenario_summary["Total depot peak load incl. baseload (MW)"],
                    "Charging electricity needed (MWh)": scenario_summary["Charging electricity needed (MWh)"],
                }
            )
        except Exception:
            scenario_rows.append(
                {
                    "Scenario": f"+{extra_buses} extra 12 m buses" if extra_buses > 0 else "Current fleet",
                    "12 m buses": scenario_inputs.n_12m,
                    "18 m buses": scenario_inputs.n_18m,
                    "Required depot power (MW)": "Not feasible",
                    "Charging electricity needed (MWh)": "Not feasible",
                }
            )

    scenario_df = pd.DataFrame(scenario_rows)
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        csv_bytes = display_table_df.to_csv(index=False).encode("utf-8")
        csv_name = (
            "depot_power_planner_sessions.csv"
            if charging_strategy == "fixed_sessions"
            else "depot_power_planner_smart_charging_blocks.csv"
        )
        st.download_button(
            "Download table CSV",
            data=csv_bytes,
            file_name=csv_name,
            mime="text/csv",
        )

    with download_col2:
        pdf_bytes = create_summary_pdf(
            inputs=inputs,
            main_results_df=main_results_df,
            advanced_df=advanced_df,
            table_df=display_table_df,
            scenario_df=scenario_df,
            table_title=table_title,
        )
        st.download_button(
            "Download summary PDF",
            data=pdf_bytes,
            file_name="depot_power_planner_summary.pdf",
            mime="application/pdf",
        )

    st.divider()

    st.subheader("About, funding, feedback and attribution")

    st.markdown(
        "**Depot Power Planner** is a simple planning tool for estimating charging power requirements "
        "for battery-electric bus depots. It is intended to support first-order planning and exploration "
        "of different charging strategies rather than detailed depot design.\n\n"
        "**Developed by:** Maria Xylia, Stockholm Environment Institute (SEI)\n\n"
        "**Funding:** This tool was developed within the **ResPT project**, funded by the "
        "**Swedish Energy Agency (Energimyndigheten)**.\n\n"
        "**Version:** Beta version (2026)\n\n"
        "**License:** Released under the **MIT License**.\n\n"
        "**Feedback:** This tool is currently in active development. Feedback is welcome on usability, "
        "assumptions, missing functionality, bugs, and ideas for future improvements. Please send comments "
        "and suggestions to **maria.xylia@sei.org**.\n\n"
        "**Suggested citation:**  \n"
        "Xylia, M. (2026). *Depot Power Planner*. Stockholm Environment Institute (SEI). "
        "DOI to be added in a future release.\n\n"
        "**Disclaimer:** This tool provides first-order planning estimates based on simplified charging "
        "assumptions. It is intended to support early-stage planning and should not replace detailed depot "
        "design or operational simulation."
    )

    st.info(
        "How this tool works:\n"
        "- All buses are assumed to be at the depot during the selected charging horizon.\n"
        "- All buses are assumed to return with the same battery level and leave at the same target battery level.\n"
        "- One charger is assumed per bus.\n"
        "- Depot baseload is added as a constant non-charging demand.\n"
        "- If priority charging is enabled, priority buses are assumed to need charging as soon as they arrive at the depot.\n"
        "- In fixed charging sessions mode, each bus must complete charging within one session only and cannot continue into later sessions.\n"
        "- In flexible smart charging mode, priority buses charge first and the rest of the fleet is then charged as flexibly as possible across the remaining horizon.\n"
        "- Charging losses are included, so electricity taken from the grid is slightly higher than the energy added to the battery.\n"
        "- This is a planning tool and does not represent full real-time depot operations."
    )

except Exception as e:
    st.error(str(e))
