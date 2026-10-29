"""
Milo Analytics — injury prevention with the ACWR.

Streamlit app for the AQX Sports Analytics Data Bowl 4.0.
Session load follows Foster's session-RPE method:
load = duration (min) × RPE (1 to 10).
The ACWR is the 7-day rolling mean divided by the 28-day rolling mean.
"""

from __future__ import annotations

import html
import io
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Logo colors: navy from MILO, green from ANALYTICS.
NAVY = "#102741"
GREEN = "#2fa77f"
LOGO_PATH = Path(__file__).resolve().parent / "logo.png"

# Protocol windows and thresholds (Hulin / Gabbett).
ACUTE_WINDOW = 7
CHRONIC_WINDOW = 28
OPTIMAL_LOW = 0.8
OPTIMAL_HIGH = 1.3
DANGER_THRESHOLD = 1.5
HISTORY_DAYS = 60

# Below this computed load, the last 6 days already fill the target:
# prescribe active recovery instead of a truncated session.
RECOVERY_LOAD_CEILING = 90.0

SOURCE_SYNTHETIC = "Synthetic data"
SOURCE_CSV = "Import a CSV"

# Credible duration by sport: (cap, preferred low, preferred high).
SESSION_SHAPE = {
    "Running": (110, 30, 95),
    "Football": (110, 30, 100),
    "Cycling": (210, 40, 180),
}
DEFAULT_SESSION_SHAPE = (150, 30, 100)

SPORTS: dict[str, dict] = {
    "Running": {
        "max_min": 110,
        # (duration min, RPE, km per minute)
        "week": [
            (50, 5, 0.160),
            (42, 8, 0.185),
            (32, 3, 0.145),
            (55, 7, 0.175),
            (48, 5, 0.160),
            (85, 6, 0.155),
            (0, 0, 0.0),
        ],
        "sunday_on": (40, 4, 0.150),
    },
    "Football": {
        "max_min": 110,
        "week": [
            (55, 5, 0.100),
            (70, 7, 0.110),
            (40, 4, 0.080),
            (75, 8, 0.120),
            (40, 3, 0.090),
            (95, 9, 0.110),
            (0, 0, 0.0),
        ],
        "sunday_on": (35, 4, 0.090),
    },
    "Cycling": {
        "max_min": 210,
        "week": [
            (70, 4, 0.48),
            (80, 7, 0.50),
            (50, 3, 0.42),
            (90, 6, 0.50),
            (60, 5, 0.48),
            (150, 6, 0.45),
            (0, 0, 0.0),
        ],
        "sunday_on": (45, 4, 0.42),
    },
}

SCENARIOS = (
    "Recent overload",
    "Optimal zone",
    "Return after a lull",
)

SCENARIO_HELP = {
    "Recent overload": (
        "Three base weeks, one lighter week, then a high block "
        "over the last 9 days. The ratio ends in the danger zone."
    ),
    "Optimal zone": (
        "Steady load with a mild wave. "
        "The ACWR stays between 0.80 and 1.30."
    ),
    "Return after a lull": (
        "Normal load, then a sharp volume drop over the last 18 days. "
        "The ACWR falls below 0.80."
    ),
}

COLUMN_ALIASES = {
    "date": ["date", "jour", "datetime", "day", "timestamp"],
    "distance_km": ["distance_km", "distance", "km", "dist_km", "distance_parcourue"],
    "duree_min": ["duree_min", "duree", "durée_min", "durée", "duration_min", "duration", "minutes", "temps_min"],
    "rpe_effort_1_10": ["rpe_effort_1_10", "rpe", "effort", "rpe_effort", "srpe"],
    "charge_totale": ["charge_totale", "charge", "load", "session_rpe", "ua"],
}


@dataclass
class Recommendation:
    """Tomorrow's session, derived from the latest ACWR."""

    ready: bool
    level: str
    zone: str
    color: str
    tone: str
    headline: str
    message: str
    session_type: str
    duration_min: int
    rpe: int
    target_load: float
    target_ratio: float
    raw_load: float
    sum_last_6: float
    target_acute: float
    acwr: float
    acute: float
    chronic: float
    as_of: str


def fmt_num(value: float | None, digits: int = 0) -> str:
    """Format a number in English (comma for thousands, point for decimals)."""
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def fmt_date(value) -> str:
    return pd.Timestamp(value).strftime("%b %d, %Y")


def assess_acwr(acwr: float) -> dict[str, str]:
    """Place an ACWR in a monitoring zone."""
    if acwr is None or (isinstance(acwr, (float, np.floating)) and np.isnan(acwr)):
        return {
            "level": "Waiting",
            "zone": "Not enough history",
            "color": "#94a3b8",
            "tone": "wait",
        }
    if acwr < OPTIMAL_LOW:
        return {"level": "Caution", "zone": "Undertraining", "color": "#fbbf24", "tone": "under"}
    if acwr <= OPTIMAL_HIGH:
        return {"level": "Safe", "zone": "Optimal zone", "color": GREEN, "tone": "safe"}
    if acwr <= DANGER_THRESHOLD:
        return {"level": "Caution", "zone": "Caution zone", "color": "#fb923c", "tone": "caution"}
    return {"level": "Danger", "zone": "Overload", "color": "#fb7185", "tone": "danger"}


def _phase(scenario: str, day_index: int) -> tuple[float, int, bool]:
    """
    Parameters for one synthetic day.

    Returns (duration scale, RPE delta, rest on Sunday).
    Indexes assume a 60-day history.
    """
    if scenario == "Recent overload":
        if 28 <= day_index <= 34:
            return 0.55, 0, True
        if day_index >= 51:
            return 1.28, 2, False
        return 1.0, 0, True
    if scenario == "Return after a lull":
        if day_index >= 42:
            return 0.45, -1, True
        return 1.0, 0, True
    # Optimal zone: a mild wave, with no spike and no cut.
    scale = 1.0 + 0.05 * float(np.sin(day_index / 5.0))
    return scale, 0, True


def generate_dataset(
    sport: str = "Running",
    scenario: str = "Recent overload",
    seed: int = 42,
    n_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """
    Build 60 days of realistic sessions.

    The weekly pattern (hard, easy, rest) stays in place.
    The scenario only changes volume and RPE, so the ACWR curve
    shows a recognizable field situation.
    """
    profile = SPORTS[sport]
    rng = np.random.default_rng(int(seed))
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    rows: list[dict] = []

    for index, date in enumerate(dates):
        scale, rpe_delta, rest_sunday = _phase(scenario, index)
        duration, rpe, km_per_min = profile["week"][int(date.dayofweek)]
        if int(date.dayofweek) == 6 and not rest_sunday:
            duration, rpe, km_per_min = profile["sunday_on"]

        if duration <= 0:
            distance = 0.0
            duration_out = 0
            rpe_out = 0
        else:
            noise_duration = float(rng.normal(1.0, 0.04))
            noise_distance = float(rng.normal(1.0, 0.04))
            duration_out = int(
                np.clip(round(duration * scale * noise_duration), 10, profile["max_min"])
            )
            rpe_out = int(np.clip(rpe + rpe_delta, 1, 10))
            distance = round(duration_out * km_per_min * noise_distance, 1)

        rows.append(
            {
                "date": date,
                "distance_km": distance,
                "duree_min": duration_out,
                "rpe_effort_1_10": rpe_out,
                "charge_totale": float(duration_out * rpe_out),
            }
        )

    return pd.DataFrame(rows)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recognize common headers, including French exports."""
    lookup = {str(column).strip().lower(): column for column in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                rename[lookup[alias]] = canonical
                break
    return df.rename(columns=rename)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Accept comma decimals (French Excel exports)."""
    if series.dtype == object:
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace("\u00a0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _read_uploaded_csv(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("This CSV could not be read. Check the separator and the quotes.") from exc


def load_user_csv(uploaded) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a session CSV and collapse it to one row per day.

    Several sessions on the same day are summed: the daily load
    is the sum of the session-RPE values, and the displayed RPE
    is the duration-weighted mean.
    """
    warnings: list[str] = []
    renamed = _normalize_columns(_read_uploaded_csv(uploaded))
    required = ["date", "duree_min", "rpe_effort_1_10"]
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        readable = ", ".join(missing)
        raise ValueError(
            "Missing columns: "
            f"{readable}. The file must contain the date, the duration in minutes, and the RPE (1 to 10)."
        )

    frame = pd.DataFrame()
    frame["date"] = pd.to_datetime(renamed["date"], dayfirst=True, format="mixed", errors="coerce")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    frame["duree_min"] = _to_numeric(renamed["duree_min"])
    frame["rpe_effort_1_10"] = _to_numeric(renamed["rpe_effort_1_10"])
    if "distance_km" in renamed.columns:
        frame["distance_km"] = _to_numeric(renamed["distance_km"])
    else:
        frame["distance_km"] = 0.0
        warnings.append("No distance column: kilometers are left at 0.")

    invalid_dates = int(frame["date"].isna().sum())
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        raise ValueError("No usable date. Use YYYY-MM-DD or DD/MM/YYYY.")
    if invalid_dates:
        warnings.append(f"{invalid_dates} row(s) skipped: the date could not be read.")

    if frame[["duree_min", "rpe_effort_1_10"]].isna().any().any():
        raise ValueError("Some durations or RPE values are not numeric.")
    if (frame["duree_min"] < 0).any() or (frame["distance_km"].fillna(0) < 0).any():
        raise ValueError("Duration and distance must be zero or positive.")

    active = frame["duree_min"] > 0
    rpe_active = frame.loc[active, "rpe_effort_1_10"]
    if ((rpe_active < 1) | (rpe_active > 10)).any():
        raise ValueError("Every session with a duration above 0 needs an RPE between 1 and 10.")

    provided_charge = None
    if "charge_totale" in renamed.columns:
        provided_charge = _to_numeric(renamed.loc[frame.index, "charge_totale"])

    frame["distance_km"] = frame["distance_km"].fillna(0)
    daily_rows: list[dict] = []
    for day, group in frame.groupby("date", sort=True):
        charge = float((group["duree_min"] * group["rpe_effort_1_10"]).sum())
        duration = float(group["duree_min"].sum())
        weighted_rpe = charge / duration if duration > 0 else 0.0
        daily_rows.append(
            {
                "date": pd.Timestamp(day),
                "distance_km": round(float(group["distance_km"].sum()), 1),
                "duree_min": round(duration, 1),
                "rpe_effort_1_10": round(weighted_rpe, 2),
                "charge_totale": charge,
            }
        )
    daily = pd.DataFrame(daily_rows)

    if provided_charge is not None:
        computed_total = float((frame["duree_min"] * frame["rpe_effort_1_10"]).sum())
        provided_total = float(provided_charge.fillna(0).sum())
        if abs(computed_total - provided_total) > max(1.0, 0.02 * computed_total):
            warnings.append(
                "The load in the file did not match duration × RPE. "
                "Milo Analytics recomputed it with the session-RPE method."
            )

    return daily, warnings


def compute_workloads(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Compute acute load, chronic load, and ACWR.

    Missing days are filled with zero load: a rolling mean must cover
    calendar days, not only the sessions that were logged.
    The ACWR is defined from day 28, and only when chronic load
    is strictly positive.

    Coupled model: the 7-day window sits inside the 28-day window,
    as in the original rolling-mean formulation of the ACWR.
    """
    out = df.sort_values("date").drop_duplicates("date", keep="last").copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    calendar = pd.date_range(out["date"].min(), out["date"].max(), freq="D")
    out = out.set_index("date").reindex(calendar)
    imputed = int(out["charge_totale"].isna().sum())
    out.index.name = "date"
    for column in ("distance_km", "duree_min", "rpe_effort_1_10", "charge_totale"):
        out[column] = out[column].fillna(0.0)
    out = out.reset_index()

    # Always recompute load so Load = duration × RPE, including imputed days (0 × 0).
    out["charge_totale"] = out["duree_min"] * out["rpe_effort_1_10"]
    out["charge_aigue"] = out["charge_totale"].rolling(ACUTE_WINDOW, min_periods=ACUTE_WINDOW).mean()
    out["charge_chronique"] = out["charge_totale"].rolling(
        CHRONIC_WINDOW, min_periods=CHRONIC_WINDOW
    ).mean()
    chronic = out["charge_chronique"].to_numpy(dtype=float)
    acute = out["charge_aigue"].to_numpy(dtype=float)
    ratio = np.divide(acute, chronic, out=np.full_like(acute, np.nan), where=chronic > 0)
    out["acwr"] = ratio
    out["zone"] = [assess_acwr(value)["zone"] for value in out["acwr"]]
    return out, imputed


def shape_session(
    load: float,
    rpe_low: int,
    rpe_high: int,
    sport: str | None,
) -> tuple[int, int, float]:
    """
    Pick an RPE and a duration close to the target load.

    Duration is kept inside the sport's usual range.
    A higher RPE is chosen when it avoids an unreasonably long session.
    """
    duration_cap, preferred_low, preferred_high = SESSION_SHAPE.get(sport, DEFAULT_SESSION_SHAPE)
    load = max(float(load), 0.0)
    best: tuple[float, int, int, float] | None = None
    for rpe in range(rpe_low, rpe_high + 1):
        raw_duration = load / rpe if load else float(preferred_low)
        duration = int(np.clip(round(raw_duration / 5.0) * 5, 20, duration_cap))
        actual = float(duration * rpe)
        in_range = preferred_low <= raw_duration <= min(preferred_high, duration_cap)
        score = abs(actual - load) + (0.0 if in_range else 1000.0)
        if best is None or score < best[0]:
            best = (score, rpe, duration, actual)
    assert best is not None
    return best[1], best[2], best[3]


def build_recommendation(df: pd.DataFrame, sport: str | None = None) -> Recommendation:
    """
    Propose tomorrow's session so the ACWR moves toward a target.

    Tomorrow, the acute window drops day J-6 and gains the unknown session L.
    Holding today's chronic load as an approximation (one day weighs little
    across 28), the load that targets a ratio T is:

        L = 7 × T × chronic − sum of the last 6 days

    If L is negative or negligible, the last 6 days already exceed the target:
    the only consistent option is active recovery.
    When undertraining, T is capped at ACWR + 0.10 to avoid a spike.
    """
    valid = df.dropna(subset=["acwr"])
    if valid.empty:
        return Recommendation(
            ready=False,
            level="Waiting",
            zone="Not enough history",
            color="#94a3b8",
            tone="wait",
            headline="Not enough history",
            message=(
                "The ACWR needs 28 days of load to estimate chronic fitness. "
                "Keep logging every session, including rest days (load 0). "
                "The recommendation appears on day 28."
            ),
            session_type="—",
            duration_min=0,
            rpe=0,
            target_load=0.0,
            target_ratio=float("nan"),
            raw_load=0.0,
            sum_last_6=0.0,
            target_acute=0.0,
            acwr=float("nan"),
            acute=float("nan"),
            chronic=float("nan"),
            as_of="—",
        )

    last = valid.iloc[-1]
    acwr = float(last["acwr"])
    acute = float(last["charge_aigue"])
    chronic = float(last["charge_chronique"])
    status = assess_acwr(acwr)
    as_of = fmt_date(last["date"])
    # Same row as the ratio: if chronic load later hits zero,
    # do not mix an older ACWR with the loads of the very last days.
    position = int(valid.index[-1])
    loads = df["charge_totale"].astype(float)
    sum_last_6 = float(loads.loc[max(0, position - 5) : position].sum())

    if acwr > DANGER_THRESHOLD:
        target_ratio = 1.25
        rpe_low, rpe_high = 3, 5
        session_type = "Recovery"
    elif acwr > OPTIMAL_HIGH:
        target_ratio = 1.15
        rpe_low, rpe_high = 3, 5
        session_type = "Easy endurance"
    elif acwr >= OPTIMAL_LOW:
        target_ratio = 1.05
        rpe_low, rpe_high = 4, 9
        session_type = "Normal training"
    else:
        target_ratio = min(0.90, acwr + 0.10)
        rpe_low, rpe_high = 4, 6
        session_type = "Gradual return"

    target_acute = target_ratio * chronic
    raw_load = target_acute * ACUTE_WINDOW - sum_last_6
    recent = loads.loc[max(0, position - 13) : position]
    recent_positive = recent[recent > 0]
    typical = float(recent_positive.median()) if len(recent_positive) else max(acute, 1.0)

    prescribe_recovery = acwr > OPTIMAL_HIGH and raw_load <= RECOVERY_LOAD_CEILING
    if prescribe_recovery:
        session_type = "Active recovery"
        rpe = 3
        duration = 25
        target_load = float(duration * rpe)
        headline = "Active recovery session recommended"
        where = (
            "above the danger threshold (1.50)"
            if acwr > DANGER_THRESHOLD
            else "in the caution zone (1.30 to 1.50)"
        )
        if raw_load <= 0:
            reason = (
                "Even with no load tomorrow, the acute mean would stay "
                f"above the target of {fmt_num(target_acute)} AU."
            )
        else:
            reason = (
                f"The remaining room to aim for a ratio of {fmt_num(target_ratio, 2)} "
                f"is only {fmt_num(raw_load)} AU."
            )
        message = (
            f"The ACWR on {as_of} is {fmt_num(acwr, 2)}, {where}. {reason} "
            f"Tomorrow: {duration} min at RPE {rpe}, about {fmt_num(target_load)} AU "
            "(mobility, a very easy jog, or a light ride). No intense work."
        )
    else:
        load_goal = max(0.0, raw_load)
        rpe, duration, target_load = shape_session(load_goal, rpe_low, rpe_high, sport)
        pct = (target_load - typical) / typical * 100.0 if typical else 0.0
        pct_rounded = int(np.round(abs(pct) / 5.0) * 5)

        if acwr > OPTIMAL_HIGH:
            if pct < 0 and pct_rounded >= 10:
                headline = f"Reduce intensity by {pct_rounded}%"
            else:
                headline = "Reduce the session intensity"
            zone_text = (
                "above the danger threshold (1.50)"
                if acwr > DANGER_THRESHOLD
                else "in the caution zone (1.30 to 1.50)"
            )
            message = (
                f"The ACWR on {as_of} is {fmt_num(acwr, 2)}, {zone_text}. "
                f"To aim for {fmt_num(target_ratio, 2)} tomorrow: {duration} min at RPE {rpe}, "
                f"or {fmt_num(target_load)} AU. "
                "Keep the session as easy endurance, with no sprinting and no threshold work."
            )
        elif acwr >= OPTIMAL_LOW:
            headline = "Normal training"
            message = (
                f"The ACWR on {as_of} is {fmt_num(acwr, 2)}, inside the optimal zone (0.80 to 1.30). "
                f"To hold it near {fmt_num(target_ratio, 2)}: {duration} min at RPE {rpe}, "
                f"or {fmt_num(target_load)} AU."
            )
            exiting = float(loads.loc[position - 6]) if position >= 6 else target_load
            if exiting < 40:
                message += (
                    " A very light day is leaving the 7-day window: "
                    "this duration keeps the ratio from rising only because of that shift."
                )
        else:
            headline = "Gradual return"
            message = (
                f"The ACWR on {as_of} is {fmt_num(acwr, 2)}, below 0.80. "
                "Chronic fitness has dropped: a sudden spike would raise the risk. "
                f"Tomorrow's target ratio is capped at {fmt_num(target_ratio, 2)} "
                "(no more than +0.10). "
                f"Session: {duration} min at RPE {rpe}, or {fmt_num(target_load)} AU."
            )

    return Recommendation(
        ready=True,
        level=status["level"],
        zone=status["zone"],
        color=status["color"],
        tone=status["tone"],
        headline=headline,
        message=message,
        session_type=session_type,
        duration_min=duration,
        rpe=rpe,
        target_load=target_load,
        target_ratio=target_ratio,
        raw_load=raw_load,
        sum_last_6=sum_last_6,
        target_acute=target_acute,
        acwr=acwr,
        acute=acute,
        chronic=chronic,
        as_of=as_of,
    )


def _apply_chart_style(fig: go.Figure) -> go.Figure:
    """Light theme aligned with the logo, without resetting axis titles already set."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI, sans-serif", color=NAVY, size=13),
        margin=dict(l=16, r=16, t=64, b=16),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GREEN, font_color=NAVY),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            title_text="",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=NAVY),
        ),
        title=dict(font=dict(size=18, color=NAVY)),
    )
    fig.update_xaxes(
        showgrid=False,
        tickformat="%b %d",
        tickfont=dict(color=NAVY),
        linecolor="rgba(16, 39, 65, 0.18)",
    )
    fig.update_yaxes(
        gridcolor="rgba(16, 39, 65, 0.10)",
        zeroline=False,
        tickfont=dict(color=NAVY),
        title_font=dict(color=NAVY),
    )
    return fig


def build_load_chart(df: pd.DataFrame) -> go.Figure:
    """Daily load bars and the 7-day / 28-day rolling means."""
    plot_df = df.copy()
    fig = px.bar(
        plot_df,
        x="date",
        y="charge_totale",
        labels={
            "charge_totale": "Daily load",
            "date": "Date",
            "distance_km": "Distance (km)",
            "duree_min": "Duration (min)",
            "rpe_effort_1_10": "RPE",
        },
        hover_data={
            "distance_km": ":.1f",
            "duree_min": ":.0f",
            "rpe_effort_1_10": ":.1f",
        },
    )
    fig.update_traces(
        name="Daily load",
        marker_color="rgba(16, 39, 65, 0.16)",
        marker_line_width=0,
        selector=dict(type="bar"),
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["charge_aigue"],
            name="Acute load (7 d)",
            mode="lines",
            line=dict(color=GREEN, width=2.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["charge_chronique"],
            name="Chronic load (28 d)",
            mode="lines",
            line=dict(color=NAVY, width=2.6),
        )
    )
    fig.update_layout(
        title="Acute and chronic load",
        yaxis_title="Arbitrary units (AU)",
        bargap=0.25,
    )
    _apply_chart_style(fig)
    fig.update_yaxes(rangemode="tozero", title="Arbitrary units (AU)")
    return fig


def build_acwr_chart(df: pd.DataFrame) -> go.Figure:
    """ACWR curve with the risk-zone bands."""
    plot_df = df.dropna(subset=["acwr"]).copy()
    y_max = max(2.0, float(plot_df["acwr"].max()) * 1.12)
    colors = [assess_acwr(value)["color"] for value in plot_df["acwr"]]

    fig = go.Figure()
    bands = [
        (0.0, OPTIMAL_LOW, "rgba(251, 191, 36, 0.16)", "Undertraining  < 0.80", "#fbbf24"),
        (OPTIMAL_LOW, OPTIMAL_HIGH, "rgba(47, 167, 127, 0.16)", "Optimal zone  0.80 – 1.30", GREEN),
        (OPTIMAL_HIGH, DANGER_THRESHOLD, "rgba(251, 146, 60, 0.16)", "Caution  1.30 – 1.50", "#fb923c"),
        (DANGER_THRESHOLD, y_max, "rgba(251, 113, 133, 0.16)", "Danger  > 1.50", "#fb7185"),
    ]
    # A real date with an empty y: the swatch stays in the legend
    # without creating a numeric axis that would hide the curve.
    legend_x = plot_df["date"].iloc[0]
    for y0, y1, fill, name, color in bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, line_width=0, layer="below")
        fig.add_trace(
            go.Scatter(
                x=[legend_x],
                y=[None],
                mode="markers",
                marker=dict(size=11, color=color, symbol="square"),
                name=name,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["acwr"],
            name="ACWR",
            mode="lines+markers",
            line=dict(color=NAVY, width=2.4),
            marker=dict(size=7, color=colors, line=dict(width=0)),
            hovertemplate="ACWR: %{y:.2f}<extra></extra>",
        )
    )

    for threshold in (OPTIMAL_LOW, OPTIMAL_HIGH, DANGER_THRESHOLD):
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="rgba(16, 39, 65, 0.28)",
            line_width=1,
        )

    last = plot_df.iloc[-1]
    fig.add_annotation(
        x=last["date"],
        y=float(last["acwr"]),
        text=fmt_num(float(last["acwr"]), 2),
        showarrow=True,
        arrowhead=2,
        arrowcolor=assess_acwr(float(last["acwr"]))["color"],
        ax=36,
        ay=-36,
        font=dict(color=NAVY, size=14),
        bgcolor="#ffffff",
        bordercolor=assess_acwr(float(last["acwr"]))["color"],
        borderwidth=1,
        borderpad=4,
    )
    fig.update_layout(title="ACWR trajectory", yaxis_title="ACWR")
    _apply_chart_style(fig)
    fig.update_xaxes(type="date")
    fig.update_yaxes(range=[0, y_max], title="ACWR")
    return fig


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(900px 320px at 0% -10%, rgba(47, 167, 127, 0.10), transparent 55%),
                radial-gradient(700px 280px at 100% 0%, rgba(16, 39, 65, 0.06), transparent 50%),
                #f4f8f7;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(16, 39, 65, 0.08); }
        .block-container { padding-top: 1.2rem; max-width: 1180px; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid rgba(16, 39, 65, 0.08);
            border-top: 3px solid #2fa77f;
            border-radius: 16px;
            padding: 14px 16px 8px 16px;
            box-shadow: 0 8px 24px rgba(16, 39, 65, 0.04);
        }
        [data-testid="stMetricLabel"] { color: #5c7284 !important; }
        [data-testid="stMetricValue"] { color: #102741; font-variant-numeric: tabular-nums; }
        .pp-kicker {
            color: #2fa77f; letter-spacing: 0.14em; text-transform: uppercase;
            font-size: 0.74rem; font-weight: 700; margin-bottom: 0.35rem;
        }
        .pp-sub { color: #3e5568; margin: 0; font-size: 1.02rem; line-height: 1.45; }
        .pp-banner {
            --accent: #94a3b8;
            border-radius: 16px;
            padding: 16px 18px 14px 18px;
            background: #ffffff;
            border: 1px solid rgba(16, 39, 65, 0.08);
            border-left: 5px solid var(--accent);
            margin: 0.2rem 0 0.6rem 0;
        }
        .pp-banner.pp-danger { --accent: #e11d48; }
        .pp-banner.pp-caution { --accent: #ea580c; }
        .pp-banner.pp-under { --accent: #d97706; }
        .pp-banner.pp-safe { --accent: #2fa77f; }
        .pp-banner.pp-wait { --accent: #64748b; }
        .pp-banner-kicker { color: var(--accent); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; }
        .pp-banner-title { color: #102741; font-size: 1.28rem; font-weight: 700; margin: 0.15rem 0 0.35rem 0; }
        .pp-banner-text { color: #3e5568; margin: 0; line-height: 1.45; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(athlete: str, sport: str | None, scenario: str | None, df: pd.DataFrame) -> None:
    context = html.escape(athlete.strip() or "Athlete")
    if sport:
        context += f" · {html.escape(sport)}"
    if scenario:
        context += f" · {html.escape(scenario)}"
    span = f"{fmt_date(df['date'].iloc[0])} – {fmt_date(df['date'].iloc[-1])} · {len(df)} days"
    logo_col, text_col = st.columns([1, 3], vertical_alignment="center")
    with logo_col:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=210)
    with text_col:
        st.markdown(
            f"""
            <div class="pp-kicker">AQX Sports Analytics Data Bowl 4.0</div>
            <p class="pp-sub">Prevent injury by keeping the load inside the optimal zone.
            <br>{context}<br>{span}</p>
            """,
            unsafe_allow_html=True,
        )


def _render_kpis(df: pd.DataFrame, recommendation: Recommendation) -> None:
    last = df.iloc[-1]
    acute = last["charge_aigue"]
    chronic = last["charge_chronique"]
    acute_label = fmt_num(acute, 0)
    chronic_label = fmt_num(chronic, 0)
    if recommendation.ready:
        acwr_label = fmt_num(recommendation.acwr, 2)
        acwr_delta = recommendation.zone
        risk = recommendation.level
    else:
        acwr_label = "—"
        acwr_delta = "28 days required"
        risk = "Waiting"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Acute load",
        f"{acute_label} AU",
        help="7-day rolling mean of load. This is recent fatigue.",
    )
    c2.metric(
        "Chronic load",
        f"{chronic_label} AU",
        help="28-day rolling mean of load. This is built fitness.",
    )
    c3.metric(
        "ACWR",
        acwr_label,
        delta=acwr_delta,
        delta_color="off",
        help="Acute load divided by chronic load. The optimal zone runs from 0.80 to 1.30.",
    )
    c4.metric(
        "Risk level",
        risk,
        help="Safe (0.80–1.30), Caution (< 0.80 or 1.30–1.50), Danger (> 1.50).",
    )


def _render_recommendation(recommendation: Recommendation) -> None:
    st.markdown(
        f"""
        <div class="pp-banner pp-{html.escape(recommendation.tone)}">
            <div class="pp-banner-kicker">{html.escape(recommendation.level)} · {html.escape(recommendation.zone)}</div>
            <div class="pp-banner-title">{html.escape(recommendation.headline)}</div>
            <p class="pp-banner-text">{html.escape(recommendation.message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not recommendation.ready:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tomorrow's session", recommendation.session_type)
    c2.metric("Target load", f"{fmt_num(recommendation.target_load)} AU")
    c3.metric("Suggested duration", f"{recommendation.duration_min} min")
    c4.metric("Suggested RPE", f"{recommendation.rpe} / 10")

    with st.expander("How this session is calculated"):
        st.markdown(
            textwrap.dedent(
                f"""
                Tomorrow's acute window drops the oldest day and adds the session being planned.

                - Sum of the last 6 days: **{fmt_num(recommendation.sum_last_6)} AU**
                - Current chronic load: **{fmt_num(recommendation.chronic)} AU**
                - Target ratio: **{fmt_num(recommendation.target_ratio, 2)}**
                - Target acute load: {fmt_num(recommendation.target_ratio, 2)} × {fmt_num(recommendation.chronic)} = **{fmt_num(recommendation.target_acute)} AU**
                - Raw load for tomorrow: 7 × {fmt_num(recommendation.target_acute)} − {fmt_num(recommendation.sum_last_6)} = **{fmt_num(recommendation.raw_load)} AU**

                The session shown rounds that target to a practical duration (5-minute steps) and an RPE suited to the zone.
                A negative raw value means even a zero load cannot reach the target tomorrow:
                the prescription is then the active-recovery ceiling (25 min × RPE 3).
                """
            ).strip()
        )


def _render_table(df: pd.DataFrame) -> None:
    view = df.copy()
    view["date"] = view["date"].dt.date
    # The table shows rounded values. The exported CSV keeps full precision.
    view["distance_km"] = view["distance_km"].round(1)
    view["duree_min"] = view["duree_min"].round(0)
    view["rpe_effort_1_10"] = view["rpe_effort_1_10"].round(1)
    view["charge_totale"] = view["charge_totale"].round(0)
    view["charge_aigue"] = view["charge_aigue"].round(0)
    view["charge_chronique"] = view["charge_chronique"].round(0)
    view["acwr"] = view["acwr"].round(2)
    column_config = {
        "date": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
        "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.1f"),
        "duree_min": st.column_config.NumberColumn("Duration (min)", format="%.0f"),
        "rpe_effort_1_10": st.column_config.NumberColumn("RPE", format="%.1f"),
        "charge_totale": st.column_config.NumberColumn("Load (AU)", format="%.0f"),
        "charge_aigue": st.column_config.NumberColumn("Acute (AU)", format="%.0f"),
        "charge_chronique": st.column_config.NumberColumn("Chronic (AU)", format="%.0f"),
        "acwr": st.column_config.NumberColumn("ACWR", format="%.2f"),
        "zone": st.column_config.TextColumn("Zone"),
    }
    st.markdown("**Last seven days**")
    st.dataframe(
        view.tail(7),
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Full history"):
        st.dataframe(
            view,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
        )


def _shuffle_seed() -> None:
    st.session_state.seed = int(np.random.default_rng().integers(0, 10_000))


def _sidebar() -> dict:
    if LOGO_PATH.is_file():
        st.sidebar.image(str(LOGO_PATH), width=220)
    source = st.sidebar.radio(
        "Data source",
        (SOURCE_SYNTHETIC, SOURCE_CSV),
        key="source_mode",
    )
    athlete = st.sidebar.text_input("Athlete", value="Demo athlete", key="athlete_name")

    sport = None
    scenario = None
    seed = 42
    uploaded = None
    if source == SOURCE_SYNTHETIC:
        sport = st.sidebar.selectbox("Sport", list(SPORTS), key="sport")
        scenario = st.sidebar.selectbox("Demo scenario", SCENARIOS, key="scenario")
        st.sidebar.caption(SCENARIO_HELP[scenario])
        if "seed" not in st.session_state:
            st.session_state.seed = 42
        seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=9999, step=1, key="seed"))
        st.sidebar.button("Regenerate dataset", on_click=_shuffle_seed, use_container_width=True)
    else:
        uploaded = st.sidebar.file_uploader("CSV file", type=["csv"], key="csv_file")
        st.sidebar.caption(
            "Expected columns: date, duree_min, rpe_effort_1_10. "
            "distance_km is optional. Comma or semicolon separator."
        )

    with st.sidebar.expander("Calculation protocol"):
        st.markdown(
            textwrap.dedent(
                f"""
                **Session-RPE (Foster)**
                Daily load = duration (min) × RPE (1 to 10).

                **Acute load** — {ACUTE_WINDOW}-day rolling mean.
                **Chronic load** — {CHRONIC_WINDOW}-day rolling mean.
                **ACWR** = acute / chronic.

                - **< 0.80** — undertraining
                - **0.80 – 1.30** — optimal zone
                - **1.30 – 1.50** — caution
                - **> 1.50** — overload risk
                """
            ).strip()
        )
        st.caption(
            "A monitoring indicator, not a diagnosis. "
            "The coupled ACWR includes the 7 acute days inside the 28 chronic days."
        )

    return {
        "source": source,
        "athlete": athlete,
        "sport": sport,
        "scenario": scenario,
        "seed": seed,
        "uploaded": uploaded,
    }


def _load_from_sidebar(choices: dict) -> tuple[pd.DataFrame, list[str], int]:
    if choices["source"] == SOURCE_SYNTHETIC:
        raw = generate_dataset(choices["sport"], choices["scenario"], choices["seed"])
        warnings: list[str] = []
    else:
        if choices["uploaded"] is None:
            st.info("Import a session CSV, or switch back to the synthetic data.")
            st.stop()
        raw, warnings = load_user_csv(choices["uploaded"])
    workloads, imputed = compute_workloads(raw)
    return workloads, warnings, imputed


def main() -> None:
    st.set_page_config(
        page_title="Milo Analytics",
        page_icon=str(LOGO_PATH) if LOGO_PATH.is_file() else "📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    choices = _sidebar()

    try:
        df, warnings, imputed = _load_from_sidebar(choices)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    for warning in warnings:
        st.warning(warning)
    if imputed:
        st.caption(
            f"{imputed} calendar day(s) without a session were counted as zero load, "
            "so the rolling means cover real days."
        )

    recommendation = build_recommendation(df, choices["sport"])
    _render_header(choices["athlete"], choices["sport"], choices["scenario"], df)
    _render_kpis(df, recommendation)

    st.markdown("### Decision support")
    with st.container(border=True):
        _render_recommendation(recommendation)

    with st.container(border=True):
        st.plotly_chart(
            build_load_chart(df),
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
        st.caption(
            "Bars: daily load (duration × RPE). "
            "Green curve: recent fatigue (7 days). "
            "Navy curve: built fitness (28 days)."
        )

    with st.container(border=True):
        if df["acwr"].notna().sum() < 2:
            st.info(
                "The ACWR curve appears on day 28, once chronic load can be computed. "
                "The load bars above are available immediately."
            )
        else:
            st.plotly_chart(
                build_acwr_chart(df),
                use_container_width=True,
                config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            )
            st.caption(
                "Yellow: undertraining (< 0.80). Green: optimal zone (0.80–1.30). "
                "Orange: caution (1.30–1.50). Red: overload and injury risk (> 1.50)."
            )

    with st.container(border=True):
        _render_table(df)
        export = df.copy()
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
        st.download_button(
            "Download the computed history (CSV)",
            data=export.to_csv(index=False).encode("utf-8-sig"),
            file_name="milo_analytics_acwr.csv",
            mime="text/csv",
            use_container_width=False,
        )

    st.caption(
        "Milo Analytics helps plan tomorrow's session. "
        "It is not a medical device: pain, unusual fatigue, or a previous injury "
        "should be discussed with the medical staff. "
        "The ACWR is a group indicator, not a certain individual prediction."
    )


if __name__ == "__main__":
    main()
