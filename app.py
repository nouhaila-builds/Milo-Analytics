"""
Milo Analytics — prévention des blessures par le ratio ACWR.

Application Streamlit pour l'AQX Sports Analytics Data Bowl 4.0.
La charge de séance suit la méthode session-RPE de Foster :
charge = durée (min) × RPE (1 à 10).
L'ACWR est le rapport de la moyenne mobile 7 jours sur la moyenne mobile 28 jours.
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

# Couleurs du logo : marine du mot MILO, vert du mot ANALYTICS.
NAVY = "#102741"
GREEN = "#2fa77f"
LOGO_PATH = Path(__file__).resolve().parent / "logo.png"

# Fenêtres et seuils du protocole (Hulin / Gabbett).
ACUTE_WINDOW = 7
CHRONIC_WINDOW = 28
OPTIMAL_LOW = 0.8
OPTIMAL_HIGH = 1.3
DANGER_THRESHOLD = 1.5
HISTORY_DAYS = 60

# En dessous de cette charge calculée, les 6 derniers jours saturent déjà
# la cible : on prescrit une récupération active plutôt qu'une séance tronquée.
RECOVERY_LOAD_CEILING = 90.0

# Durée crédible par sport : (plafond, préférence basse, préférence haute).
SESSION_SHAPE = {
    "Course à pied": (110, 30, 95),
    "Football": (110, 30, 100),
    "Cyclisme": (210, 40, 180),
}
DEFAULT_SESSION_SHAPE = (150, 30, 100)

SPORTS: dict[str, dict] = {
    "Course à pied": {
        "max_min": 110,
        # (durée min, RPE, km par minute)
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
    "Cyclisme": {
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
    "Surcharge récente",
    "Zone optimale",
    "Reprise après relâche",
)

SCENARIO_HELP = {
    "Surcharge récente": (
        "Trois semaines de base, une semaine allégée, puis un bloc élevé "
        "sur les 9 derniers jours. Le ratio termine en zone de danger."
    ),
    "Zone optimale": (
        "Charge régulière avec une légère ondulation. "
        "L'ACWR reste entre 0,80 et 1,30."
    ),
    "Reprise après relâche": (
        "Charge normale, puis volume fortement réduit sur les 18 derniers jours. "
        "L'ACWR passe sous 0,80."
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
    """Prescription de la séance du lendemain, dérivée du dernier ACWR."""

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
    """Formate un nombre en français (espace pour les milliers, virgule décimale)."""
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    formatted = f"{float(value):,.{digits}f}"
    return formatted.replace(",", " ").replace(".", ",")


def fmt_date(value) -> str:
    return pd.Timestamp(value).strftime("%d/%m/%Y")


def assess_acwr(acwr: float) -> dict[str, str]:
    """Classe un ACWR dans les zones de surveillance."""
    if acwr is None or (isinstance(acwr, (float, np.floating)) and np.isnan(acwr)):
        return {
            "level": "En attente",
            "zone": "Historique insuffisant",
            "color": "#94a3b8",
            "tone": "wait",
        }
    if acwr < OPTIMAL_LOW:
        return {"level": "Attention", "zone": "Sous-entraînement", "color": "#fbbf24", "tone": "under"}
    if acwr <= OPTIMAL_HIGH:
        return {"level": "Sécurité", "zone": "Zone optimale", "color": GREEN, "tone": "safe"}
    if acwr <= DANGER_THRESHOLD:
        return {"level": "Attention", "zone": "Zone d'attention", "color": "#fb923c", "tone": "caution"}
    return {"level": "Danger", "zone": "Surcharge", "color": "#fb7185", "tone": "danger"}


def _phase(scenario: str, day_index: int) -> tuple[float, int, bool]:
    """
    Paramètres du jour synthétique.

    Retourne (échelle de durée, delta de RPE, repos le dimanche).
    Les indices supposent un historique de 60 jours.
    """
    if scenario == "Surcharge récente":
        if 28 <= day_index <= 34:
            return 0.55, 0, True
        if day_index >= 51:
            return 1.28, 2, False
        return 1.0, 0, True
    if scenario == "Reprise après relâche":
        if day_index >= 42:
            return 0.45, -1, True
        return 1.0, 0, True
    # Zone optimale : ondulation douce, sans pic ni coupure.
    scale = 1.0 + 0.05 * float(np.sin(day_index / 5.0))
    return scale, 0, True


def generate_dataset(
    sport: str = "Course à pied",
    scenario: str = "Surcharge récente",
    seed: int = 42,
    n_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """
    Génère 60 jours de séances réalistes.

    La structure hebdomadaire (dure, facile, repos) est conservée.
    Le scénario ne change que le volume et le RPE, pour que la courbe
    ACWR illustre une situation de terrain identifiable.
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
    """Reconnaît les en-têtes courants, y compris les exports français."""
    lookup = {str(column).strip().lower(): column for column in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                rename[lookup[alias]] = canonical
                break
    return df.rename(columns=rename)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Accepte les décimales avec virgule (exports Excel français)."""
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
        raise ValueError("Le fichier CSV est vide.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("Impossible de lire ce CSV. Vérifiez le séparateur et les guillemets.") from exc


def load_user_csv(uploaded) -> tuple[pd.DataFrame, list[str]]:
    """
    Charge un CSV de séances et le ramène à une ligne par jour.

    Plusieurs séances le même jour sont additionnées : la charge du jour
    est la somme des session-RPE, et le RPE affiché est la moyenne
    pondérée par la durée.
    """
    warnings: list[str] = []
    renamed = _normalize_columns(_read_uploaded_csv(uploaded))
    required = ["date", "duree_min", "rpe_effort_1_10"]
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        readable = ", ".join(missing)
        raise ValueError(
            "Colonnes manquantes : "
            f"{readable}. Le fichier doit contenir la date, la durée en minutes et le RPE (1 à 10)."
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
        warnings.append("Colonne de distance absente : les kilomètres sont laissés à 0.")

    invalid_dates = int(frame["date"].isna().sum())
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        raise ValueError("Aucune date exploitable. Utilisez le format AAAA-MM-JJ ou JJ/MM/AAAA.")
    if invalid_dates:
        warnings.append(f"{invalid_dates} ligne(s) ignorée(s) : date illisible.")

    if frame[["duree_min", "rpe_effort_1_10"]].isna().any().any():
        raise ValueError("Certaines durées ou certains RPE ne sont pas numériques.")
    if (frame["duree_min"] < 0).any() or (frame["distance_km"].fillna(0) < 0).any():
        raise ValueError("La durée et la distance doivent être positives ou nulles.")

    active = frame["duree_min"] > 0
    rpe_active = frame.loc[active, "rpe_effort_1_10"]
    if ((rpe_active < 1) | (rpe_active > 10)).any():
        raise ValueError("Chaque séance avec une durée > 0 doit avoir un RPE entre 1 et 10.")

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
                "La charge du fichier différait de durée × RPE. "
                "Milo Analytics a recalculé la charge avec la méthode session-RPE."
            )

    return daily, warnings


def compute_workloads(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Calcule charge aiguë, charge chronique et ACWR.

    Les jours sans ligne sont complétés à charge nulle : une moyenne mobile
    doit porter sur des jours calendaires, pas seulement sur les séances saisies.
    L'ACWR n'est défini qu'à partir du 28e jour, et seulement si la chronique
    est strictement positive.

    Modèle couplé : la fenêtre de 7 jours est incluse dans celle de 28 jours,
    comme dans la formulation historique de l'ACWR par moyennes mobiles.
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

    # On recalcule toujours la charge pour garantir Charge = durée × RPE,
    # y compris sur les jours imputés (0 × 0).
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
    Choisit un couple RPE / durée proche de la charge cible.

    On privilégie une durée dans la plage habituelle du sport.
    Un RPE plus haut est retenu s'il évite une séance déraisonnablement longue.
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
    Propose la séance de demain pour rapprocher l'ACWR d'une cible.

    Demain, la fenêtre aiguë perd le jour J-6 et gagne la séance inconnue L.
    En gardant la chronique du jour comme approximation (un jour pèse peu
    sur 28), la charge qui vise un ratio T est :

        L = 7 × T × chronique − somme des 6 derniers jours

    Si L est négatif ou négligeable, les 6 derniers jours dépassent déjà
    la cible : la seule option cohérente est une récupération active.
    En sous-entraînement, T est plafonné à ACWR + 0,10 pour éviter un pic.
    """
    valid = df.dropna(subset=["acwr"])
    if valid.empty:
        return Recommendation(
            ready=False,
            level="En attente",
            zone="Historique insuffisant",
            color="#94a3b8",
            tone="wait",
            headline="Historique insuffisant",
            message=(
                "L'ACWR a besoin de 28 jours de charge pour estimer la condition chronique. "
                "Continuez à enregistrer chaque séance, y compris les jours de repos (charge 0). "
                "La recommandation s'affichera au 28e jour."
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
    # Même ligne que le ratio : si la chronique retombe à zéro en fin de série,
    # on ne mélange pas un ACWR ancien avec les charges des tout derniers jours.
    position = int(valid.index[-1])
    loads = df["charge_totale"].astype(float)
    sum_last_6 = float(loads.loc[max(0, position - 5) : position].sum())

    if acwr > DANGER_THRESHOLD:
        target_ratio = 1.25
        rpe_low, rpe_high = 3, 5
        session_type = "Récupération"
    elif acwr > OPTIMAL_HIGH:
        target_ratio = 1.15
        rpe_low, rpe_high = 3, 5
        session_type = "Endurance légère"
    elif acwr >= OPTIMAL_LOW:
        target_ratio = 1.05
        rpe_low, rpe_high = 4, 9
        session_type = "Entraînement normal"
    else:
        target_ratio = min(0.90, acwr + 0.10)
        rpe_low, rpe_high = 4, 6
        session_type = "Reprise progressive"

    target_acute = target_ratio * chronic
    raw_load = target_acute * ACUTE_WINDOW - sum_last_6
    recent = loads.loc[max(0, position - 13) : position]
    recent_positive = recent[recent > 0]
    typical = float(recent_positive.median()) if len(recent_positive) else max(acute, 1.0)

    prescribe_recovery = acwr > OPTIMAL_HIGH and raw_load <= RECOVERY_LOAD_CEILING
    if prescribe_recovery:
        session_type = "Récupération active"
        rpe = 3
        duration = 25
        target_load = float(duration * rpe)
        headline = "Séance de récupération active recommandée"
        where = (
            "au-dessus du seuil de danger (1,50)"
            if acwr > DANGER_THRESHOLD
            else "dans la zone d'attention (1,30 à 1,50)"
        )
        if raw_load <= 0:
            reason = (
                "Même sans aucune charge demain, la moyenne aiguë resterait "
                f"au-dessus de la cible de {fmt_num(target_acute)} UA."
            )
        else:
            reason = (
                f"La marge restante pour viser un ratio de {fmt_num(target_ratio, 2)} "
                f"n'est que de {fmt_num(raw_load)} UA."
            )
        message = (
            f"L'ACWR du {as_of} est à {fmt_num(acwr, 2)}, {where}. {reason} "
            f"Demain : {duration} min à RPE {rpe}, soit environ {fmt_num(target_load)} UA "
            "(mobilité, footing très souple ou vélo léger). Pas de travail intense."
        )
    else:
        load_goal = max(0.0, raw_load)
        rpe, duration, target_load = shape_session(load_goal, rpe_low, rpe_high, sport)
        pct = (target_load - typical) / typical * 100.0 if typical else 0.0
        pct_rounded = int(np.round(abs(pct) / 5.0) * 5)

        if acwr > OPTIMAL_HIGH:
            if pct < 0 and pct_rounded >= 10:
                headline = f"Diminuer l'intensité de {pct_rounded} %"
            else:
                headline = "Diminuer l'intensité de la séance"
            zone_text = (
                "au-dessus du seuil de danger (1,50)"
                if acwr > DANGER_THRESHOLD
                else "dans la zone d'attention (1,30 à 1,50)"
            )
            message = (
                f"L'ACWR du {as_of} est à {fmt_num(acwr, 2)}, {zone_text}. "
                f"Pour viser {fmt_num(target_ratio, 2)} dès demain : {duration} min à RPE {rpe}, "
                f"soit {fmt_num(target_load)} UA. "
                "Gardez la séance en endurance légère, sans sprint ni travail au seuil."
            )
        elif acwr >= OPTIMAL_LOW:
            headline = "Entraînement normal"
            message = (
                f"L'ACWR du {as_of} est à {fmt_num(acwr, 2)}, dans la zone optimale (0,80 à 1,30). "
                f"Pour le maintenir près de {fmt_num(target_ratio, 2)} : {duration} min à RPE {rpe}, "
                f"soit {fmt_num(target_load)} UA."
            )
            exiting = float(loads.loc[position - 6]) if position >= 6 else target_load
            if exiting < 40:
                message += (
                    " Un jour très léger sort de la fenêtre de 7 jours : "
                    "cette durée évite de faire monter le ratio uniquement à cause de ce glissement."
                )
        else:
            headline = "Reprise progressive"
            message = (
                f"L'ACWR du {as_of} est à {fmt_num(acwr, 2)}, sous 0,80. "
                "La condition chronique a baissé : un pic brutal augmenterait le risque. "
                f"Le ratio cible de demain est plafonné à {fmt_num(target_ratio, 2)} "
                "(pas plus de +0,10). "
                f"Séance : {duration} min à RPE {rpe}, soit {fmt_num(target_load)} UA."
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
    """Thème clair aligné sur le logo, sans réinitialiser les titres d'axes déjà posés."""
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
        tickformat="%d/%m",
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
    """Barres de charge quotidienne et courbes des moyennes mobiles 7 j / 28 j."""
    plot_df = df.copy()
    fig = px.bar(
        plot_df,
        x="date",
        y="charge_totale",
        labels={"charge_totale": "Charge du jour", "date": "Date"},
        hover_data={
            "distance_km": ":.1f",
            "duree_min": ":.0f",
            "rpe_effort_1_10": ":.1f",
        },
    )
    fig.update_traces(
        name="Charge du jour",
        marker_color="rgba(16, 39, 65, 0.16)",
        marker_line_width=0,
        selector=dict(type="bar"),
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["charge_aigue"],
            name="Charge aiguë (7 j)",
            mode="lines",
            line=dict(color=GREEN, width=2.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["charge_chronique"],
            name="Charge chronique (28 j)",
            mode="lines",
            line=dict(color=NAVY, width=2.6),
        )
    )
    fig.update_layout(
        title="Charge aiguë et charge chronique",
        yaxis_title="Unités arbitraires (UA)",
        bargap=0.25,
    )
    _apply_chart_style(fig)
    fig.update_yaxes(rangemode="tozero", title="Unités arbitraires (UA)")
    return fig


def build_acwr_chart(df: pd.DataFrame) -> go.Figure:
    """Courbe d'ACWR avec les bandes de zone de risque."""
    plot_df = df.dropna(subset=["acwr"]).copy()
    y_max = max(2.0, float(plot_df["acwr"].max()) * 1.12)
    colors = [assess_acwr(value)["color"] for value in plot_df["acwr"]]

    fig = go.Figure()
    bands = [
        (0.0, OPTIMAL_LOW, "rgba(251, 191, 36, 0.16)", "Sous-entraînement  < 0,80", "#fbbf24"),
        (OPTIMAL_LOW, OPTIMAL_HIGH, "rgba(47, 167, 127, 0.16)", "Zone optimale  0,80 – 1,30", GREEN),
        (OPTIMAL_HIGH, DANGER_THRESHOLD, "rgba(251, 146, 60, 0.16)", "Attention  1,30 – 1,50", "#fb923c"),
        (DANGER_THRESHOLD, y_max, "rgba(251, 113, 133, 0.16)", "Danger  > 1,50", "#fb7185"),
    ]
    for y0, y1, fill, name, color in bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, line_width=0, layer="below")
        fig.add_trace(
            go.Scatter(
                x=[None],
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
            hovertemplate="ACWR : %{y:.2f}<extra></extra>",
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
    fig.update_layout(title="Trajectoire de l'ACWR", yaxis_title="ACWR")
    _apply_chart_style(fig)
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
    context = html.escape(athlete.strip() or "Athlète")
    if sport:
        context += f" · {html.escape(sport)}"
    if scenario:
        context += f" · {html.escape(scenario)}"
    span = f"{fmt_date(df['date'].iloc[0])} – {fmt_date(df['date'].iloc[-1])} · {len(df)} jours"
    logo_col, text_col = st.columns([1, 3], vertical_alignment="center")
    with logo_col:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=210)
    with text_col:
        st.markdown(
            f"""
            <div class="pp-kicker">AQX Sports Analytics Data Bowl 4.0</div>
            <p class="pp-sub">Prévenir la blessure en gardant la charge dans la zone optimale.
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
        acwr_delta = "28 jours requis"
        risk = "En attente"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Charge aiguë",
        f"{acute_label} UA",
        help="Moyenne mobile de la charge sur 7 jours. C'est la fatigue récente.",
    )
    c2.metric(
        "Charge chronique",
        f"{chronic_label} UA",
        help="Moyenne mobile de la charge sur 28 jours. C'est la condition construite.",
    )
    c3.metric(
        "ACWR",
        acwr_label,
        delta=acwr_delta,
        delta_color="off",
        help="Charge aiguë divisée par la charge chronique. La zone optimale va de 0,80 à 1,30.",
    )
    c4.metric(
        "Niveau de risque",
        risk,
        help="Sécurité (0,80–1,30), Attention (< 0,80 ou 1,30–1,50), Danger (> 1,50).",
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
    c1.metric("Séance de demain", recommendation.session_type)
    c2.metric("Charge cible", f"{fmt_num(recommendation.target_load)} UA")
    c3.metric("Durée suggérée", f"{recommendation.duration_min} min")
    c4.metric("RPE suggéré", f"{recommendation.rpe} / 10")

    with st.expander("Comment cette séance est calculée"):
        st.markdown(
            textwrap.dedent(
                f"""
                La fenêtre aiguë de demain abandonne le jour le plus ancien et ajoute la séance à planifier.

                - Somme des 6 derniers jours : **{fmt_num(recommendation.sum_last_6)} UA**
                - Charge chronique actuelle : **{fmt_num(recommendation.chronic)} UA**
                - Ratio cible : **{fmt_num(recommendation.target_ratio, 2)}**
                - Charge aiguë visée : {fmt_num(recommendation.target_ratio, 2)} × {fmt_num(recommendation.chronic)} = **{fmt_num(recommendation.target_acute)} UA**
                - Charge brute de demain : 7 × {fmt_num(recommendation.target_acute)} − {fmt_num(recommendation.sum_last_6)} = **{fmt_num(recommendation.raw_load)} UA**

                La séance affichée arrondit cette cible à une durée pratique (pas de 5 min) et à un RPE adapté à la zone.
                Une valeur brute négative signifie que même une charge nulle ne suffit pas à atteindre la cible dès demain :
                on prescrit alors le plafond de récupération active (25 min × RPE 3).
                """
            ).strip()
        )


def _render_table(df: pd.DataFrame) -> None:
    view = df.copy()
    view["date"] = view["date"].dt.date
    # Le tableau affiche des valeurs arrondies. Le CSV exporté garde la précision complète.
    view["distance_km"] = view["distance_km"].round(1)
    view["duree_min"] = view["duree_min"].round(0)
    view["rpe_effort_1_10"] = view["rpe_effort_1_10"].round(1)
    view["charge_totale"] = view["charge_totale"].round(0)
    view["charge_aigue"] = view["charge_aigue"].round(0)
    view["charge_chronique"] = view["charge_chronique"].round(0)
    view["acwr"] = view["acwr"].round(2)
    column_config = {
        "date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
        "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.1f"),
        "duree_min": st.column_config.NumberColumn("Durée (min)", format="%.0f"),
        "rpe_effort_1_10": st.column_config.NumberColumn("RPE", format="%.1f"),
        "charge_totale": st.column_config.NumberColumn("Charge (UA)", format="%.0f"),
        "charge_aigue": st.column_config.NumberColumn("Aiguë (UA)", format="%.0f"),
        "charge_chronique": st.column_config.NumberColumn("Chronique (UA)", format="%.0f"),
        "acwr": st.column_config.NumberColumn("ACWR", format="%.2f"),
        "zone": st.column_config.TextColumn("Zone"),
    }
    st.markdown("**Sept derniers jours**")
    st.dataframe(
        view.tail(7),
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Historique complet"):
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
        "Source des données",
        ("Données synthétiques", "Importer un CSV"),
        key="source_mode",
    )
    athlete = st.sidebar.text_input("Athlète", value="Athlète démo", key="athlete_name")

    sport = None
    scenario = None
    seed = 42
    uploaded = None
    if source == "Données synthétiques":
        sport = st.sidebar.selectbox("Sport", list(SPORTS), key="sport")
        scenario = st.sidebar.selectbox("Scénario de démonstration", SCENARIOS, key="scenario")
        st.sidebar.caption(SCENARIO_HELP[scenario])
        if "seed" not in st.session_state:
            st.session_state.seed = 42
        seed = int(st.sidebar.number_input("Graine aléatoire", min_value=0, max_value=9999, step=1, key="seed"))
        st.sidebar.button("Régénérer le jeu", on_click=_shuffle_seed, use_container_width=True)
    else:
        uploaded = st.sidebar.file_uploader("Fichier CSV", type=["csv"], key="csv_file")
        st.sidebar.caption(
            "Colonnes attendues : date, duree_min, rpe_effort_1_10. "
            "distance_km est optionnelle. Séparateur virgule ou point-virgule."
        )

    with st.sidebar.expander("Protocole de calcul"):
        st.markdown(
            textwrap.dedent(
                f"""
                **Session-RPE (Foster)**
                Charge du jour = durée (min) × RPE (1 à 10).

                **Charge aiguë** — moyenne mobile de {ACUTE_WINDOW} jours.
                **Charge chronique** — moyenne mobile de {CHRONIC_WINDOW} jours.
                **ACWR** = aiguë / chronique.

                - **< 0,80** — sous-entraînement
                - **0,80 – 1,30** — zone optimale
                - **1,30 – 1,50** — attention
                - **> 1,50** — risque de surcharge
                """
            ).strip()
        )
        st.caption(
            "Indicateur de surveillance, pas un diagnostic. "
            "L'ACWR couplé inclut les 7 jours aigus dans les 28 jours chroniques."
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
    if choices["source"] == "Données synthétiques":
        raw = generate_dataset(choices["sport"], choices["scenario"], choices["seed"])
        warnings: list[str] = []
    else:
        if choices["uploaded"] is None:
            st.info("Importez un CSV de séances, ou repassez sur les données synthétiques.")
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
            f"{imputed} jour(s) sans séance dans le calendrier ont été comptés à charge nulle, "
            "afin que les moyennes mobiles portent sur des jours réels."
        )

    recommendation = build_recommendation(df, choices["sport"])
    _render_header(choices["athlete"], choices["sport"], choices["scenario"], df)
    _render_kpis(df, recommendation)

    st.markdown("### Aide à la décision")
    with st.container(border=True):
        _render_recommendation(recommendation)

    with st.container(border=True):
        st.plotly_chart(
            build_load_chart(df),
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
        st.caption(
            "Barres : charge du jour (durée × RPE). "
            "Courbe verte : fatigue récente (7 jours). "
            "Courbe marine : condition construite (28 jours)."
        )

    with st.container(border=True):
        if df["acwr"].notna().sum() < 2:
            st.info(
                "La courbe d'ACWR apparaît au 28e jour, quand la charge chronique est calculable. "
                "Les barres de charge ci-dessus restent disponibles immédiatement."
            )
        else:
            st.plotly_chart(
                build_acwr_chart(df),
                use_container_width=True,
                config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            )
            st.caption(
                "Jaune : sous-entraînement (< 0,80). Vert : zone optimale (0,80–1,30). "
                "Orange : attention (1,30–1,50). Rouge : surcharge et risque de blessure (> 1,50)."
            )

    with st.container(border=True):
        _render_table(df)
        export = df.copy()
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
        st.download_button(
            "Télécharger l'historique calculé (CSV)",
            data=export.to_csv(index=False).encode("utf-8-sig"),
            file_name="milo_analytics_acwr.csv",
            mime="text/csv",
            use_container_width=False,
        )

    st.caption(
        "Milo Analytics aide à planifier la séance du lendemain. "
        "Ce n'est pas un dispositif médical : une douleur, une fatigue anormale ou un antécédent "
        "de blessure doivent être discutés avec le staff médical. "
        "L'ACWR est un indicateur de groupe, pas une prédiction individuelle certaine."
    )


if __name__ == "__main__":
    main()
