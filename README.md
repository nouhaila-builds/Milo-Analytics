# Milo Analytics — Smart Workload & Injury Prevention Dashboard

> **AQX Sports Analytics Data Bowl 4.0 Submission**
> *Preventing athletic injuries through science-backed workload analytics and dynamic decision support.*

---

## Project overview

Milo Analytics is a web app that monitors training load with the **Session-RPE (Foster)** model and the **ACWR (Acute-to-Chronic Workload Ratio)**. It helps athletes and coaches anticipate injury risk and adjust the next session.

The idea is simple: athletes often get injured when the fatigue of the last few days outruns the fitness built over several weeks. The app turns a training log into four readings — acute load, chronic load, ACWR, and risk level — then proposes a session for the next day.

```text
Sessions (duration × RPE)
        │
        ▼
Daily load ──► 7-day mean  = acute load ──┐
        │                                   ├──► ACWR ──► zone ──► tomorrow's session
        └──────────► 28-day mean = chronic load ┘
```

![Dashboard in overload: metrics, Danger level, and active recovery](docs/tableau-de-bord.png)

*Running, “Recent overload” scenario, seed 42. The last ACWR is 1.52, so the next session switches to active recovery.*

---

## Why "Milo Analytics"?

Named after **Milo of Croton**, the legendary 6-time Ancient Olympic wrestling champion famous for carrying a growing calf every day until it became a full-grown bull.

Milo’s legend is the earliest historical example of **progressive overload** and workload management:

* **The Calf:** Small, consistent daily load increments build long-term resilience (**Chronic Workload**).
* **The Bull:** Attempting to lift the full-grown bull on Day 1 leads to immediate injury (**Acute Workload Spike / ACWR > 1.5**).

**Milo Analytics** brings this ancient wisdom into modern sports science—ensuring athletes build that capacity gradually and stay in the optimal zone (ACWR 0.8–1.3).

### Where the name comes from

In ancient Greece, Milo of Croton (6th century BC) was the greatest wrestler in Olympic history, crowned champion six times.

The legend says he built his strength in a simple way: every day he carried a newborn calf on his shoulders. Over the months the calf grew heavier, and Milo’s body adapted so gradually that he could eventually carry a full-grown bull without getting injured.

This story is the earliest known picture of progressive overload and load management:

* **The calf** is chronic load. Small, repeated increases build a body that can carry more.
* **The bull on day one** is an acute load spike. Above an ACWR of 1.5, the body gives way.
* **Controlled progression** keeps the ratio between 0.8 and 1.3, inside the optimal zone.

---

## Scientific basis

1. **Session-RPE (Foster):**
   $$\text{Load} = \text{Duration (min)} \times \text{RPE (perceived exertion 1–10)}$$
2. **Acute-to-Chronic Workload Ratio (ACWR):**
   $$\text{ACWR} = \frac{\text{7-day rolling mean (acute fatigue)}}{\text{28-day rolling mean (chronic fitness)}}$$
3. **Reference thresholds:** Based on the work of *Gabbett et al.* and *Hulin et al.* (2014–2016).

| ACWR | Zone | Reading in the app |
| --- | --- | --- |
| < 0.80 | Undertraining | Caution — fitness is dropping, and a sudden spike would be risky |
| 0.80 – 1.30 | Optimal zone | Safe — load and fitness stay aligned |
| 1.30 – 1.50 | Caution zone | Load is accelerating: ease off before the red threshold |
| > 1.50 | Overload | Danger — in these studies, a zone linked to a higher injury risk |

The 1.30–1.50 band is the practical buffer between the optimal zone, often cited up to 1.30, and the danger threshold above 1.50.

![ACWR trajectory across the four risk bands](docs/acwr.png)

*Same scenario. The curve leaves the optimal zone, crosses the caution band, and enters the danger zone.*

The implemented model is the **coupled ACWR**: the 7 acute days sit inside the 28 chronic days, as in the original rolling-mean formula. That coupling pulls the ratio toward 1. Days without a session count as zero load, so the windows are calendar days.

ACWR is a monitoring indicator. It is not a diagnosis, and it is not an individual probability of injury. RPE is subjective. Sleep, stress, pain, and injury history are not in the file.

References:

- Foster C. et al. (2001). *A new approach to monitoring exercise training.* Journal of Strength and Conditioning Research.
- Hulin B.T., Gabbett T.J. et al. (2014). *Spikes in acute workload are associated with increased injury risk in elite cricket fast bowlers.* British Journal of Sports Medicine.
- Gabbett T.J. (2016). *The training—injury prevention paradox.* British Journal of Sports Medicine.
- Blanch P., Gabbett T.J. (2016). *Has the athlete trained enough to return to play safely?* British Journal of Sports Medicine.
- Impellizzeri F.M. et al. (2020). *Acute:Chronic Workload Ratio: Conceptual Issues and Fundamental Pitfalls.* International Journal of Sports Physiology and Performance.

---

## Key features

* **Import and data generation:** custom CSV files, plus a realistic 60-day generator.
  * Sports: running, football, cycling.
  * Demo scenarios: recent overload, optimal zone, return after a lull. A random seed makes each dataset reproducible.
  * CSV: comma or semicolon separator, dates as `YYYY-MM-DD` or `DD/MM/YYYY`, decimals with a point or a comma. Several sessions on the same day are summed. The file’s load is recomputed when it does not equal duration × RPE. French headers such as `duree_min` and `charge_totale` are still accepted.
* **Interactive dashboard:** load trends and risk zones, drawn with Plotly.
  * Four metrics: acute load, chronic load, latest ACWR, risk level.
  * Acute load plotted against chronic load.
  * ACWR curve on four bands: yellow (< 0.80), green (0.80–1.30), orange (1.30–1.50), red (> 1.50).
* **Prescription:** a training recommendation for the next day, based on the detected risk level.
  * The session targets a ratio of 1.25 above 1.50, 1.15 between 1.30 and 1.50, 1.05 inside the optimal zone, and at most +0.10 below 0.80.
  * Tomorrow’s load follows `L = 7 × target ratio × chronic − sum of the last 6 days`.
  * If that load is negative or negligible, the prescription is active recovery (25 min, RPE 3). Otherwise it is turned into a duration and an RPE that are credible for the sport.
  * The figures behind the calculation are shown under the recommendation. The rule is deterministic: it is not a black-box model.

![Acute and chronic load over 60 days](docs/charges.png)

*Bars are the daily load. The green curve is the fatigue of the last 7 days, and the navy curve is the fitness of the last 28 days. The final block lifts the acute load faster than the chronic load.*

![Dashboard in the optimal zone: ACWR 0.94 and a normal training session](docs/zone-optimale.png)

*“Optimal zone” scenario. The ratio stays inside the 0.80–1.30 band, and the proposed session maintains the load.*

One example, the “Recent overload” running scenario (seed 42), is in [`data/exemple_course_surcharge.csv`](data/exemple_course_surcharge.csv).

| Column | Required | Description |
| --- | --- | --- |
| `date` | yes | Session day |
| `duree_min` | yes | Duration in minutes |
| `rpe_effort_1_10` | yes | Perceived exertion from 1 to 10 (0 only when duration is zero) |
| `distance_km` | no | Displayed volume, not used in the ACWR |
| `charge_totale` | no | Recomputed from duration and RPE |

---

## Install and run

Python 3.11 or newer. Dependencies: `streamlit`, `pandas`, `numpy`, `plotly`.

```bash
# 1. Clone the repository
git clone https://github.com/nouhaila-builds/Milo-Analytics.git
cd Milo-Analytics

# 2. Install the dependencies
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

On Windows, if `streamlit` is not recognized:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Streamlit Community Cloud

The GitHub repository must have `app.py` at the root.

- Main file: `app.py`
- Requirements: `requirements.txt`

---

## Repository layout

```text
milo-analytics/
├── app.py
├── requirements.txt
├── README.md
├── logo.png
├── logo.jpg
├── docs/
│   ├── tableau-de-bord.png
│   ├── zone-optimale.png
│   ├── charges.png
│   └── acwr.png
├── data/
│   └── exemple_course_surcharge.csv
└── .streamlit/
    └── config.toml
```

## Limitations

- The synthetic data is for the demo. These are not a real athlete’s sessions.
- The tool helps with planning. It does not replace a doctor, a physiotherapist, or a strength and conditioning coach.
- The thresholds are group reference points. An athlete can get injured at 1.10 or tolerate 1.40.
- This version does not use the exponentially weighted moving average (EWMA), which gives more weight to recent days inside each window.
