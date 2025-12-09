import math
import textwrap
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# CONFIG (must be first)
# =========================
st.set_page_config(
    page_title="Petiverse Atlas",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# THEME / CSS
# =========================
PETIVERSE_CSS = """
<style>
/* Background cosmic gradient */
.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(108, 99, 255, 0.12), transparent 35%),
                radial-gradient(circle at 90% 20%, rgba(0, 245, 255, 0.10), transparent 40%),
                radial-gradient(circle at 20% 90%, rgba(255, 0, 153, 0.08), transparent 35%),
                linear-gradient(135deg, #05060a 0%, #0b0f1a 45%, #070a12 100%);
    color: #E6E6F0;
}

/* Global text tweaks */
html, body, [class*="css"]  {
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}

/* Metric cards glow */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    padding: 14px 18px;
    border-radius: 16px;
    box-shadow: 0 0 18px rgba(108,99,255,0.08);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.12);
    background: linear-gradient(135deg, rgba(108,99,255,0.18), rgba(0,245,255,0.10));
    color: white;
}
.stButton>button:hover {
    border-color: rgba(255,255,255,0.22);
    box-shadow: 0 0 14px rgba(0,245,255,0.18);
}

/* Dataframe container */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Section headers */
.petiverse-h1 {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    margin-bottom: 0.2rem;
}
.petiverse-sub {
    opacity: 0.85;
    margin-top: 0;
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    margin-right: 6px;
}
</style>
"""
st.markdown(PETIVERSE_CSS, unsafe_allow_html=True)

# =========================
# CONSTANTS
# =========================
SPECIES = ["Dog", "Cat", "Rabbit", "Bird", "Fish", "Reptile"]
REGIONS = ["North", "South", "East", "West", "Central", "Coastal", "Metro"]
INTAKE_TYPES = ["Stray", "Owner Surrender", "Transfer", "Rescue Intake"]
OUTCOMES = ["Adopted", "Returned to Owner", "Fostered", "Medical Hold", "Other"]
SEXES = ["Female", "Male"]

DOG_BREEDS = [
    "Labrador Retriever", "German Shepherd", "Golden Retriever", "Bulldog",
    "Poodle", "Beagle", "Dachshund", "Siberian Husky", "Mixed (Dog)"
]
CAT_BREEDS = [
    "Domestic Shorthair", "Maine Coon", "Ragdoll", "British Shorthair",
    "Siamese", "Persian", "Sphynx", "Bengal", "Mixed (Cat)"
]
RABBIT_BREEDS = ["Holland Lop", "Netherland Dwarf", "Rex", "Lionhead", "Mixed (Rabbit)"]
BIRD_BREEDS = ["Budgerigar", "Cockatiel", "Lovebird", "Parakeet", "Mixed (Bird)"]
FISH_BREEDS = ["Goldfish", "Betta", "Guppy", "Tetra", "Mixed (Fish)"]
REPTILE_BREEDS = ["Leopard Gecko", "Bearded Dragon", "Corn Snake", "Turtle", "Mixed (Reptile)"]

BREEDS_BY_SPECIES = {
    "Dog": DOG_BREEDS,
    "Cat": CAT_BREEDS,
    "Rabbit": RABBIT_BREEDS,
    "Bird": BIRD_BREEDS,
    "Fish": FISH_BREEDS,
    "Reptile": REPTILE_BREEDS,
}

# =========================
# KNOWLEDGE BASE (short, punchy, expandable)
# =========================
KNOWLEDGE = {
    "Dog": {
        "Care Core": [
            "Daily physical + mental enrichment reduces problem behaviors.",
            "Routine vaccines and parasite prevention are key pillars of longevity.",
            "Most dogs thrive on predictable schedules and clear cues."
        ],
        "Behavior Myths": [
            "‘Stubborn’ often means the reward isn’t clear or motivating.",
            "Anxiety can look like disobedience—observe patterns and triggers."
        ],
    },
    "Cat": {
        "Care Core": [
            "Cats are sensitive to environmental changes—gradual transitions help.",
            "Play mimics hunting; short sessions multiple times a day are ideal.",
            "Litter box issues often reflect stress or medical discomfort."
        ],
        "Behavior Myths": [
            "Cats are social in their own way; choice-based interaction matters."
        ],
    },
    "Rabbit": {
        "Care Core": [
            "Rabbits require high-fiber diets; hay is foundational.",
            "They hide illness well—subtle appetite changes matter."
        ],
    },
    "Bird": {
        "Care Core": [
            "Enrichment prevents feather-destructive behaviors.",
            "Air quality (no smoke, fumes) is especially critical."
        ],
    },
    "Fish": {
        "Care Core": [
            "Stable water parameters matter more than almost anything else.",
            "Overfeeding is a common cause of tank issues."
        ],
    },
    "Reptile": {
        "Care Core": [
            "UVB and temperature gradients are non-negotiable for many species.",
            "Humidity mismanagement is a frequent hidden problem."
        ],
    },
}

# =========================
# DATA GENERATION
# =========================
def _random_dates(n, start_date, end_date, rng):
    delta = (end_date - start_date).days
    offsets = rng.integers(0, max(delta, 1), size=n)
    return [start_date + timedelta(days=int(o)) for o in offsets]

def _breed_for_species(species, rng, size):
    breeds = BREEDS_BY_SPECIES.get(species, ["Mixed"])
    return rng.choice(breeds, size=size, replace=True)

@st.cache_data(show_spinner=False)
def generate_petiverse_data(
    n=60000,
    start_year=2019,
    end_year=2025,
    seed=42
):
    rng = np.random.default_rng(seed)

    # Date range
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    species = rng.choice(SPECIES, size=n, p=[0.38, 0.38, 0.07, 0.06, 0.06, 0.05])
    region = rng.choice(REGIONS, size=n)
    sex = rng.choice(SEXES, size=n)

    # Age in months varies by species
    base_age = rng.gamma(shape=2.2, scale=18, size=n)  # long-tail
    adjust = np.where(species == "Fish", 0.35, 1.0)
    adjust = np.where(species == "Bird", 0.65, adjust)
    adjust = np.where(species == "Rabbit", 0.75, adjust)
    age_months = np.clip((base_age * adjust).round(), 1, 240).astype(int)

    intake_type = rng.choice(INTAKE_TYPES, size=n, p=[0.45, 0.25, 0.18, 0.12])

    health_score = np.clip(rng.normal(72, 14, size=n), 20, 100).round(1)
    behavior_score = np.clip(rng.normal(70, 16, size=n), 10, 100).round(1)

    vaccinated = rng.random(n) < 0.68
    neutered = rng.random(n) < 0.58

    # Weight (rough synthetic model)
    base_weight = rng.normal(8, 6, size=n)
    base_weight = np.where(species == "Dog", rng.normal(18, 10, size=n), base_weight)
    base_weight = np.where(species == "Cat", rng.normal(4.5, 1.5, size=n), base_weight)
    base_weight = np.where(species == "Rabbit", rng.normal(2.3, 0.8, size=n), base_weight)
    base_weight = np.where(species == "Bird", rng.normal(0.2, 0.15, size=n), base_weight)
    base_weight = np.where(species == "Fish", rng.normal(0.08, 0.05, size=n), base_weight)
    base_weight = np.where(species == "Reptile", rng.normal(1.8, 1.2, size=n), base_weight)
    weight_kg = np.clip(base_weight, 0.02, 80).round(2)

    # Fees vary by species and health
    base_fee = rng.normal(120, 45, size=n)
    species_fee_adjust = pd.Series(species).map({
        "Dog": 1.0, "Cat": 0.9, "Rabbit": 0.55, "Bird": 0.5, "Fish": 0.25, "Reptile": 0.6
    }).to_numpy()
    fee = np.clip(base_fee * species_fee_adjust * (0.75 + health_score/200), 5, 350).round(0)

    # Outcome probabilities with simplistic dependencies
    adopt_bias = (
        0.50
        + (health_score - 60) * 0.004
        + (behavior_score - 60) * 0.003
        - (age_months / 240) * 0.18
    )
    adopt_bias = np.clip(adopt_bias, 0.05, 0.85)

    # Convert to categorical outcome via multinomial-like logic
    outcomes = []
    for i in range(n):
        p_adopt = adopt_bias[i]
        p_rto = 0.10 if intake_type[i] == "Stray" else 0.05
        p_foster = 0.12
        p_med = 0.08 if health_score[i] < 55 else 0.04
        p_other = max(0.01, 1 - (p_adopt + p_rto + p_foster + p_med))
        probs = np.array([p_adopt, p_rto, p_foster, p_med, p_other])
        probs = probs / probs.sum()
        outcomes.append(rng.choice(OUTCOMES, p=probs))

    # Days in shelter depends on outcome and scores
    base_days = rng.gamma(2.0, 8.0, size=n)
    base_days += np.where(np.array(outcomes) == "Adopted", 6, 12)
    base_days += np.where(np.array(outcomes) == "Medical Hold", 18, 0)
    base_days += np.where(health_score < 55, 10, 0)
    base_days += np.where(behavior_score < 55, 8, 0)
    days_in_shelter = np.clip(base_days.round(), 1, 180).astype(int)

    dates = _random_dates(n, start_date, end_date, rng)

    # Breed assignment
    breed = []
    for sp in species:
        breed.append(_breed_for_species(sp, rng, 1)[0])

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "year": pd.to_datetime(dates).year,
        "month": pd.to_datetime(dates).month,
        "species": species,
        "breed": breed,
        "region": region,
        "sex": sex,
        "age_months": age_months,
        "weight_kg": weight_kg,
        "intake_type": intake_type,
        "outcome": outcomes,
        "days_in_shelter": days_in_shelter,
        "health_score": health_score,
        "behavior_score": behavior_score,
        "vaccinated": vaccinated,
        "neutered": neutered,
        "adoption_fee_usd": fee
    })

    return df

@st.cache_data(show_spinner=False)
def generate_micro_reference():
    # Tiny "reference-like" synthetic table for nutrition/health visualizations
    rows = []
    for sp in SPECIES:
        for size in ["Tiny", "Small", "Medium", "Large"]:
            kcal = {
                "Dog": {"Tiny": 220, "Small": 420, "Medium": 760, "Large": 1200},
                "Cat": {"Tiny": 160, "Small": 220, "Medium": 280, "Large": 340},
                "Rabbit": {"Tiny": 80, "Small": 110, "Medium": 140, "Large": 170},
                "Bird": {"Tiny": 25, "Small": 45, "Medium": 70, "Large": 110},
                "Fish": {"Tiny": 5, "Small": 8, "Medium": 12, "Large": 18},
                "Reptile": {"Tiny": 30, "Small": 55, "Medium": 85, "Large": 130},
            }[sp][size]
            rows.append({
                "species": sp,
                "size_class": size,
                "estimated_daily_kcal": kcal,
                "notes": "Synthetic educational reference"
            })
    return pd.DataFrame(rows)

# =========================
# FILTER UTIL
# =========================
def apply_filters(df, species_sel, region_sel, year_range, outcome_sel, intake_sel):
    out = df.copy()
    if species_sel:
        out = out[out["species"].isin(species_sel)]
    if region_sel:
        out = out[out["region"].isin(region_sel)]
    if year_range:
        out = out[(out["year"] >= year_range[0]) & (out["year"] <= year_range[1])]
    if outcome_sel:
        out = out[out["outcome"].isin(outcome_sel)]
    if intake_sel:
        out = out[out["intake_type"].isin(intake_sel)]
    return out

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

# =========================
# SIDEBAR - GLOBAL CONTROLS
# =========================
with st.sidebar:
    st.markdown("### 🪐 Petiverse Navigation")
    page = st.radio(
        "Choose a sector",
        [
            "Cosmic Dashboard",
            "Species & Breed Observatory",
            "Health & Nutrition Lab",
            "Adoption & Welfare Analytics",
            "Behavior Signals",
            "My Pet Planner",
            "Data Playground",
            "Petiverse Codex (Knowledge)",
            "About"
        ],
        index=0
    )

    st.divider()
    st.markdown("### 🌌 Universal Filters")

    df_all = generate_petiverse_data()

    species_sel = st.multiselect("Species", SPECIES, default=["Dog", "Cat"])
    region_sel = st.multiselect("Region", REGIONS, default=[])
    year_min = int(df_all["year"].min())
    year_max = int(df_all["year"].max())
    year_range = st.slider("Year range", year_min, year_max, (max(year_min, year_max-4), year_max))

    outcome_sel = st.multiselect("Outcome", OUTCOMES, default=[])
    intake_sel = st.multiselect("Intake type", INTAKE_TYPES, default=[])

    st.divider()
    st.markdown("### 🧪 Simulation Dial")
    n_samples = st.slider("Synthetic records", 20000, 120000, 60000, step=5000)
    seed = st.number_input("Seed", min_value=1, max_value=999999, value=42)

    regenerate = st.button("Regenerate Universe")

# Regenerate data based on controls
if regenerate:
    st.cache_data.clear()

df = generate_petiverse_data(n=n_samples, seed=int(seed))
df_ref = generate_micro_reference()
df_f = apply_filters(df, species_sel, region_sel, year_range, outcome_sel, intake_sel)

# Safety fallback to avoid empty crashes
if df_f.empty:
    df_f = df.head(0)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div>
      <div class="badge">Synthetic Educational Data</div>
      <div class="badge">Multi-Module Lab</div>
      <div class="badge">Petiverse Theme</div>
      <p class="petiverse-h1">🪐 The Petiverse Atlas</p>
      <p class="petiverse-sub">
        An imaginative, data-rich command center where every species is a star system,
        and every insight is a navigational beacon.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# PAGE: Cosmic Dashboard
# =========================
if page == "Cosmic Dashboard":
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Galaxy Pulse (Intake → Outcomes)")
        if df_f.empty:
            st.info("Adjust filters to reveal this sector’s signals.")
        else:
            # Monthly trend
            g = df_f.copy()
            g["ym"] = g["date"].dt.to_period("M").dt.to_timestamp()
            ts = g.groupby(["ym", "species"]).size().reset_index(name="records")
            fig = px.line(ts, x="ym", y="records", color="species", markers=False)
            fig.update_layout(height=360, legend_title_text="Species")
            st.plotly_chart(fig, use_container_width=True)

            # Outcomes distribution
            oc = df_f["outcome"].value_counts().reset_index()
            oc.columns = ["outcome", "count"]
            fig2 = px.bar(oc, x="outcome", y="count", text="count")
            fig2.update_layout(height=320)
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.subheader("Key Signals")
        if df_f.empty:
            st.warning("No data in current filter scope.")
        else:
            total = len(df_f)
            adopted = int((df_f["outcome"] == "Adopted").sum())
            fostered = int((df_f["outcome"] == "Fostered").sum())
            avg_days = float(df_f["days_in_shelter"].mean())
            avg_health = float(df_f["health_score"].mean())
            avg_behavior = float(df_f["behavior_score"].mean())

            st.metric("Records in view", f"{total:,}")
            st.metric("Adoption share", f"{safe_div(adopted, total)*100:.1f}%")
            st.metric("Foster share", f"{safe_div(fostered, total)*100:.1f}%")
            st.metric("Avg days in shelter", f"{avg_days:.1f}")
            st.metric("Avg health score", f"{avg_health:.1f}")
            st.metric("Avg behavior score", f"{avg_behavior:.1f}")

            st.divider()
            st.subheader("Quick Slice Explorer")
            dim = st.selectbox("Slice by", ["region", "intake_type", "sex", "species"])
            metric = st.selectbox("Metric", ["count", "adoption_rate", "avg_days"])
            grp = df_f.groupby(dim)

            if metric == "count":
                dd = grp.size().reset_index(name="value")
            elif metric == "adoption_rate":
                dd = grp.apply(lambda x: safe_div((x["outcome"] == "Adopted").sum(), len(x))).reset_index(name="value")
                dd["value"] = (dd["value"] * 100).round(2)
            else:
                dd = grp["days_in_shelter"].mean().reset_index(name="value")
                dd["value"] = dd["value"].round(2)

            fig3 = px.bar(dd, x=dim, y="value")
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)

# =========================
# PAGE: Species & Breed Observatory
# =========================
elif page == "Species & Breed Observatory":
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Species Gate")
        sp = st.selectbox("Select a species star system", SPECIES, index=0)
        breeds = BREEDS_BY_SPECIES[sp]
        breed_sel = st.multiselect("Breed constellation", breeds, default=breeds[:3])

        local = df[df["species"] == sp].copy()
        if breed_sel:
            local = local[local["breed"].isin(breed_sel)]

        st.caption("This module ignores universal outcome/intake filters to keep breed comparisons stable.")

        if local.empty:
            st.info("No records for the chosen breed constellation.")
        else:
            # Summary table
            summary = local.groupby("breed").agg(
                records=("breed", "size"),
                avg_age_months=("age_months", "mean"),
                avg_weight_kg=("weight_kg", "mean"),
                adoption_rate=("outcome", lambda x: safe_div((x == "Adopted").sum(), len(x))),
                avg_health=("health_score", "mean"),
                avg_behavior=("behavior_score", "mean"),
            ).reset_index()

            summary["adoption_rate"] = (summary["adoption_rate"] * 100).round(1)
            summary["avg_age_months"] = summary["avg_age_months"].round(1)
            summary["avg_weight_kg"] = summary["avg_weight_kg"].round(2)
            summary["avg_health"] = summary["avg_health"].round(1)
            summary["avg_behavior"] = summary["avg_behavior"].round(1)

            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.download_button(
                "Download breed summary CSV",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name=f"{sp.lower()}_breed_summary.csv",
                mime="text/csv"
            )

    with col2:
        st.subheader("Trait Radar")
        if 'local' in locals() and not local.empty:
            # Pick up to 4 breeds for radar
            top_breeds = (
                local["breed"].value_counts()
                .head(4).index.tolist()
            )
            radar_breeds = st.multiselect(
                "Choose up to 4 breeds for radar",
                options=sorted(local["breed"].unique()),
                default=top_breeds
            )[:4]

            if radar_breeds:
                radar_df = []
                for b in radar_breeds:
                    bd = local[local["breed"] == b]
                    radar_df.append({
                        "breed": b,
                        "Health": bd["health_score"].mean(),
                        "Behavior": bd["behavior_score"].mean(),
                        "Adoption Likelihood (proxy)": safe_div((bd["outcome"] == "Adopted").sum(), len(bd)) * 100,
                        "Speed to Home (inverse days)": 100 - min(100, bd["days_in_shelter"].mean() / 1.8),
                    })
                r = pd.DataFrame(radar_df)

                categories = ["Health", "Behavior", "Adoption Likelihood (proxy)", "Speed to Home (inverse days)"]

                fig = go.Figure()
                for _, row in r.iterrows():
                    values = [row[c] for c in categories]
                    values.append(values[0])
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=row["breed"]
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    height=520
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one breed for the radar.")

        st.subheader("Breed Story Capsules")
        st.write(
            "These are *educational narrative hints* generated from synthetic patterns. "
            "They are not medical or breed-standard claims."
        )
        if 'summary' in locals() and not summary.empty:
            for _, row in summary.sort_values("records", ascending=False).head(5).iterrows():
                txt = (
                    f"**{row['breed']}** appears with an adoption proxy of **{row['adoption_rate']}%** "
                    f"and average shelter time around **{row['avg_days'] if 'avg_days' in row else 'N/A'}**. "
                    f"Health/behavior signals suggest focusing on personalized enrichment and routine care."
                )
                st.markdown(f"- {txt}")

# =========================
# PAGE: Health & Nutrition Lab
# =========================
elif page == "Health & Nutrition Lab":
    st.subheader("Bio-Interface: Health & Fuel")

    left, mid, right = st.columns([1.1, 1, 1.2])

    with left:
        st.markdown("#### 1) Species Energy Reference")
        sp_ref = st.selectbox("Species", SPECIES, index=0, key="ref_sp")
        ref_slice = df_ref[df_ref["species"] == sp_ref].copy()
        st.dataframe(ref_slice, use_container_width=True, hide_index=True)

        fig = px.bar(ref_slice, x="size_class", y="estimated_daily_kcal")
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)

    with mid:
        st.markdown("#### 2) Quick Wellness Snapshot")
        if df_f.empty:
            st.info("Adjust universal filters to populate wellness distributions.")
        else:
            sp_pick = st.multiselect("Compare species", SPECIES, default=species_sel or ["Dog", "Cat"], key="wl_sp")
            sub = df_f[df_f["species"].isin(sp_pick)] if sp_pick else df_f

            if sub.empty:
                st.warning("No data for that comparison set.")
            else:
                fig = px.violin(
                    sub, x="species", y="health_score",
                    box=True, points=False
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.violin(
                    sub, x="species", y="behavior_score",
                    box=True, points=False
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown("#### 3) Non-medical Care Companion")
        st.write(
            "This assistant produces **general educational guidance**. "
            "For medical concerns, consult a qualified veterinarian."
        )

        sp_c = st.selectbox("Your pet species", SPECIES, index=0, key="care_sp")
        weight = st.number_input("Weight (kg)", min_value=0.1, max_value=120.0, value=6.0, step=0.1)
        age_m = st.number_input("Age (months)", min_value=1, max_value=300, value=24)
        activity = st.select_slider("Activity level", options=["Low", "Moderate", "High"], value="Moderate")

        # Rough educational estimate
        # Use a simple scaling rule by species
        base_factor = {
            "Dog": 70, "Cat": 60, "Rabbit": 35, "Bird": 120, "Fish": 10, "Reptile": 25
        }[sp_c]

        activity_mult = {"Low": 0.9, "Moderate": 1.0, "High": 1.15}[activity]
        est_kcal = base_factor * (weight ** 0.75) * activity_mult

        st.metric("Estimated daily energy (kcal)", f"{est_kcal:.0f}")

        st.markdown("##### Care Focus Suggestions")
        suggestions = [
            "Prioritize consistent routines and gentle environmental stability.",
            "Use enrichment that matches natural instincts (foraging, hunting, climbing).",
            "Monitor appetite, hydration, and stool/urine patterns for early signals.",
            "Schedule routine preventive care and parasite management where relevant."
        ]
        if age_m < 12:
            suggestions.insert(0, "Young pets benefit from short, frequent training/play cycles.")
        if weight > 25 and sp_c == "Dog":
            suggestions.insert(0, "Large dogs may benefit from joint-friendly, low-impact activity.")
        for s in suggestions:
            st.markdown(f"- {s}")

# =========================
# PAGE: Adoption & Welfare Analytics
# =========================
elif page == "Adoption & Welfare Analytics":
    st.subheader("Shelter Constellations: Adoption & Welfare")

    if df_f.empty:
        st.info("Adjust filters to reveal adoption dynamics.")
    else:
        c1, c2 = st.columns([1.1, 1])

        with c1:
            st.markdown("#### Outcome Matrix")
            pivot_dim = st.selectbox("Rows", ["species", "region", "intake_type", "sex"], index=0)
            pivot_col = st.selectbox("Columns", ["outcome", "region", "intake_type"], index=0)

            pt = pd.pivot_table(
                df_f,
                index=pivot_dim,
                columns=pivot_col,
                values="days_in_shelter",
                aggfunc="size",
                fill_value=0
            )

            st.dataframe(pt, use_container_width=True)

        with c2:
            st.markdown("#### Time-to-Home Signals")
            dim = st.selectbox("Segment by", ["species", "region", "intake_type"], index=0, key="tt_dim")
            metric = st.selectbox("Metric view", ["Avg days in shelter", "Adoption rate"], index=0, key="tt_met")

            grp = df_f.groupby(dim)
            if metric == "Avg days in shelter":
                tt = grp["days_in_shelter"].mean().reset_index(name="value")
                tt["value"] = tt["value"].round(2)
                fig = px.bar(tt, x=dim, y="value")
            else:
                ar = grp.apply(lambda x: safe_div((x["outcome"] == "Adopted").sum(), len(x))).reset_index(name="value")
                ar["value"] = (ar["value"] * 100).round(2)
                fig = px.bar(ar, x=dim, y="value")

            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Micro-Policy Sandbox (Educational)")
        st.write(
            "A playful simulator to visualize how improving wellness signals "
            "could *shift adoption proxies* in this synthetic universe."
        )

        s1, s2, s3 = st.columns(3)
        with s1:
            boost_health = st.slider("Health improvement (+)", 0, 20, 5)
        with s2:
            boost_behavior = st.slider("Behavior improvement (+)", 0, 20, 5)
        with s3:
            focus_species = st.multiselect("Focus species", SPECIES, default=species_sel or ["Dog", "Cat"])

        sim = df_f.copy()
        if focus_species:
            mask = sim["species"].isin(focus_species)
        else:
            mask = pd.Series([True] * len(sim))

        sim.loc[mask, "health_score"] = np.clip(sim.loc[mask, "health_score"] + boost_health, 0, 100)
        sim.loc[mask, "behavior_score"] = np.clip(sim.loc[mask, "behavior_score"] + boost_behavior, 0, 100)

        # Recompute a simple adoption proxy
        proxy = (
            0.50
            + (sim["health_score"] - 60) * 0.004
            + (sim["behavior_score"] - 60) * 0.003
            - (sim["age_months"] / 240) * 0.18
        ).clip(0.05, 0.85)

        sim_view = pd.DataFrame({
            "species": sim["species"],
            "adoption_proxy": proxy
        })
        k = sim_view.groupby("species")["adoption_proxy"].mean().reset_index()
        k["adoption_proxy"] = (k["adoption_proxy"] * 100).round(2)

        fig = px.bar(k, x="species", y="adoption_proxy")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# =========================
# PAGE: Behavior Signals
# =========================
elif page == "Behavior Signals":
    st.subheader("Neural Echoes: Behavior Signals")

    if df_f.empty:
        st.info("Adjust filters to detect behavior patterns.")
    else:
        col1, col2 = st.columns([1.1, 1])

        with col1:
            st.markdown("#### Behavior vs. Context")
            x = st.selectbox("X-axis", ["age_months", "health_score", "days_in_shelter", "adoption_fee_usd"], index=0)
            y = st.selectbox("Y-axis", ["behavior_score", "health_score"], index=0)
            color = st.selectbox("Color", ["species", "region", "intake_type", "outcome"], index=0)

            fig = px.scatter(
                df_f.sample(min(len(df_f), 8000), random_state=1),
                x=x, y=y, color=color,
                hover_data=["breed", "sex", "weight_kg"]
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Correlation Capsule")
            num_cols = ["age_months", "weight_kg", "days_in_shelter", "health_score", "behavior_score", "adoption_fee_usd"]
            corr = df_f[num_cols].corr(numeric_only=True).round(3)

            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto"
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Behavior Risk Flag (Educational)")
        st.write(
            "This rule-based flag is a **non-clinical** synthetic indicator "
            "for demonstration of triage-style dashboards."
        )
        threshold = st.slider("Flag threshold for low behavior score", 20, 70, 50)
        flagged = df_f[df_f["behavior_score"] < threshold].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Flagged records", f"{len(flagged):,}")
        c2.metric("Flag share", f"{safe_div(len(flagged), len(df_f))*100:.1f}%")
        c3.metric("Avg days (flagged)", f"{flagged['days_in_shelter'].mean():.1f}" if not flagged.empty else "0.0")

        if not flagged.empty:
            top = flagged["breed"].value_counts().head(10).reset_index()
            top.columns = ["breed", "count"]
            fig = px.bar(top, x="breed", y="count")
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

# =========================
# PAGE: My Pet Planner
# =========================
elif page == "My Pet Planner":
    st.subheader("Personal Orbit: My Pet Planner")

    st.write(
        "Build a gentle, practical care orbit for your companion. "
        "This is **educational planning**, not veterinary advice."
    )

    c1, c2 = st.columns([1, 1.2])

    with c1:
        sp = st.selectbox("Species", SPECIES, index=0, key="planner_sp")
        breed = st.selectbox("Breed (optional)", BREEDS_BY_SPECIES[sp], index=len(BREEDS_BY_SPECIES[sp]) - 1)
        age = st.number_input("Age (months)", min_value=1, max_value=300, value=18, key="planner_age")
        weight = st.number_input("Weight (kg)", min_value=0.1, max_value=120.0, value=5.0, step=0.1, key="planner_w")
        lifestyle = st.selectbox("Lifestyle", ["Apartment", "House", "Rural/Outdoor", "Mixed"], index=0)
        goals = st.multiselect(
            "Primary goals",
            ["Healthy weight", "Reduce stress", "Improve social behavior", "Skill training", "Elder comfort", "New pet transition"],
            default=["Healthy weight"]
        )

    with c2:
        st.markdown("#### Your 14-Day Orbit Blueprint")

        # Create a simple schedule generator
        play_blocks = 2 if sp in ["Dog", "Cat"] else 1
        if age < 12:
            play_blocks += 1

        enrichment = {
            "Dog": ["Scent games", "Short training drills", "Puzzle feeders", "Leash exploration"],
            "Cat": ["Wand play", "Vertical climbing routes", "Food puzzles", "Hide-and-seek toys"],
            "Rabbit": ["Foraging mats", "Cardboard tunnels", "Gentle handling practice"],
            "Bird": ["Shredding toys", "New perches rotation", "Target training"],
            "Fish": ["Tank layout enrichment", "Feeding variation (safe, modest)"],
            "Reptile": ["Habitat texture changes", "Safe basking/hide optimization"]
        }[sp]

        focus_lines = []
        if "Healthy weight" in goals:
            focus_lines.append("Use measured feeding and track weekly body condition trends.")
        if "Reduce stress" in goals:
            focus_lines.append("Keep routine stable; introduce changes slowly.")
        if "Improve social behavior" in goals:
            focus_lines.append("Reward calm exposure to new people/animals at a distance.")
        if "Skill training" in goals:
            focus_lines.append("Aim for 3–5 minute micro-sessions several times per day.")
        if "Elder comfort" in goals:
            focus_lines.append("Prioritize low-impact enrichment and easy access to essentials.")
        if "New pet transition" in goals:
            focus_lines.append("Set up a quiet safe zone and expand territory gradually.")

        st.markdown("**Focus directives**")
        for line in focus_lines or ["Define one small, measurable goal for the next two weeks."]:
            st.markdown(f"- {line}")

        st.markdown("**Daily rhythm suggestion**")
        st.markdown(f"- {play_blocks} short enrichment/play blocks per day.")
        st.markdown("- 1 calm bonding block (gentle handling, grooming, or quiet presence).")
        st.markdown("- 1 environment check (water, safety, comfort, cleanliness).")

        st.markdown("**Enrichment menu**")
        for item in enrichment:
            st.markdown(f"- {item}")

        st.divider()
        st.markdown("#### Checklist Export")
        checklist = pd.DataFrame({
            "task": [
                "Measure food portions",
                "Refresh clean water",
                "Short enrichment block",
                "Observe appetite & energy",
                "Clean habitat/litter area",
                "Note any unusual signs"
            ],
            "frequency": ["Daily", "Daily", "Daily", "Daily", "Daily", "Daily"]
        })
        st.dataframe(checklist, use_container_width=True, hide_index=True)
        st.download_button(
            "Download checklist CSV",
            checklist.to_csv(index=False).encode("utf-8"),
            file_name="pet_orbit_checklist.csv",
            mime="text/csv"
        )

# =========================
# PAGE: Data Playground
# =========================
elif page == "Data Playground":
    st.subheader("Open Lab: Data Playground")

    st.write(
        "Explore the synthetic Petiverse dataset or upload your own CSV. "
        "All charts are safeguarded against empty selections."
    )

    t1, t2 = st.tabs(["Petiverse Dataset", "Upload Your CSV"])

    with t1:
        if df.empty:
            st.info("No base data available (unexpected). Try regenerating.")
        else:
            st.markdown("#### Smart Filter Console")
            cols = st.multiselect(
                "Columns to view",
                options=df.columns.tolist(),
                default=["date", "species", "breed", "region", "age_months", "health_score", "behavior_score", "outcome", "days_in_shelter"]
            )

            view = df_f[cols] if (cols and not df_f.empty) else df.head(0 if not cols else 200)[cols if cols else df.columns]
            st.dataframe(view, use_container_width=True, hide_index=True)

            st.markdown("#### Chart Forge")
            numeric_cols = [c for c in df.columns if df[c].dtype != "object" and c not in ["year", "month"]]
            cat_cols = ["species", "breed", "region", "sex", "intake_type", "outcome"]

            chart_type = st.selectbox("Chart type", ["Histogram", "Box", "Scatter", "Bar (aggregate)"])
            if chart_type == "Histogram":
                col = st.selectbox("Numeric column", numeric_cols, index=0)
                data = df_f if not df_f.empty else df
                fig = px.histogram(data, x=col, color="species" if "species" in data.columns else None)
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Box":
                y = st.selectbox("Numeric column", numeric_cols, index=0, key="box_y")
                x = st.selectbox("Group by", cat_cols, index=0, key="box_x")
                data = df_f if not df_f.empty else df
                fig = px.box(data, x=x, y=y)
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Scatter":
                x = st.selectbox("X", numeric_cols, index=0, key="sc_x")
                y = st.selectbox("Y", numeric_cols, index=min(1, len(numeric_cols)-1), key="sc_y")
                color = st.selectbox("Color", cat_cols, index=0, key="sc_c")
                data = df_f if not df_f.empty else df
                sample = data.sample(min(len(data), 7000), random_state=2)
                fig = px.scatter(sample, x=x, y=y, color=color)
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

            else:
                dim = st.selectbox("Dimension", cat_cols, index=0, key="bar_dim")
                metric = st.selectbox("Aggregate metric", ["count", "avg_health", "avg_days", "adoption_rate"], index=0, key="bar_met")

                data = df_f if not df_f.empty else df
                g = data.groupby(dim)

                if metric == "count":
                    dd = g.size().reset_index(name="value")
                elif metric == "avg_health":
                    dd = g["health_score"].mean().reset_index(name="value").round(2)
                elif metric == "avg_days":
                    dd = g["days_in_shelter"].mean().reset_index(name="value").round(2)
                else:
                    dd = g.apply(lambda x: safe_div((x["outcome"] == "Adopted").sum(), len(x))).reset_index(name="value")
                    dd["value"] = (dd["value"] * 100).round(2)

                fig = px.bar(dd, x=dim, y="value")
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)

    with t2:
        uploaded = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to unlock custom exploration.")
        else:
            try:
                user_df = pd.read_csv(uploaded)
                st.success(f"Loaded {len(user_df):,} rows and {len(user_df.columns)} columns.")

                st.dataframe(user_df.head(500), use_container_width=True)

                num = [c for c in user_df.columns if pd.api.types.is_numeric_dtype(user_df[c])]
                non = [c for c in user_df.columns if c not in num]

                st.markdown("#### Quick EDA")
                if num:
                    sel = st.selectbox("Numeric column", num)
                    fig = px.histogram(user_df, x=sel)
                    fig.update_layout(height=320)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns detected.")

                if non:
                    dim = st.selectbox("Category column", non)
                    vc = user_df[dim].astype(str).value_counts().head(30).reset_index()
                    vc.columns = [dim, "count"]
                    fig2 = px.bar(vc, x=dim, y="count")
                    fig2.update_layout(height=320)
                    st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error("Could not read this CSV. Please check encoding or formatting.")
                st.caption(str(e))

# =========================
# PAGE: Petiverse Codex (Knowledge)
# =========================
elif page == "Petiverse Codex (Knowledge)":
    st.subheader("The Petiverse Codex")

    st.write(
        "A concise, creative knowledge vault. "
        "Short, memorable, and designed to pair with the data modules."
    )

    sp = st.selectbox("Choose a species chapter", SPECIES, index=0, key="codex_sp")
    chapters = KNOWLEDGE.get(sp, {})

    if not chapters:
        st.info("No codex entries for this species yet.")
    else:
        for chapter, bullets in chapters.items():
            with st.expander(f"📘 {chapter}", expanded=True):
                for b in bullets:
                    st.markdown(f"- {b}")

        st.divider()
        st.markdown("#### Micro-Lessons Generator (Fun)")
        seed_text = st.text_input("Give a theme word (e.g., 'play', 'nutrition', 'fear')", value="play")
        rng = np.random.default_rng(int(seed))
        style = st.selectbox("Tone", ["Scientific-lite", "Friendly mentor", "Myth-buster", "Space-zen"], index=1)

        templates = {
            "Scientific-lite": "In the {theme} domain, {species} benefit from evidence-informed routines and careful observation.",
            "Friendly mentor": "Think of {theme} as a daily gift to your {species}—small, consistent steps beat big, rare efforts.",
            "Myth-buster": "A common {theme} myth about {species} is overgeneralization. Context and individual temperament rule.",
            "Space-zen": "In the quiet orbit of {theme}, your {species} finds stability, safety, and trust."
        }

        line = templates[style].format(theme=seed_text.strip() or "care", species=sp.lower())
        st.markdown(f"**Generated capsule:** {line}")

# =========================
# PAGE: About
# =========================
else:
    st.subheader("About the Petiverse Atlas")

    st.markdown(
        """
        **Petiverse Atlas** is a high-fidelity, creative, multi-module Streamlit app
        built for demonstration, education, and storytelling.

        **What makes it feel 'advanced':**
        - Large-scale synthetic dataset with realistic-ish dependencies.
        - Global filters that remain stable across modules.
        - Multiple analytical lenses: trends, outcome matrices, proxies, correlations.
        - Safe-guarded UI logic to avoid empty-selection errors.
        - Upload-your-own-data sandbox.

        **Important note**
        - The data here is synthetic and educational.
        - Health/nutrition outputs are not medical advice.
        """
    )

    st.markdown("#### Repo suggestions")
    st.markdown(
        "- Add a `README.md` describing the theme and modules.\n"
        "- Add screenshots/GIFs for Streamlit Cloud preview.\n"
        "- Optionally split modules into `/pages` later."
    )

