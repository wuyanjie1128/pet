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
