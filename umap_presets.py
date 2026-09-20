"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe km_presets.py in kmeans-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import umap_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_tours_slider": SettingSpec("n", int, C.DEFAULT_N_TOURS, C.N_TOURS_MIN, C.N_TOURS_MAX),
    "q_slider": SettingSpec("q", int, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "curvature_slider": SettingSpec("curv", float, C.DEFAULT_CURVATURE, C.CURVATURE_MIN, C.CURVATURE_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "outlier_slider": SettingSpec("out", int, C.DEFAULT_OUTLIER_PCT, C.OUTLIER_PCT_MIN, C.OUTLIER_PCT_MAX),
    "n_neighbors_slider": SettingSpec("nn", int, C.DEFAULT_N_NEIGHBORS, C.N_NEIGHBORS_MIN, C.N_NEIGHBORS_MAX),
    "min_dist_select": SettingSpec("md", float, C.DEFAULT_MIN_DIST, min(C.MIN_DIST_CHOICES), max(C.MIN_DIST_CHOICES)),
    "n_epochs_slider": SettingSpec("ep", int, C.DEFAULT_N_EPOCHS, C.N_EPOCHS_MIN, C.N_EPOCHS_MAX),
    "neg_rate_slider": SettingSpec("neg", int, C.DEFAULT_NEG_RATE, C.NEG_RATE_MIN, C.NEG_RATE_MAX),
    "init_select": SettingSpec("init", _choice(C.INITS), C.DEFAULT_INIT),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"n_tours": "n_tours_slider", "q": "q_slider", "curvature": "curvature_slider", "noise": "noise_slider", "outlier_pct": "outlier_slider",
               "n_neighbors": "n_neighbors_slider", "min_dist": "min_dist_select", "n_epochs": "n_epochs_slider", "negative_sample_rate": "neg_rate_slider", "init": "init_select",
               "seed": "seed_input"}


def snap_min_dist(value):
    return min(C.MIN_DIST_CHOICES, key=lambda choice: abs(choice - value))


def n_neighbors_max(n_tours):
    """Obere Grenze von n_neighbors: höchstens n − 1 (mehr Nachbarn als andere Touren gibt es nicht)."""
    return int(min(C.N_NEIGHBORS_MAX, n_tours - 1))


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["min_dist_select"] = snap_min_dist(st.session_state.get("min_dist_select", C.DEFAULT_MIN_DIST))
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
