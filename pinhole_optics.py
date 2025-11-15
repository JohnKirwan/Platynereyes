from __future__ import annotations

from itertools import product
from numbers import Real
from typing import Iterable, Mapping, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

__all__ = [
    "effective_aperture",
    "wavelength_in_medium_um",
    "diffraction_fwhm_rad",
    "tube_acceptance_fwhm_rad",
    "receptor_subtense_deg",
    "cup_occlusion_fwhm_rad",
    "ocellus_row",
    "ocellus_df",
    "ocellus_envelope",
    "describe_condition",
    "conditions_dataframe",
]


def _require_positive(name: str, value: float) -> float:
    """Ensure a provided scalar is finite and > 0."""
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite value greater than zero.")
    return float(value)


def wavelength_in_medium_um(lambda0: float, n_medium: float) -> float:
    """Return the wavelength inside a medium with refractive index n."""
    return (lambda0 / 1000.0) / n_medium


def diffraction_fwhm_rad(lambda_med_um: float, aperture_um: float) -> float:
    """Diffraction-limited FWHM (radians) for a circular aperture."""
    if aperture_um <= 0 or not np.isfinite(aperture_um):
        return np.inf
    return 1.03 * lambda_med_um / aperture_um


def tube_acceptance_fwhm_rad(d_um: float | None, L_um: float | None) -> float:
    """
    FWHM imposed by the receptor tube/collar.

    Missing diameter/length => no constraint (np.inf). Non-positive values are invalid.
    """
    if L_um is None or not np.isfinite(L_um):
        return np.inf
    if L_um <= 0:
        raise ValueError("receptor_length must be > 0 when provided.")
    if d_um is None or not np.isfinite(d_um):
        return np.inf
    if d_um <= 0:
        raise ValueError("receptor_diameter must be > 0 when provided.")
    return 2.0 * np.arctan(d_um / (2.0 * L_um))


def receptor_subtense_deg(d_um: float, depth_um: float) -> float:
    """
    Convenience helper that returns the receptor's angular subtense (degrees)
    based on its diameter and axial distance from the pupil opening (cup depth).
    Equivalent to ``np.degrees(tube_acceptance_fwhm_rad(d, depth))``.
    """
    return np.degrees(tube_acceptance_fwhm_rad(d_um, depth_um))


def cup_occlusion_fwhm_rad(a_um: float | None, h_um: float | None) -> float:
    """
    FWHM imposed by pigment cup occlusion.

    a = half-opening radius; h = receptor-centroid to opening plane distance.
    Missing values => no constraint (np.inf).
    """
    if a_um is None or not np.isfinite(a_um) or h_um is None or not np.isfinite(h_um):
        return np.inf
    if a_um <= 0 or h_um <= 0:
        raise ValueError("cup openings and depths must be > 0 when provided.")
    return 2.0 * np.arctan(a_um / h_um)


def _rss(*vals: float) -> float:
    """
    Root-sum-square that propagates invalid contributors instead of quietly
    dropping them.
    """
    cleaned = []
    for v in vals:
        if v is None:
            continue
        if not np.isfinite(v) or v < 0:
            return np.inf
        cleaned.append(float(v))
    if not cleaned:
        return np.inf
    return float(np.sqrt(np.sum(np.square(cleaned))))


def ocellus_row(
    pupil_diameter: float,
    receptor_diameter: float | None,
    receptor_length: float | None,
    lambda0: float = 490.0,
    n_medium: float = 1.33,
    lens_diameter: float | None = None,
    cup_a: float | None = None,    # opening radius
    cup_h: float | None = None,    # receptor-centroid to opening plane distance
    notes: str = ""
) -> dict:
    """
    Compute the combined geometric + diffraction-limited FWHM for a single ocellus.

    Parameters
    ----------
    cup_a : float, optional
        Radius (µm) of the clear pigment-cup opening that limits incoming rays.
        When omitted, the opening is assumed to equal the effective pupil
        (i.e., D_eff / 2).
    cup_h : float, optional
        Depth (µm) from the receptor centroid to the plane of the pigment-cup
        opening. Larger values represent deeper cups, which tighten the
        geometric acceptance angle more strongly.
    """
    pupil_diameter = _require_positive("pupil_diameter", pupil_diameter)
    lambda0 = _require_positive("lambda0", lambda0)
    n_medium = _require_positive("n_medium", n_medium)

    # Effective aperture
    D_eff = effective_aperture(pupil_diameter, lens_diameter, cup_a)
    if cup_h is not None and np.isfinite(cup_h) and cup_h <= 0:
        raise ValueError("cup_h must be > 0 when provided.")

    lam_med_um = wavelength_in_medium_um(lambda0, n_medium)
    theta_diff = diffraction_fwhm_rad(lam_med_um, D_eff)

    theta_tube = tube_acceptance_fwhm_rad(receptor_diameter, receptor_length)
    theta_cup = cup_occlusion_fwhm_rad(
        cup_a if cup_a is not None else (D_eff / 2.0),
        cup_h
    )

    # Geometric acceptance is limited by the *minimum* stop (sharpest angular gate)
    theta_geom = min(theta_tube, theta_cup)
    theta_total = _rss(theta_diff, theta_geom)

    # Dominance diagnostics
    var_total = theta_total**2 if np.isfinite(theta_total) else np.nan
    var_diff = theta_diff**2 if np.isfinite(theta_diff) else 0.0
    var_geom = theta_geom**2 if np.isfinite(theta_geom) else 0.0
    frac_diff = (var_diff / var_total) if (var_total and np.isfinite(var_total) and var_total > 0) else np.nan
    frac_geom = (var_geom / var_total) if (var_total and np.isfinite(var_total) and var_total > 0) else np.nan

    limiting = "mixed"
    if np.isfinite(frac_geom) and frac_geom >= 0.8:
        limiting = "geometry-limited"
    if np.isfinite(frac_diff) and frac_diff >= 0.8:
        limiting = "diffraction-limited"

    to_deg = 180.0 / np.pi
    return {
        "pupil_diameter": pupil_diameter,
        "lens_diameter": lens_diameter,
        "D_eff_um": D_eff,
        "receptor_diameter": receptor_diameter,
        "receptor_length": receptor_length,
        "cup_a": cup_a if cup_a is not None else D_eff / 2.0,
        "cup_h": cup_h,
        "lambda0": lambda0,
        "n_medium": n_medium,
        "lambda_med_um": lam_med_um,
        "theta_diff_fwhm_deg": theta_diff * to_deg,
        "theta_tube_fwhm_deg": theta_tube * to_deg,
        "theta_cup_fwhm_deg": theta_cup * to_deg,
        "theta_geom_fwhm_deg": theta_geom * to_deg,
        "theta_total_fwhm_deg": theta_total * to_deg,
        "frac_variance_diffraction": frac_diff,
        "frac_variance_geometry": frac_geom,
        "limiting_term": limiting,
        "notes": notes,
    }


def ocellus_df(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Apply `ocellus_row` to each row of a pandas DataFrame, case-insensitive columns.
    """
    if pd is None:
        raise ImportError("pandas is required to use ocellus_df; please install pandas first.")
    df_in = df.copy()
    df_in.columns = [c.lower() for c in df_in.columns]
    rows = []
    for _, r in df_in.iterrows():
        rows.append(ocellus_row(
            pupil_diameter=float(r.get("pupil_diameter")),
            receptor_diameter=(None if pd.isna(r.get("receptor_diameter")) else float(r.get("receptor_diameter")))
                if "receptor_diameter" in df_in.columns else None,
            receptor_length=(None if pd.isna(r.get("receptor_length")) else float(r.get("receptor_length")))
                if "receptor_length" in df_in.columns else None,
            lambda0=float(r.get("lambda0")) if "lambda0" in df_in.columns and not pd.isna(r.get("lambda0")) else 490.0,
            n_medium=float(r.get("n_medium")) if "n_medium" in df_in.columns and not pd.isna(r.get("n_medium")) else 1.33,
            lens_diameter=float(r.get("lens_diameter")) if "lens_diameter" in df_in.columns and not pd.isna(r.get("lens_diameter")) else None,
            cup_a=float(r.get("cup_a")) if "cup_a" in df_in.columns and not pd.isna(r.get("cup_a")) else None,
            cup_h=float(r.get("cup_h")) if "cup_h" in df_in.columns and not pd.isna(r.get("cup_h")) else None,
            notes=str(r.get("notes")) if "notes" in df_in.columns else ""
        ))
    return pd.DataFrame(rows)


def _range_from_value(value):
    """
    Normalize a scalar, list/tuple pair, or string like '5..10' into (low, high).
    """
    if value is None:
        return (None, None)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return (None, None)
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1]
            parts = inner.split(",")
            if len(parts) == 2:
                return _range_from_value((parts[0], parts[1]))
        if ".." in text:
            left, right = text.split("..", 1)
            return _range_from_value((left, right))
        return _range_from_value(float(text))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low = float(value[0])
        high = float(value[1])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("Range bounds must be finite numbers.")
        if low > high:
            low, high = high, low
        return (float(low), float(high))
    if isinstance(value, Real):
        val = float(value)
        if not np.isfinite(val):
            raise ValueError("Scalar bounds must be finite numbers.")
        return (val, val)
    return _range_from_value(float(value))


def _range_values(bounds):
    low, high = bounds
    if low is None and high is None:
        return [None]
    if high is None:
        return [low]
    if low is None:
        return [high]
    if np.isclose(low, high):
        return [float(low)]
    return [float(low), float(high)]


def ocellus_envelope(params: dict,
                     metrics: tuple[str, ...] = (
                         "theta_total_fwhm_deg",
                         "theta_geom_fwhm_deg",
                         "theta_diff_fwhm_deg",
                         "theta_tube_fwhm_deg",
                         "theta_cup_fwhm_deg",
                     )) -> dict:
    """
    Evaluate `ocellus_row` across the corners of the provided parameter ranges.

    Parameters
    ----------
    params : dict
        Keys correspond to `ocellus_row` arguments. Each value can be:
        - a scalar (single value)
        - a list/tuple ``[min, max]`` or ``(min, max)``
        - a string "min..max" or "[min, max]"
    metrics : tuple[str, ...]
        Which output fields should be summarized into min/max ranges.

    Returns
    -------
    dict
        Flattened summary containing min/max for inputs and requested metrics,
        plus bookkeeping (combinations evaluated, limiting terms list, notes).
    """
    if not isinstance(params, dict):
        raise TypeError("params must be a dict of ocellus_row arguments.")

    defaults = {
        "pupil_diameter": None,
        "receptor_diameter": None,
        "receptor_length": None,
        "lambda0": 490.0,
        "n_medium": 1.33,
        "lens_diameter": None,
        "cup_a": None,
        "cup_h": None,
    }
    normalized = {}
    for key, default in defaults.items():
        value = params.get(key, default)
        normalized[key] = _range_from_value(value)

    pupil_low, pupil_high = normalized["pupil_diameter"]
    if pupil_low is None and pupil_high is None:
        raise ValueError("pupil_diameter is required for ocellus_envelope.")

    notes = params.get("notes", "")
    keys = list(normalized.keys())
    value_sets = [_range_values(normalized[k]) for k in keys]

    results = []
    for combo in product(*value_sets):
        kwargs = {}
        for key, value in zip(keys, combo):
            if value is None:
                continue
            kwargs[key] = value
        kwargs["notes"] = notes
        results.append(ocellus_row(**kwargs))

    if not results:
        raise RuntimeError("No parameter combinations were generated.")

    summary = {"combinations_evaluated": len(results), "notes": notes}

    for key, (low, high) in normalized.items():
        if low is None and high is None:
            continue
        summary[f"{key}_min"] = low
        summary[f"{key}_max"] = high if high is not None else low

    for metric in metrics:
        values = [res.get(metric) for res in results]
        numeric_values = [float(v) for v in values if isinstance(v, Real)]
        if not numeric_values:
            continue
        summary[f"{metric}_min"] = float(np.min(numeric_values))
        summary[f"{metric}_max"] = float(np.max(numeric_values))

    summary["limiting_terms"] = sorted({res.get("limiting_term", "") for res in results})
    return summary


def describe_condition(label: str, **params) -> dict:
    """
    Convenience wrapper that tags the ocellus_envelope output with a label.
    """
    env = ocellus_envelope(params)
    env["condition"] = label
    return env


def conditions_dataframe(
    cases: Mapping[str, dict],
    metrics: Optional[Iterable[str]] = None
) -> "pd.DataFrame":
    """
    Convert a mapping of {label: parameter_dict} into a tidy pandas DataFrame.

    Parameters
    ----------
    cases : mapping
        Keys are condition labels; values are passed to `describe_condition`.
    metrics : iterable of str, optional
        Which ocellus_envelope outputs to retain. Defaults to a minimal set
        (min/max total FWHM, limiting terms, notes, counted combinations).
    """
    if pd is None:
        raise ImportError("pandas is required for conditions_dataframe; please install pandas first.")
    rows = [describe_condition(name, **cfg) for name, cfg in cases.items()]
    df = pd.DataFrame(rows)

    if metrics is None:
        base_cols = [
            "condition",
            "combinations_evaluated",
            "theta_total_fwhm_deg_min",
            "theta_total_fwhm_deg_max",
            "limiting_terms",
            "notes",
        ]
        present = [col for col in base_cols if col in df.columns]
        df = df[present]
    else:
        cols = ["condition"] + list(metrics)
        present = [col for col in cols if col in df.columns]
        df = df[present]

    return df
def effective_aperture(pupil_diameter: float,
                       lens_diameter: float | None,
                       cup_a: float | None) -> float:
    """
    Return the effective clear aperture in micrometers, matching ocellus_row.
    """
    D_eff = _require_positive("pupil_diameter", pupil_diameter)
    if lens_diameter is not None and np.isfinite(lens_diameter):
        if lens_diameter <= 0:
            raise ValueError("lens_diameter must be > 0 when provided.")
        D_eff = min(D_eff, float(lens_diameter))
    if cup_a is not None and np.isfinite(cup_a):
        if cup_a <= 0:
            raise ValueError("cup_a must be > 0 when provided.")
        D_eff = min(D_eff, 2.0 * float(cup_a))
    return D_eff
