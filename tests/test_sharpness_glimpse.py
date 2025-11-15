import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpness_glimpse import (
    sharpness_glimpse,
    remove_red_channel,
)
from pinhole_optics import (
    ocellus_row,
    ocellus_envelope,
)

from PIL import Image


def test_sharpness_glimpse_outputs_uint8_and_saves(tmp_path):
    # Create a simple 64x64 RGB gradient image
    arr = np.linspace(0, 255, 64, dtype=np.uint8)
    img = np.stack(np.meshgrid(arr, arr), axis=-1)
    img = np.repeat(img[..., :1], 3, axis=2)
    photo = Image.fromarray(img, mode="RGB")

    output_path = tmp_path / "acuity.jpg"
    out = sharpness_glimpse(
        photo=photo,
        distance=2.0,
        real_width=2.0,
        eye_resolution_x=0.5,
        plot=False,
        output=str(output_path),
    )

    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    assert output_path.exists()


def test_remove_red_channel_zeroes_red(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[..., 0] = 200
    arr[..., 1] = 50
    img = Image.fromarray(arr, mode="RGB")

    out = remove_red_channel(img, output=tmp_path / "no_red.jpg")
    out_arr = np.asarray(out)

    assert np.all(out_arr[..., 0] == 0)
    assert np.array_equal(out_arr[..., 1:], arr[..., 1:])


def test_ocellus_row_reports_geometry_limited():
    result = ocellus_row(
        pupil_diameter=20.0,
        receptor_diameter=1.0,
        receptor_length=5.0,
        cup_h=50.0,
    )
    assert result["limiting_term"] == "geometry-limited"


def test_ocellus_envelope_min_max_ordered():
    span = ocellus_envelope({
        "pupil_diameter": "30..40",
        "receptor_diameter": 5.0,
        "receptor_length": 20.0,
        "cup_h": 80.0,
    })
    assert span["theta_total_fwhm_deg_min"] <= span["theta_total_fwhm_deg_max"]
    assert span["combinations_evaluated"] == 2
