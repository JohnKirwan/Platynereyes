
This repository contains the code used to analyze the electrophysiological recordings of the *Platynereis dumerilii* eyes as mature adults (epitokes) and juveniles (atokes).

![Platynereis eyes](https://github.com/JohnKirwan/Platynereyes/blob/main/platynereis_eye_montage.jpg)

The analysis is summarised is several jupyter notebooks. The notebook Pd_data_summary.ipybn summarises the electrophysiology data and visualizes the V log I plots. 

The code was written by [John Kirwan](https://github.com/JohnKirwan). The library versions we used are specified in environment.yml. To install and activate the environment, run:

```bash
conda env create -f environment.yml
conda activate platynereis
```

For Windows, you may need to add m2w64-toolchain, if you want to run the pymc part, associated with 'Pd_flicker_fusion.ipynb'. Either uncomment the bottommost line in environment.yml prior to creating the environment, or run the following afterwards:

```bash
conda install conda-forge::m2w64-toolchain 
```

The pymc and arviz packages can also be left out if not running 'Pd_flicker_fusion.ipynb'.


The notebook 'sharpness_glimpse.ipynb' contains code to simulate the effect of the measured spatial resolution on images viewed by the animal. This uses the 'sharpness_glimpse.py' module, which is also included in the repository. They are based on the R package [acuityview](https://github.com/eleanorcaves/AcuityView/).

## Installation

The repository now ships a simple Python module (with an accompanying `pyproject.toml`) so you can install everything with pip:

```bash
pip install git+https://github.com/JohnKirwan/Platynereyes
```

or, from a local clone:

```bash
pip install .
```

This exposes the `sharpness_glimpse` module system-wide so you can import it in any notebook or script without modifying `sys.path`. Optional plotting relies on `matplotlib`; install it (or use `pip install .[plot]`) if you want the side-by-side visualization.

> Tip: if you are actively editing the repository, ensure the repo root is at the front of `sys.path` (e.g., `sys.path.insert(0, os.path.abspath(\"..\"))`) so notebooks see the local source rather than an older pip-installed copy.

## Modules

### `sharpness_glimpse.py` — scene acuity pipeline
Python port of `AcuityView::AcuityView`. Key features:

- Image acuity pipeline matches the R package (per-channel rescale → linear RGB → FFT blur → sRGB). Images **must** be square, power-of-two, 3-channel RGB. Photos loaded from disk honor EXIF orientation automatically.
- `sharpness_glimpse()` takes scene geometry (`distance`, `real_width`) and eye resolution (`eye_resolution_x`/`_y`) in degrees, then returns/saves the blurred view. Optional plotting uses matplotlib when present.
- `remove_red_channel()` zeros the R channel of any image path/PIL Image (optional JPEG output) so you can reproduce the notebook's background-removal step without a separate script.

### `pinhole_optics.py` — pinhole acuity model
Contains the ocellus/pinhole optics helpers. Key features:

- `ocellus_row()` / `ocellus_df()` implement the diffraction + tube acceptance + pigment-cup occlusion model. 
- `ocellus_envelope()` accepts a dict of scalars or ranges (lists/tuples or `"min..max"` strings), evaluates all corner cases, and reports the resulting min/max blur angles—ideal for summarising stage/sex scenarios with anatomical uncertainty.
- Provide `cup_a` (opening radius) and `cup_h` (distance from receptor centroid to the opening plane) to activate occlusion limits; omit them to reproduce the classic no-gap model. The notebook demonstrates how to drive `ocellus_envelope` with stage/sex-specific ranges and convert the summaries into tidy tables.
- `describe_condition()` / `conditions_dataframe()` wrap `ocellus_envelope` to turn a dict of named cases into a tidy pandas table.

Usage example:

```python
from sharpness_glimpse import sharpness_glimpse
from pinhole_optics import (
    effective_aperture,
    receptor_subtense_deg,
    ocellus_envelope
)

sharpness_glimpse(
    photo="Pd13.3_BG.jpg",
    distance=3,
    real_width=3,
    eye_resolution_x=7,
    output="Pd13.3_filtered_7deg.jpg"
)

condition = ocellus_envelope({
    "pupil_um": "30..55",
    "receptor_d_um": 3.5,
    "receptor_L_um": [5, 10],
    "cup_h_um": "70..120"
})

D_eff = effective_aperture(45, lens_diameter=100, cup_a=None)
theta_receptor = receptor_subtense_deg(d_um=3.5, depth_um=120)
```

This code is released under the MIT License.

## Tests

Install the package (optionally with plotting extras) and run pytest:

```bash
pip install ".[plot]" pytest
pytest
```

## Notebook overview

- `notebooks/sharpness_glimpse.ipynb` walks through:
  1. Loading anatomical measurements and specifying uncertainty ranges per life stage/sex.
  2. Summarising predicted angular resolution via `conditions_dataframe` (wrapping `ocellus_envelope`; formulas in `docs/pinhole_optics_method.md`).
  3. Applying `remove_red_channel` and `sharpness_glimpse` to generate degraded images matching the computed acuity.
- `notebooks/Pd_pupil_dynamic.ipynb` analyses pupil dynamics and receptor dimensions that feed into the optics model.
