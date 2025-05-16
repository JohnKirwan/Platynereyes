
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
