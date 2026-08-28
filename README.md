# HalifaxDT

HalifaxDT is a wireless digital twin of the Halifax Peninsula, Nova Scotia,
Canada, built from open geospatial and spectrum data and exported as a
Sionna RT-ready scene.

This public bundle contains:

- a ready-to-use Sionna RT scene for the Halifax Peninsula;
- generated terrain and building meshes;
- selected building-height and antenna datasets used by the tutorials;
- example notebooks for loading the scene and running simple radio simulations;
- the reconstruction scripts used to prepare the geospatial inputs, infer
  building heights, assign coarse material metadata, and generate Sionna assets.

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- data/
|   |-- README.md
|   |-- raw_data/             # local source data, ignored by Git
|   |-- interim_data/         # local prepared inputs, ignored by Git
|   |-- processed_data/
|   |   |-- building_heights_selected.gpkg
|   |   `-- peninsula_cellular_antennas_2g_to_5g.csv
|   |-- scenes/
|   |   `-- halifax_peninsula/
|   |       |-- halifax_peninsula_mono_concrete.xml
|   |       |-- halifax_peninsula_multi_material.xml
|   |       |-- terrain_metadata.txt
|   |       `-- meshes/
|   |           |-- terrain.ply
|   |           `-- buildings_*.ply
|   `-- visuals/             # regenerated visual outputs, ignored by Git
|-- examples/
|   `-- outputs/             # pre-generated example maps and figures
|-- tutorials/
|   |-- halifax_twin_demo.ipynb
|   `-- halifax_twin_custom_simulation.ipynb
|-- scripts/
|   |-- preprocessing/
|   |-- heights/
|   |-- materials/
|   |-- sionna/
|   `-- visuals/
```

The `data/processed_data/` and `data/scenes/` folders form the publication
bundle: they are sufficient for loading HalifaxDT in Sionna RT and running the
tutorial notebooks. The `scripts/` folder and `data/README.md` document the
construction pipeline used to regenerate the scene from source geospatial data.

## Scene Variants

Two Sionna-ready XML scene files are included:

- `data/scenes/halifax_peninsula/halifax_peninsula_mono_concrete.xml`: the geometry used for the
  coverage-map evaluation, with all buildings assigned to a single concrete
  material and terrain assigned to medium dry ground.
- `data/scenes/halifax_peninsula/halifax_peninsula_multi_material.xml`: the same geometry with
  material-separated building meshes using coarse semantic material metadata.

Both XML files reference the shared PLY meshes in
`data/scenes/halifax_peninsula/meshes/` using relative paths.

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name halifaxdt --display-name "HalifaxDT"
```

Launch JupyterLab:

```bash
python -m jupyter lab
```

Open one of the notebooks in `tutorials/` and select the `HalifaxDT` kernel.
The notebooks discover the repository root by walking upward until they find the
top-level `data/` directory, so keep the folder layout unchanged.

## Rebuilding the Scene

The ready-to-use publication scene is already included. To rebuild HalifaxDT
from source geospatial data, follow the setup guide in:

```text
data/README.md
```

Then run the pipeline scripts from the repository root:

```bash
python scripts/preprocessing/download_osm_buildings.py
python scripts/preprocessing/prepare_footprints_peninsula.py
python scripts/preprocessing/prepare_dsm_peninsula.py
python scripts/preprocessing/prepare_dem_peninsula.py
python scripts/preprocessing/prepare_nstdb_peninsula.py
python scripts/preprocessing/prepare_peninsula_antennas.py
python scripts/heights/select_building_heights.py
python scripts/materials/classify_building_materials.py
python scripts/sionna/build_terrain_mesh.py
python scripts/sionna/build_building_mesh.py
python scripts/sionna/build_scene_xml.py
```

Large raw source files and regenerated intermediate products should remain under
`data/`; they are intentionally ignored by Git.

## License

This repository is released under the MIT License. The original public
geospatial and spectrum datasets used to construct HalifaxDT remain subject to
their respective source licenses and terms.
