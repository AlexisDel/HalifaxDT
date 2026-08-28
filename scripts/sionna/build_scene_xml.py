"""
Assemble the Halifax Sionna scene XML from generated PLY meshes.

This script does not generate geometry. It only references existing terrain,
wall, and roof meshes and assigns Sionna radio materials to them.

Inputs:
  data/scenes/halifax_peninsula/meshes/terrain.ply
  data/scenes/halifax_peninsula/meshes/buildings_wall_<material>.ply
  data/scenes/halifax_peninsula/meshes/buildings_roof_<material>.ply

Output:
  data/scenes/halifax_peninsula/halifax_peninsula.xml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path(__file__).parents[2] / "data"
SCENE_DIR = DATA_ROOT / "scenes" / "halifax_peninsula"
MESH_DIR = SCENE_DIR / "meshes"
OUTPUT_XML = SCENE_DIR / "halifax_peninsula.xml"
TERRAIN_MESH = MESH_DIR / "terrain.ply"

MATERIAL_RGB = {
    "terrain": "0.420000 0.440000 0.390000",
    "asphalt": "0.090000 0.090000 0.085000",
    "brick": "0.401968 0.111874 0.086764",
    "concrete": "0.539479 0.539479 0.539480",
    "glass": "0.420000 0.740000 0.880000",
    "metal": "0.620000 0.650000 0.660000",
    "wood": "0.550000 0.360000 0.210000",
}

SIONNA_ITU_MATERIAL = {
    # Sionna RT exposes the ITU-R P.2040 material table. It does not include
    # asphalt shingles, so we keep the mesh/material label but use the closest
    # available radio proxy for propagation.
    "asphalt": "wood",
    "brick": "brick",
    "concrete": "concrete",
    "glass": "glass",
    "metal": "metal",
    "terrain": "medium_dry_ground",
    "wood": "wood",
}


@dataclass(frozen=True)
class MeshReference:
    kind: str
    material: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.kind}_{self.material}"

    @property
    def shape_id(self) -> str:
        return f"mesh_buildings_{self.kind}_{self.material}"


def itu_material_type(material: str) -> str:
    return SIONNA_ITU_MATERIAL.get(material, "concrete")


def material_id(material: str) -> str:
    return f"mat-itu_{itu_material_type(material)}"


def parse_building_mesh(path: Path) -> MeshReference | None:
    stem = path.stem
    for prefix, kind in (("buildings_wall_", "wall"), ("buildings_roof_", "roof")):
        if stem.startswith(prefix):
            material = stem.removeprefix(prefix)
            if material:
                return MeshReference(kind=kind, material=material, path=path)
    return None


def discover_building_meshes(mesh_dir: Path) -> list[MeshReference]:
    meshes: list[MeshReference] = []
    for path in sorted(mesh_dir.glob("buildings_*.ply")):
        mesh = parse_building_mesh(path)
        if mesh is not None:
            meshes.append(mesh)
    return meshes


def write_scene_xml(path: Path, building_meshes: list[MeshReference]) -> None:
    used_materials = sorted({mesh.material for mesh in building_meshes} | {"terrain"})
    used_itu_types = sorted({itu_material_type(material) for material in used_materials})
    lines = [
        '<scene version="2.1.0">',
        "",
        "<!-- Camera and rendering parameters -->",
        "",
        '    <integrator type="path" id="integrator" name="integrator"/>',
        "",
        "<!-- Materials -->",
        "",
    ]

    for itu_type in used_itu_types:
        mat_id = f"mat-itu_{itu_type}"
        display_material = next(
            (material for material in used_materials if itu_material_type(material) == itu_type),
            "concrete",
        )
        rgb = MATERIAL_RGB.get(display_material, MATERIAL_RGB["concrete"])
        lines.extend(
            [
                f'    <bsdf type="diffuse" id="{mat_id}" name="{mat_id}">',
                f'        <rgb value="{rgb}" name="reflectance"/>',
                "    </bsdf>",
                "",
            ]
        )

    lines.extend(
        [
            "<!-- Emitters -->",
            "",
            '    <emitter type="constant" id="light" name="light"/>',
            "",
            "<!-- Shapes -->",
            "",
            '    <shape type="ply" id="mesh_terrain" name="mesh_terrain">',
            '        <string name="filename" value="meshes/terrain.ply"/>',
            '        <boolean name="face_normals" value="true"/>',
            f'        <ref id="{material_id("terrain")}" name="bsdf"/>',
            "    </shape>",
            "",
        ]
    )

    for mesh in sorted(building_meshes, key=lambda item: item.key):
        filename = mesh.path.relative_to(SCENE_DIR).as_posix()
        lines.extend(
            [
                f'    <shape type="ply" id="{mesh.shape_id}" name="{mesh.shape_id}">',
                f'        <string name="filename" value="{filename}"/>',
                '        <boolean name="face_normals" value="true"/>',
                f'        <ref id="{material_id(mesh.material)}" name="bsdf"/>',
                "    </shape>",
                "",
            ]
        )

    lines.append("</scene>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not TERRAIN_MESH.exists():
        raise FileNotFoundError(f"Missing terrain mesh: {TERRAIN_MESH}")

    building_meshes = discover_building_meshes(MESH_DIR)
    if not building_meshes:
        raise FileNotFoundError(f"No building meshes found in: {MESH_DIR}")

    write_scene_xml(OUTPUT_XML, building_meshes)

    print(f"Building mesh parts: {len(building_meshes):,}")
    for mesh in building_meshes:
        print(f"  {mesh.key}: {mesh.path}")
    print(f"Wrote scene XML: {OUTPUT_XML}")


if __name__ == "__main__":
    main()
