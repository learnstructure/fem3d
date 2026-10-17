"""
Reusable 2-Story 3D Reinforced Concrete (RC) Space Frame Model.

Canonical model definition used across multiple linear static, dynamic modal,
equivalent lateral force (ELF), and code-checking examples in fem3d and ecosystem packages.

Structure Description:
    - 2 stories, height h = 3.5 m each (total height H = 7.0 m).
    - 2 bays along X: Bay 1 = 4.0 m, Bay 2 = 6.0 m (total length L_x = 10.0 m).
    - 1 bay along Y:  Bay width = 5.0 m (total width L_y = 5.0 m).
    - Columns: 0.30 m (along X) x 0.40 m (along Y) rectangular section.
        * A  = 0.1200 m^2
        * I2 = 0.0009 m^4 (bending about Y, resisting X-translation)
        * I3 = 0.0016 m^4 (bending about X, resisting Y-translation)
        * J  = 0.001944 m^4 (St. Venant torsion)
    - Beams: 0.25 m (width) x 0.40 m (depth) rectangular section.
        * A  = 0.1000 m^2
        * I3 = 0.001333 m^4 (strong-axis vertical bending)
        * I2 = 0.000521 m^4 (weak-axis horizontal bending)
        * J  = 0.001273 m^4 (St. Venant torsion)
    - Material: Standard concrete grade (C30):
        * E  = 30,000,000 kPa (30 GPa)
        * nu = 0.20
        * G  = 12,500,000 kPa (12.5 GPa)
        * unit_weight = 24.0 kN/m^3 (rho = 2400 kg/m^3)
    - Supports: All 6 base nodes at ground level (Z = 0) are fully fixed [1, 1, 1, 1, 1, 1].
    - Total Nodes: 18 (6 per level at Z = 0.0, 3.5, 7.0 m).
    - Total Elements: 26 (12 columns, 14 beams).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from fem3d import (
    ElasticMaterial,
    FrameElement,
    Node,
    Section,
    Structure,
)

try:
    from struct_core import (
        BeamElement as ScBeamElement,
        ElasticMaterial as ScElasticMaterial,
        Node as ScNode,
        RectangularSection as ScRectangularSection,
        StructuralModel,
        Support as ScSupport,
        parse_units,
    )
    _STRUCT_CORE_AVAILABLE = True
except ImportError:
    _STRUCT_CORE_AVAILABLE = False


# Canonical Geometry Coordinates
X_COORDS = [0.0, 4.0, 10.0]  # 2 bays: 4.0 m and 6.0 m
Y_COORDS = [0.0, 5.0]        # 1 bay: 5.0 m
Z_COORDS = [0.0, 3.5, 7.0]   # 2 stories: 3.5 m each
STORY_ELEVATIONS = [3.5, 7.0]

# Canonical Section Dimensions
COL_WIDTH = 0.30   # along global X
COL_DEPTH = 0.40   # along global Y
BEAM_WIDTH = 0.25  # horizontal width
BEAM_DEPTH = 0.40  # vertical depth

# Material Properties (C30 Concrete)
CONCRETE_E = 30.0e6       # kPa (30 GPa)
CONCRETE_NU = 0.20
CONCRETE_UNIT_WEIGHT = 24.0  # kN/m^3
CONCRETE_RHO = 2400.0        # kg/m^3


def create_rc_frame(
    include_shear_deformation: bool = True,
    units: str = "kN, m",
) -> Structure:
    """
    Construct the canonical 2-story 3D RC Space Frame as a fem3d.Structure.

    Parameters
    ----------
    include_shear_deformation : bool, optional
        Whether to include Timoshenko shear deformation. Defaults to True (matching SAP2000).
    units : str, optional
        Engineering unit system string. Defaults to "kN, m".

    Returns
    -------
    Structure
        Populated fem3d.Structure object with attached convenience attributes:
        - `node_grid`: Dict[Tuple[float, float, float], Node]
        - `columns`: List[FrameElement] (12 column elements)
        - `beams`: List[FrameElement] (14 beam elements)
        - `x_coords`: List[float] [0.0, 4.0, 10.0]
        - `y_coords`: List[float] [0.0, 5.0]
        - `z_coords`: List[float] [0.0, 3.5, 7.0]
        - `story_elevations`: List[float] [3.5, 7.0]
        - `material`: ElasticMaterial
        - `col_section`: Section
        - `beam_section`: Section
    """
    material = ElasticMaterial(E=CONCRETE_E, nu=CONCRETE_NU, unit_weight=CONCRETE_UNIT_WEIGHT)
    col_section = Section.from_rectangle(width=COL_WIDTH, depth=COL_DEPTH)
    beam_section = Section.from_rectangle(width=BEAM_WIDTH, depth=BEAM_DEPTH)

    struct = Structure(units=units)

    # 1. Nodes (18 nodes total)
    node_grid: Dict[Tuple[float, float, float], Node] = {}
    nid = 1
    for z in Z_COORDS:
        for y in Y_COORDS:
            for x in X_COORDS:
                node = Node(nid, x, y, z)
                if z == 0.0:
                    node.set_support([1, 1, 1, 1, 1, 1])  # Fully fixed base
                struct.add_node(node)
                node_grid[(x, y, z)] = node
                nid += 1

    # 2. Columns (12 elements: 6 per story)
    elem_id = 1
    columns: List[FrameElement] = []
    for story in range(2):
        z_bot = Z_COORDS[story]
        z_top = Z_COORDS[story + 1]
        for y in Y_COORDS:
            for x in X_COORDS:
                col_elem = FrameElement(
                    eid=elem_id,
                    node_i=node_grid[(x, y, z_bot)],
                    node_j=node_grid[(x, y, z_top)],
                    material=material,
                    section=col_section,
                    include_shear_deformation=include_shear_deformation,
                )
                struct.add_element(col_elem)
                columns.append(col_elem)
                elem_id += 1

    # 3. Beams (14 elements: 7 per floor level)
    beams: List[FrameElement] = []
    # X-direction beams (2 bays along X * 2 lines along Y * 2 floors = 8 beams)
    for story in (1, 2):
        z = Z_COORDS[story]
        for y in Y_COORDS:
            for bay in (0, 1):
                bm_elem = FrameElement(
                    eid=elem_id,
                    node_i=node_grid[(X_COORDS[bay], y, z)],
                    node_j=node_grid[(X_COORDS[bay + 1], y, z)],
                    material=material,
                    section=beam_section,
                    include_shear_deformation=include_shear_deformation,
                )
                struct.add_element(bm_elem)
                beams.append(bm_elem)
                elem_id += 1

    # Y-direction beams (1 bay along Y * 3 lines along X * 2 floors = 6 beams)
    for story in (1, 2):
        z = Z_COORDS[story]
        for x in X_COORDS:
            bm_elem = FrameElement(
                eid=elem_id,
                node_i=node_grid[(x, Y_COORDS[0], z)],
                node_j=node_grid[(x, Y_COORDS[1], z)],
                material=material,
                section=beam_section,
                include_shear_deformation=include_shear_deformation,
            )
            struct.add_element(bm_elem)
            beams.append(bm_elem)
            elem_id += 1

    # Attach convenience attributes to the Structure instance
    struct.node_grid = node_grid
    struct.columns = columns
    struct.beams = beams
    struct.x_coords = list(X_COORDS)
    struct.y_coords = list(Y_COORDS)
    struct.z_coords = list(Z_COORDS)
    struct.story_elevations = list(STORY_ELEVATIONS)
    struct.material = material
    struct.col_section = col_section
    struct.beam_section = beam_section

    return struct


def create_rc_frame_core(
    include_shear_deformation: bool = True,
    units: str = "kN, m",
) -> StructuralModel:
    """
    Construct the canonical 2-story 3D RC Space Frame directly as a struct_core.StructuralModel.

    Parameters
    ----------
    include_shear_deformation : bool, optional
        Whether to include shear deformation. Defaults to True.
    units : str, optional
        Unit system string. Defaults to "kN, m".

    Returns
    -------
    StructuralModel
        Populated struct_core.StructuralModel object with defined nodes, elements,
        materials, sections, supports, and story_elevations=[3.5, 7.0].
    """
    if not _STRUCT_CORE_AVAILABLE:
        raise ImportError("struct_core package is required for create_rc_frame_core()")

    model = StructuralModel(units=units, story_elevations=list(STORY_ELEVATIONS))

    # Material
    mat = ScElasticMaterial(
        id="C30_Concrete",
        E=CONCRETE_E,
        nu=CONCRETE_NU,
        rho=CONCRETE_RHO,
        unit_weight=CONCRETE_UNIT_WEIGHT,
    )
    model.materials.append(mat)

    # Sections
    sec_col = ScRectangularSection(id="Col_300x400", b=COL_WIDTH, h=COL_DEPTH)
    sec_bm = ScRectangularSection(id="Beam_250x400", b=BEAM_WIDTH, h=BEAM_DEPTH)
    model.sections.extend([sec_col, sec_bm])

    # Nodes (18) and Supports (6 at ground)
    nid = 1
    node_grid: Dict[Tuple[float, float, float], int] = {}
    for z in Z_COORDS:
        for y in Y_COORDS:
            for x in X_COORDS:
                model.nodes.append(ScNode(id=nid, x=x, y=y, z=z))
                if z == 0.0:
                    model.supports.append(ScSupport.fixed_3d(node_id=nid))
                node_grid[(x, y, z)] = nid
                nid += 1

    # Columns (12)
    elem_id = 1
    for story in range(2):
        z_bot = Z_COORDS[story]
        z_top = Z_COORDS[story + 1]
        for y in Y_COORDS:
            for x in X_COORDS:
                model.elements.append(
                    ScBeamElement(
                        id=elem_id,
                        start_node=node_grid[(x, y, z_bot)],
                        end_node=node_grid[(x, y, z_top)],
                        material_id=mat.id,
                        section_id=sec_col.id,
                        include_shear_deformation=include_shear_deformation,
                    )
                )
                elem_id += 1

    # Beams (14)
    # X-direction beams
    for story in (1, 2):
        z = Z_COORDS[story]
        for y in Y_COORDS:
            for bay in (0, 1):
                model.elements.append(
                    ScBeamElement(
                        id=elem_id,
                        start_node=node_grid[(X_COORDS[bay], y, z)],
                        end_node=node_grid[(X_COORDS[bay + 1], y, z)],
                        material_id=mat.id,
                        section_id=sec_bm.id,
                        include_shear_deformation=include_shear_deformation,
                    )
                )
                elem_id += 1

    # Y-direction beams
    for story in (1, 2):
        z = Z_COORDS[story]
        for x in X_COORDS:
            model.elements.append(
                ScBeamElement(
                    id=elem_id,
                    start_node=node_grid[(x, Y_COORDS[0], z)],
                    end_node=node_grid[(x, Y_COORDS[1], z)],
                    material_id=mat.id,
                    section_id=sec_bm.id,
                    include_shear_deformation=include_shear_deformation,
                )
            )
            elem_id += 1

    return model
