"""
Adapter module bridging fem3d and struct_core.

Provides three conversion functions:
- model_from_core : convert a struct_core.StructuralModel into a fem3d.Structure
- model_to_core   : convert a fem3d.Structure into a struct_core.StructuralModel
- result_to_core  : convert fem3d analysis outputs into a struct_core.AnalysisResult
"""

from __future__ import annotations
from typing import Optional, Union, TYPE_CHECKING
import numpy as np

try:
    from struct_core import (
        StructuralModel,
        Project,
        Node as ScNode,
        Support as ScSupport,
        ElasticMaterial as ScElasticMaterial,
        RectangularSection as ScRectangularSection,
        CircularSection as ScCircularSection,
        GeneralSection as ScGeneralSection,
        BeamElement as ScBeamElement,
        TrussElement as ScTrussElement,
        SpringElement as ScSpringElement,
        PointLoad as ScPointLoad,
        DistributedLoad as ScDistributedLoad,
        LoadCase as ScLoadCase,
        Units,
        parse_units,
        AnalysisResult,
        NodeResult,
        NodeDisplacement,
        NodeReaction,
        ElementResult,
        BeamSectionForce,
    )
    _STRUCT_CORE_AVAILABLE = True
except ImportError:
    _STRUCT_CORE_AVAILABLE = False

try:
    from struct_load.engine.self_weight import calculate_self_weight as _calc_self_weight
    from struct_load.engine.mass_source import calculate_mass_source as _calc_mass_source
    _STRUCT_LOAD_AVAILABLE = True
except ImportError:
    _calc_self_weight = None
    _calc_mass_source = None
    _STRUCT_LOAD_AVAILABLE = False

from ..nodes import Node
from ..materials.elastic import ElasticMaterial
from ..sections.section import Section
from ..elements.frame import FrameElement
from ..elements.truss import TrussElement
from ..elements.spring import SpringElement
from ..loads import PointLoad, DistributedLoad
from ..structure import Structure
from ..results import Results


def _check_struct_core():
    if not _STRUCT_CORE_AVAILABLE:
        raise ImportError(
            "The 'struct_core' package is required for adapter features.\n"
            "Install it with:  pip install struct_core\n"
            "or install extra: pip install fem3d[ecosystem]"
        )


def model_from_core(
    source: Union[StructuralModel, Project],
    load_case_id: Optional[Union[int, str, List[Union[int, str]], Tuple[Union[int, str], ...]]] = None,
    load_combination_id: Optional[str] = None,
    mass_source: Optional[Union[Any, str]] = None,
    include_self_weight: bool = True,
    units: Optional[Union[str, Units, dict]] = None,
    gravity_accel: Optional[float] = None,
    include_shear_deformation: Optional[bool] = None,
) -> Structure:
    """
    Convert a struct_core.StructuralModel or struct_core.Project into a fem3d.Structure.

    Parameters
    ----------
    source : StructuralModel or Project
        The structural model or project to convert.
    load_case_id : int, str, or list of (int or str), optional
        Target load case ID or list of IDs. If None (default), all load cases
        are transferred and summed. Pass 'ALL' for all cases, or specific IDs like 'DEAD'.
    load_combination_id : str, optional
        Target load combination ID. If specified, applies loads with combination factors.
    mass_source : MassSource, dict, or str, optional
        Inertial mass source configuration. If provided, evaluates masses from dead/live load
        patterns and element self-masses and assigns them to nodes and frame elements.
    include_self_weight : bool, optional
        Whether to automatically compute and include element self-weight for any
        load case where `include_self_weight=True`. Defaults to True.
        Uses `struct_load.engine.self_weight.calculate_self_weight`.
    units : str, Units, or dict, optional
        Units specification (e.g. 'kN, m' or 'kip, in'). If None, inferred automatically
        from source.units or source.metadata.units. Defaults to 'kN, m'.
    gravity_accel : float, optional
        Gravity acceleration. Defaults to `active_units.gravity_acceleration`.
    include_shear_deformation : bool, optional
        Whether to consider shear deformation (Timoshenko beam theory) in frame elements.
        If None (default), reads `include_shear_deformation` from each `struct_core.BeamElement`.
        If True or False, overrides all frame elements in the model.
    """
    _check_struct_core()

    if isinstance(source, Project):
        if source.model is None:
            raise ValueError("Project contains no structural model.")
        model = source.model
    else:
        model = source

    active_units = parse_units(
        units
        or getattr(source, "units", None)
        or getattr(getattr(source, "metadata", None), "units", None)
        or "kN, m"
    )

    structure = Structure(units=active_units)

    if gravity_accel is None:
        gravity_accel = active_units.gravity_acceleration

    mass_factor = active_units.consistent_mass_factor

    def _convert_rho(mat_schema: Any) -> float:
        # Dynamic mass density in consistent units: [F] * [T]^2 / [L]^4
        uw = getattr(mat_schema, "unit_weight", None)
        if uw is not None and float(uw) > 0.0:
            return float(uw) / gravity_accel
        raw_rho = getattr(mat_schema, "rho", 0.0)
        val = float(raw_rho or 0.0)
        if val <= 0.0:
            return 0.0
        return val * mass_factor

    def _convert_mass(raw_m: float) -> float:
        val = float(raw_m or 0.0)
        if val <= 0.0:
            return 0.0
        return val * mass_factor

    # 1. Nodes
    for n in model.nodes:
        node = Node(nid=n.id, x=n.x, y=n.y, z=getattr(n, "z", 0.0))
        m = _convert_mass(getattr(n, "mass", 0.0))
        it = _convert_mass(getattr(n, "inertia", 0.0))
        if m > 0 or it > 0:
            node.set_mass(mass=m, inertia=it)
        structure.add_node(node)

    # 2. Supports
    for s in model.supports:
        if s.node_id in structure.nodes:
            nd = structure.nodes[s.node_id]
            ux = bool(getattr(s, "ux", False))
            uy = bool(getattr(s, "uy", False))
            uz = bool(getattr(s, "uz", False))
            rx = bool(getattr(s, "rx", False))
            ry = bool(getattr(s, "ry", False))
            rz = bool(getattr(s, "rz", False))
            nd.set_support(
                ux_fixed=bool(getattr(s, "ux", False)),
                uy_fixed=bool(getattr(s, "uy", False)),
                uz_fixed=bool(getattr(s, "uz", False)),
                rx_fixed=bool(getattr(s, "rx", False)),
                ry_fixed=bool(getattr(s, "ry", False)),
                rz_fixed=bool(getattr(s, "rz", False)),
            )

    # 3. Materials dictionary
    mat_map = {}
    for m in model.materials:
        E = getattr(m, "E", 29000.0)
        nu = getattr(m, "nu", 0.3)
        rho = _convert_rho(m)
        uw = getattr(m, "unit_weight", None)
        mat_map[m.id] = ElasticMaterial(E=E, nu=nu, rho=rho, unit_weight=uw)

    # 4. Sections dictionary
    sec_map = {}
    for sec in model.sections:
        sec_type = getattr(sec, "type", "general")
        if sec_type == "rectangular":
            sec_map[sec.id] = Section.from_rectangle(width=sec.b, depth=sec.h)
        elif sec_type == "circular":
            sec_map[sec.id] = Section.from_circle(diameter=sec.d)
        else:
            A = getattr(sec, "A", 1.0)
            I3 = getattr(sec, "I3", 1.0)
            I2 = getattr(sec, "I2", I3)
            J = getattr(sec, "J", I3 + I2)
            as2 = getattr(sec, "As2", getattr(sec, "as2", getattr(sec, "Ay", getattr(sec, "AS2", None))))
            as3 = getattr(sec, "As3", getattr(sec, "as3", getattr(sec, "Az", getattr(sec, "AS3", None))))
            sec_map[sec.id] = Section(A=A, I2=I2, I3=I3, J=J, As2=as2, As3=as3)

    # 5. Elements
    for elem in model.elements:
        if elem.start_node not in structure.nodes or elem.end_node not in structure.nodes:
            continue
        n_i = structure.nodes[elem.start_node]
        n_j = structure.nodes[elem.end_node]
        elem_type = getattr(elem, "type", "beam")

        dx = float(n_j.x - n_i.x)
        dy = float(n_j.y - n_i.y)
        dz = float(n_j.z - n_i.z)
        if (dx**2 + dy**2 + dz**2) < 1e-12 and elem_type != "spring":
            continue

        if elem_type == "truss":
            mat = mat_map.get(elem.material_id, ElasticMaterial(E=29000.0))
            sec = sec_map.get(elem.section_id, Section(A=1.0))
            fem_el = TrussElement(eid=elem.id, node_i=n_i, node_j=n_j, material=mat, section=sec)
        elif elem_type == "spring":
            kx = getattr(elem, "kx", 0.0)
            ky = getattr(elem, "ky", 0.0)
            kz = getattr(elem, "kz", 0.0)
            fem_el = SpringElement(eid=elem.id, node_i=n_i, node_j=n_j, kx=kx, ky=ky, kz=kz)
        else:  # Beam / Frame
            mat = mat_map.get(elem.material_id, ElasticMaterial(E=29000.0))
            sec = sec_map.get(elem.section_id, Section(A=10.0, I2=100.0, I3=200.0, J=50.0))
            elem_shear = (
                include_shear_deformation
                if include_shear_deformation is not None
                else getattr(elem, "include_shear_deformation", False)
            )
            fem_el = FrameElement(
                eid=elem.id,
                node_i=n_i,
                node_j=n_j,
                material=mat,
                section=sec,
                include_shear_deformation=elem_shear,
            )

        structure.add_element(fem_el)

    # 6. Loads
    # Unit system and gravity acceleration already resolved above

    # 5b. Evaluate Mass Source if specified
    ms_target = mass_source if mass_source is not None else getattr(model, "mass_source", None)
    if ms_target is not None:
        if _calc_mass_source is not None:
            ms_res = _calc_mass_source(
                model=model,
                mass_source=ms_target,
                gravity_accel=gravity_accel,
                units=active_units,
            )
            for nid, (mx, my, mz) in ms_res.nodal_masses.items():
                if nid in structure.nodes:
                    structure.nodes[nid].set_mass(mass=[mx, my, mz])

            for eid, m_extra in ms_res.element_masses.items():
                if eid in structure.elements:
                    structure.elements[eid].mass += float(m_extra)

            structure.mass_source_result = ms_res
        else:
            import warnings
            warnings.warn(
                "A mass_source was specified, but 'struct_load' is not installed to evaluate it. "
                "Install struct_load to enable automated mass source calculation.",
                UserWarning,
                stacklevel=2,
            )

    sw_cache = {}

    def _apply_case_self_weight_3d(case_id: str, factor: float = 1.0):
        if not include_self_weight:
            return
        if _calc_self_weight is None:
            import warnings
            warnings.warn(
                f"Load case '{case_id}' has include_self_weight=True, but 'struct_load' is not installed. "
                "Self-weight was not added to the analysis model. Install struct_load to enable automatic self-weight.",
                UserWarning,
                stacklevel=2,
            )
            return

        if case_id not in sw_cache:
            sw_cache[case_id] = _calc_self_weight(
                model=model,
                load_case_id=case_id,
                gravity_accel=gravity_accel,
                units=active_units,
                direction="-z",
            )
        sw_case = sw_cache[case_id]

        for dl in getattr(sw_case, "element_loads", []):
            eid = dl.element_id
            if eid in structure.elements:
                el = structure.elements[eid]
                wx = float(getattr(dl, "wx", 0.0) or 0.0) * factor
                wy = float(getattr(dl, "wy", 0.0) or 0.0) * factor
                wz = float(getattr(dl, "wz", 0.0) or 0.0) * factor
                structure.add_load(
                    DistributedLoad(el, wx=wx, wy=wy, wz=wz, coord_system="global", is_self_weight=True)
                )

    target_comb = None
    if load_combination_id is not None:
        target_comb = next(
            (c for c in getattr(model, "load_combinations", []) if c.id == load_combination_id), None
        )
    elif load_case_id is not None and getattr(model, "load_combinations", None):
        target_comb = next(
            (c for c in model.load_combinations if c.id == load_case_id), None
        )

    if target_comb is not None:
        load_cases_map = {lc.id: lc for lc in getattr(model, "load_cases", [])}
        for factor_obj in target_comb.factors:
            lc = load_cases_map.get(factor_obj.load_case_id)
            if not lc:
                raise KeyError(
                    f"Load combination '{combination_id}' references unknown load case '{factor_obj.load_case_id}'."
                )
            factor = factor_obj.factor
            for pl in getattr(lc, "point_loads", []):
                nid = pl.node_id
                if nid in structure.nodes:
                    fx = getattr(pl, "fx", 0.0) * factor
                    fy = getattr(pl, "fy", 0.0) * factor
                    fz = getattr(pl, "fz", 0.0) * factor
                    mx = getattr(pl, "mx", 0.0) * factor
                    my = getattr(pl, "my", 0.0) * factor
                    mz = getattr(pl, "mz", 0.0) * factor
                    structure.add_load(PointLoad(structure.nodes[nid], fx=fx, fy=fy, fz=fz, mx=mx, my=my, mz=mz))
            for dl in getattr(lc, "element_loads", []):
                eid = dl.element_id
                if eid in structure.elements:
                    el = structure.elements[eid]
                    if getattr(dl, "is_local", False):
                        w1 = float(getattr(dl, "w1", 0.0) or 0.0) * factor
                        w2 = float(getattr(dl, "w2", 0.0) or 0.0) * factor
                        w3 = float(getattr(dl, "w3", 0.0) or 0.0) * factor
                        structure.add_load(DistributedLoad(el, w1=w1, w2=w2, w3=w3, coord_system="local"))
                    else:
                        wx = float(getattr(dl, "wx", 0.0) or 0.0) * factor
                        wy = float(getattr(dl, "wy", 0.0) or 0.0) * factor
                        wz = float(getattr(dl, "wz", 0.0) or 0.0) * factor
                        structure.add_load(DistributedLoad(el, wx=wx, wy=wy, wz=wz, coord_system="global"))
            if getattr(lc, "include_self_weight", False):
                _apply_case_self_weight_3d(lc.id, factor=factor)
    else:
        load_cases = getattr(model, "load_cases", [])
        if load_cases:
            if load_case_id is None:
                cases_to_apply = load_cases
            elif isinstance(load_case_id, (list, tuple, set)):
                targets = {str(cid).upper() for cid in load_case_id}
                cases_to_apply = [
                    lc for lc in load_cases
                    if str(lc.id).upper() in targets or str(getattr(lc, "name", "")).upper() in targets
                ]
                if not cases_to_apply:
                    raise ValueError(
                        f"None of the requested load cases {load_case_id} found in model. Available: {[lc.id for lc in load_cases]}"
                    )
            else:
                cid_str = str(load_case_id).upper()
                if cid_str in ("ALL", "TOTAL", "ENVELOPE"):
                    cases_to_apply = load_cases
                else:
                    cases_to_apply = [
                        lc for lc in load_cases
                        if str(lc.id).upper() == cid_str or str(getattr(lc, "name", "")).upper() == cid_str
                    ]
                    if not cases_to_apply:
                        raise ValueError(
                            f"Requested load case '{load_case_id}' not found in model load cases: {[lc.id for lc in load_cases]}"
                        )

            for lc in cases_to_apply:
                for pl in getattr(lc, "point_loads", []):
                    nid = pl.node_id
                    if nid in structure.nodes:
                        fx = getattr(pl, "fx", 0.0)
                        fy = getattr(pl, "fy", 0.0)
                        fz = getattr(pl, "fz", 0.0)
                        mx = getattr(pl, "mx", 0.0)
                        my = getattr(pl, "my", 0.0)
                        mz = getattr(pl, "mz", 0.0)
                        structure.add_load(PointLoad(structure.nodes[nid], fx=fx, fy=fy, fz=fz, mx=mx, my=my, mz=mz))

                for dl in getattr(lc, "element_loads", []):
                    eid = dl.element_id
                    if eid in structure.elements:
                        el = structure.elements[eid]
                        if getattr(dl, "is_local", False):
                            w1 = float(getattr(dl, "w1", 0.0) or 0.0)
                            w2 = float(getattr(dl, "w2", 0.0) or 0.0)
                            w3 = float(getattr(dl, "w3", 0.0) or 0.0)
                            structure.add_load(DistributedLoad(el, w1=w1, w2=w2, w3=w3, coord_system="local"))
                        else:
                            wx = float(getattr(dl, "wx", 0.0) or 0.0)
                            wy = float(getattr(dl, "wy", 0.0) or 0.0)
                            wz = float(getattr(dl, "wz", 0.0) or 0.0)
                            structure.add_load(DistributedLoad(el, wx=wx, wy=wy, wz=wz, coord_system="global"))

                if getattr(lc, "include_self_weight", False):
                    _apply_case_self_weight_3d(lc.id, factor=1.0)

    return structure


def model_to_core(
    source: Union[Structure, object],
    units: Optional[Union[str, Units, dict]] = None,
    default_load_case_id: str = "LIVE",
) -> StructuralModel:
    """
    Convert a fem3d Structure or SimpleFrame into a struct_core.StructuralModel.

    Parameters
    ----------
    source : Structure or SimpleFrame
        Source fem3d model.
    units : str, Units, or dict, optional
        Units specification (e.g. 'kN, m'). If None, inferred from structure.units.
    default_load_case_id : str, optional
        Target load case ID to store existing structure loads into. Defaults to 'LIVE'.
    """
    _check_struct_core()

    structure = source.structure if hasattr(source, "structure") else source

    model = StructuralModel()

    units_spec = units or getattr(structure, "units", None)
    if units_spec is not None:
        model.units = parse_units(units_spec) if isinstance(units_spec, (str, dict)) else units_spec

    # 1. Nodes
    for n in structure.nodes.values():
        model.nodes.append(
            ScNode(
                id=n.id,
                x=n.x,
                y=n.y,
                z=n.z,
                mass=float(n.mass[0]),
                inertia=float(n.inertia[0]),
            )
        )

    # 2. Supports — full 6-DOF mapping using the extended struct_core.Support schema.
    for n in structure.nodes.values():
        if any(n.support):
            model.supports.append(
                ScSupport(
                    node_id=n.id,
                    ux=bool(n.support[0]),
                    uy=bool(n.support[1]),
                    uz=bool(n.support[2]),
                    rx=bool(n.support[3]),
                    ry=bool(n.support[4]),
                    rz=bool(n.support[5]),
                )
            )

    # 3. Materials and sections deduplication
    mat_ids = {}
    sec_ids = {}
    for el in structure.elements.values():
        if hasattr(el, "material") and el.material is not None:
            mat = el.material
            mat_key = (mat.E, getattr(mat, "nu", 0.3), mat.rho, getattr(mat, "unit_weight", None))
            if mat_key not in mat_ids:
                mid = f"mat_{len(mat_ids)+1}"
                mat_ids[mat_key] = mid
                model.materials.append(
                    ScElasticMaterial(
                        id=mid,
                        E=mat.E,
                        nu=getattr(mat, "nu", 0.3),
                        rho=mat.rho,
                        unit_weight=getattr(mat, "unit_weight", None),
                    )
                )

        if hasattr(el, "section") and el.section is not None:
            sec = el.section
            as2_val = getattr(sec, "As2", 0.0) or 0.0
            as3_val = getattr(sec, "As3", 0.0) or 0.0
            sec_key = (sec.A, sec.I2, sec.I3, sec.J, as2_val, as3_val)
            if sec_key not in sec_ids:
                sid = f"sec_{len(sec_ids)+1}"
                sec_ids[sec_key] = sid
                model.sections.append(
                    ScGeneralSection(
                        id=sid,
                        A=sec.A,
                        I2=sec.I2 if sec.I2 > 0 else 1.0,
                        I3=sec.I3 if sec.I3 > 0 else 1.0,
                        J=sec.J if sec.J > 0 else 1.0,
                        As2=as2_val if as2_val > 0 else None,
                        As3=as3_val if as3_val > 0 else None,
                    )
                )

    # 4. Elements
    for el in structure.elements.values():
        mat_key = (el.material.E, getattr(el.material, "nu", 0.3), el.material.rho, getattr(el.material, "unit_weight", None))
        mid = mat_ids.get(mat_key)
        as2_val = getattr(el.section, "As2", 0.0) or 0.0
        as3_val = getattr(el.section, "As3", 0.0) or 0.0
        sid = sec_ids.get((el.section.A, el.section.I2, el.section.I3, el.section.J, as2_val, as3_val))

        if isinstance(el, TrussElement):
            model.elements.append(
                ScTrussElement(
                    id=el.id,
                    start_node=el.node_i.id,
                    end_node=el.node_j.id,
                    material_id=mid,
                    section_id=sid,
                )
            )
        elif isinstance(el, SpringElement):
            model.elements.append(
                ScSpringElement(
                    id=el.id,
                    start_node=el.node_i.id,
                    end_node=el.node_j.id,
                    kx=float(el.k_diag[0]),
                    ky=float(el.k_diag[1]),
                    kz=float(el.k_diag[2]),
                )
            )
        else:
            model.elements.append(
                ScBeamElement(
                    id=el.id,
                    start_node=el.node_i.id,
                    end_node=el.node_j.id,
                    material_id=mid,
                    section_id=sid,
                    include_shear_deformation=bool(getattr(el, "include_shear_deformation", False)),
                )
            )

    # 5. Loads (transfer existing loads into default_load_case_id)
    sc_point_loads = []
    sc_element_loads = []

    for n in structure.nodes.values():
        if hasattr(n, "load") and any(abs(v) > 0.0 for v in n.load):
            sc_point_loads.append(
                ScPointLoad(
                    node_id=n.id,
                    fx=float(n.load[0]),
                    fy=float(n.load[1]),
                    fz=float(n.load[2]),
                    mx=float(n.load[3]),
                    my=float(n.load[4]),
                    mz=float(n.load[5]),
                )
            )

    for load in getattr(structure, "loads", []):
        if isinstance(load, PointLoad):
            sc_point_loads.append(
                ScPointLoad(
                    node_id=load.node.id,
                    fx=float(load.fx),
                    fy=float(load.fy),
                    fz=float(load.fz),
                    mx=float(load.mx),
                    my=float(load.my),
                    mz=float(load.mz),
                )
            )
        elif isinstance(load, DistributedLoad):
            is_loc = getattr(load, "coord_system", "global").lower() == "local"
            if is_loc:
                sc_element_loads.append(
                    ScDistributedLoad(
                        element_id=load.element.id,
                        w1=float(getattr(load, "w1", 0.0) or 0.0),
                        w2=float(getattr(load, "w2", 0.0) or 0.0),
                        w3=float(getattr(load, "w3", 0.0) or 0.0),
                        is_local=True,
                    )
                )
            else:
                sc_element_loads.append(
                    ScDistributedLoad(
                        element_id=load.element.id,
                        wx=float(getattr(load, "wx", 0.0) or 0.0),
                        wy=float(getattr(load, "wy", 0.0) or 0.0),
                        wz=float(getattr(load, "wz", 0.0) or 0.0),
                        is_local=False,
                    )
                )

    if sc_point_loads or sc_element_loads:
        model.load_cases.append(
            ScLoadCase(
                id=default_load_case_id,
                name="Live Load" if default_load_case_id == "LIVE" else default_load_case_id,
                point_loads=sc_point_loads,
                element_loads=sc_element_loads,
            )
        )

    return model


def result_to_core(source: Union[Results, Structure, object]) -> AnalysisResult:
    """
    Convert fem3d analysis outputs into a struct_core.AnalysisResult.
    """
    _check_struct_core()

    # Accept a Results object directly or a Structure/SimpleFrame
    if isinstance(source, Results):
        results = source
    else:
        results = Results(source)

    disp_df = results.node_displacements()
    reac_df = results.reactions()
    forces_df = results.element_forces()

    node_results: dict = {}
    reac_by_node = {row["node"]: row for _, row in reac_df.iterrows()}

    for _, row in disp_df.iterrows():
        nid = row["node"]
        d = NodeDisplacement(
            ux=float(row["ux"]),
            uy=float(row["uy"]),
            uz=float(row["uz"]),
            rx=float(row["rx"]),
            ry=float(row["ry"]),
            rz=float(row["rz"]),
        )
        r = None
        if nid in reac_by_node:
            rr = reac_by_node[nid]
            r = NodeReaction(
                fx=float(rr["Fx"]),
                fy=float(rr["Fy"]),
                fz=float(rr["Fz"]),
                mx=float(rr["Mx"]),
                my=float(rr["My"]),
                mz=float(rr["Mz"]),
            )
        node_results[nid] = NodeResult(node_id=nid, displacement=d, reaction=r)

    element_results: dict = {}
    for _, row in forces_df.iterrows():
        eid = row["element"]
        sf_i = BeamSectionForce(
            station=0.0,
            P=float(row["f1_i"]),
            V2=float(row["f2_i"]),
            V3=float(row["f3_i"]),
            T=float(row["m1_i"]),
            M2=float(row["m2_i"]),
            M3=float(row["m3_i"]),
        )
        sf_j = BeamSectionForce(
            station=1.0,
            P=float(row["f1_j"]),
            V2=float(row["f2_j"]),
            V3=float(row["f3_j"]),
            T=float(row["m1_j"]),
            M2=float(row["m2_j"]),
            M3=float(row["m3_j"]),
        )
        element_results[eid] = ElementResult(
            element_id=eid,
            forces=[sf_i, sf_j],
            max_axial=max(abs(sf_i.P), abs(sf_j.P)),
            max_moment=max(abs(sf_i.M3), abs(sf_j.M3), abs(sf_i.M2), abs(sf_j.M2)),
            max_shear=max(abs(sf_i.V2), abs(sf_j.V2), abs(sf_i.V3), abs(sf_j.V3)),
        )

    return AnalysisResult(
        analysis_case_id="fem3d_linear_static",
        node_results=node_results,
        element_results=element_results,
    )
