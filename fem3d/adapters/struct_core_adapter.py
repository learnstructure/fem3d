"""
Adapter module bridging fem3d and struct_core.

Provides three conversion functions:
- model_from_core : convert a struct_core.StructuralModel into a fem3d.Structure
- model_to_core   : convert a fem3d.Structure into a struct_core.StructuralModel
- result_to_core  : convert fem3d analysis outputs into a struct_core.AnalysisResult
"""

from __future__ import annotations
import copy
from typing import Any, Dict, Optional, Union, TYPE_CHECKING
import numpy as np

try:
    from struct_core import (
        StructuralModel,
        Project,
        Metadata,
        Node as ScNode,
        Support as ScSupport,
        ElasticMaterial as ScElasticMaterial,
        BilinearMaterial as ScBilinearMaterial,
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
from ..loads import PointLoad, DistributedLoad, ElementPointLoad
from ..structure import Structure
from ..results import Results


def _apply_mass_modifier(element: Any, base_section: Any, modifiers: Dict[str, float]) -> None:
    """Give a frame element the mass of struct_core's modifiers, as SAP2000 and ETABS define them.

    The mass modifier scales all the member's mass (translational and rotational); the A modifier scales the axial
    stiffness only. The element's mass is rho x (its area) per length plus ``element.mass``, and its rotational
    inertia rho x I: the element gets a copy of its material with rho x mass modifier, and ``element.mass`` the
    difference rho x mass x A0 x (1 - A modifier), so that its mass per length is rho x A0 x mass (A0: unmodified).
    """
    mass_mod = float(modifiers.get("mass", 1.0))
    area_mod = float(modifiers.get("A", 1.0))
    if mass_mod == 1.0 and area_mod == 1.0:
        return
    rho = base_section.material.rho
    if mass_mod != 1.0:
        element.material = copy.copy(element.material)
        element.material.rho = rho * mass_mod
    element.mass += rho * mass_mod * base_section.A * (1.0 - area_mod)


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
    flexural_stiffness_modifiers: Optional[Dict[Any, float]] = None,
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
    flexural_stiffness_modifiers : dict, optional
        Multiplier on both moments of inertia (I2, I3) per element ID (e.g. ACI 318
        Table 6.6.3.1.1: 0.70 for columns, 0.35 for beams). Area, J, and mass are not modified.
    """
    _check_struct_core()

    if isinstance(source, Project):
        if source.model is None:
            raise ValueError("Project contains no structural model.")
        model = source.model
    else:
        model = source

    if getattr(model, "shell_elements", None):
        raise NotImplementedError(
            f"fem3d does not analyze shell elements yet (fem3d/SHELL_PLAN.md, milestone M5); the model has "
            f"{len(model.shell_elements)}."
        )

    active_units = parse_units(
        units
        or getattr(getattr(source, "metadata", None), "units", None)
        or getattr(source, "units", None)
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
            kx = float(getattr(s, "kx", 0.0) or 0.0)
            ky = float(getattr(s, "ky", 0.0) or 0.0)
            kz = float(getattr(s, "kz", 0.0) or 0.0)
            krx = float(getattr(s, "krx", 0.0) or 0.0)
            kry = float(getattr(s, "kry", 0.0) or 0.0)
            krz = float(getattr(s, "krz", 0.0) or 0.0)
            if kx > 0.0 or ky > 0.0 or kz > 0.0 or krx > 0.0 or kry > 0.0 or krz > 0.0:
                nd.set_springs(kx=kx, ky=ky, kz=kz, krx=krx, kry=kry, krz=krz)

    # 3. Materials dictionary (every struct_core elastic material has E; concrete and rebar are elastic for analysis)
    mat_map = {}
    for m in model.materials:
        if isinstance(m, ScBilinearMaterial) or not isinstance(m, ScElasticMaterial):
            raise TypeError(f"Material {m.id!r}: fem3d has no model for {type(m).__name__}")
        mat_map[m.id] = ElasticMaterial(E=m.E, nu=m.nu, rho=_convert_rho(m), unit_weight=m.unit_weight)

    # 4. Sections dictionary, each with its material
    sec_map = {}
    for sec in model.sections:
        if sec.material_id not in mat_map:
            raise ValueError(f"Section {sec.id!r} references missing material {sec.material_id!r}")
        mat = mat_map[sec.material_id]
        if sec.type == "rectangular":
            sec_map[sec.id] = Section.from_rectangle(width=sec.b, depth=sec.h, material=mat)
        elif sec.type == "circular":
            sec_map[sec.id] = Section.from_circle(diameter=sec.d, material=mat)
        else:
            needs_shear = (
                include_shear_deformation is True
                or any(
                    getattr(e, "include_shear_deformation", False)
                    for e in model.elements
                    if getattr(e, "section_id", None) == sec.id
                )
            )
            if needs_shear and (sec.As2 is None or sec.As3 is None):
                missing = []
                if sec.As2 is None:
                    missing.append("As2")
                if sec.As3 is None:
                    missing.append("As3")
                raise ValueError(
                    f"Section {sec.id!r} is a GeneralSection without shear area(s) {', '.join(missing)}, but "
                    f"shear deformation was requested. Specify As2 and As3 on the section or disable shear deformation."
                )
            sec_map[sec.id] = Section(A=sec.A, I2=sec.I2, I3=sec.I3, J=sec.J, As2=sec.As2, As3=sec.As3, material=mat)

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

        if elem_type in ("truss", "beam") and elem.section_id not in sec_map:
            raise ValueError(f"Element {elem.id!r} references missing section {elem.section_id!r}")
        if elem_type == "truss":
            fem_el = TrussElement(eid=elem.id, node_i=n_i, node_j=n_j, section=sec_map[elem.section_id])
        elif elem_type == "spring":
            kx = getattr(elem, "kx", 0.0)
            ky = getattr(elem, "ky", 0.0)
            kz = getattr(elem, "kz", 0.0)
            fem_el = SpringElement(eid=elem.id, node_i=n_i, node_j=n_j, kx=kx, ky=ky, kz=kz)
        else:  # Beam / Frame
            sec = sec_map[elem.section_id]
            elem_mods = getattr(elem, "stiffness_modifiers", None)
            if elem_mods or (flexural_stiffness_modifiers and elem.id in flexural_stiffness_modifiers):
                sec = copy.copy(sec)
                if elem_mods:
                    if "A" in elem_mods:
                        sec.A *= float(elem_mods["A"])
                    if "I2" in elem_mods:
                        sec.I2 *= float(elem_mods["I2"])
                    if "I3" in elem_mods:
                        sec.I3 *= float(elem_mods["I3"])
                    if "J" in elem_mods:
                        sec.J *= float(elem_mods["J"])
                    if "As2" in elem_mods:
                        sec.As2 *= float(elem_mods["As2"])
                    if "As3" in elem_mods:
                        sec.As3 *= float(elem_mods["As3"])
                if flexural_stiffness_modifiers and elem.id in flexural_stiffness_modifiers:
                    factor = float(flexural_stiffness_modifiers[elem.id])
                    sec.I2 *= factor
                    sec.I3 *= factor
            elem_shear = (
                include_shear_deformation
                if include_shear_deformation is not None
                else getattr(elem, "include_shear_deformation", False)
            )
            rel_i = getattr(elem, "release_i", None)
            rel_j = getattr(elem, "release_j", None)
            releases_i_3d = [bool(x) for x in rel_i] if rel_i is not None and any(rel_i) else None
            releases_j_3d = [bool(x) for x in rel_j] if rel_j is not None and any(rel_j) else None
            fem_el = FrameElement(
                eid=elem.id,
                node_i=n_i,
                node_j=n_j,
                material=sec.material,
                section=sec,
                include_shear_deformation=elem_shear,
                roll_angle=float(getattr(elem, "roll_angle", 0.0) or 0.0),
                releases_i=releases_i_3d,
                releases_j=releases_j_3d,
            )
            if elem_mods:
                _apply_mass_modifier(fem_el, sec_map[elem.section_id], elem_mods)

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
                    f"Load combination '{target_comb.id}' references unknown load case '{factor_obj.load_case_id}'."
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
                    st = float(getattr(dl, "start", 0.0) or 0.0)
                    en = float(getattr(dl, "end", 1.0) or 1.0)
                    if getattr(dl, "is_local", False):
                        w1 = float(getattr(dl, "w1", 0.0) or 0.0) * factor
                        w2 = float(getattr(dl, "w2", 0.0) or 0.0) * factor
                        w3 = float(getattr(dl, "w3", 0.0) or 0.0) * factor
                        w1e = getattr(dl, "w1_end", None)
                        w2e = getattr(dl, "w2_end", None)
                        w3e = getattr(dl, "w3_end", None)
                        w1_end = float(w1 if w1e is None else w1e * factor)
                        w2_end = float(w2 if w2e is None else w2e * factor)
                        w3_end = float(w3 if w3e is None else w3e * factor)
                        structure.add_load(DistributedLoad(el, w1=w1, w2=w2, w3=w3, w1_end=w1_end, w2_end=w2_end, w3_end=w3_end, start=st, end=en, coord_system="local"))
                    else:
                        wx = float(getattr(dl, "wx", 0.0) or 0.0) * factor
                        wy = float(getattr(dl, "wy", 0.0) or 0.0) * factor
                        wz = float(getattr(dl, "wz", 0.0) or 0.0) * factor
                        wxe = getattr(dl, "wx_end", None)
                        wye = getattr(dl, "wy_end", None)
                        wze = getattr(dl, "wz_end", None)
                        wx_end = float(wx if wxe is None else wxe * factor)
                        wy_end = float(wy if wye is None else wye * factor)
                        wz_end = float(wz if wze is None else wze * factor)
                        structure.add_load(DistributedLoad(el, wx=wx, wy=wy, wz=wz, wx_end=wx_end, wy_end=wy_end, wz_end=wz_end, start=st, end=en, coord_system="global"))

            for epl in getattr(lc, "element_point_loads", []):
                eid = epl.element_id
                if eid in structure.elements:
                    el = structure.elements[eid]
                    pos = float(getattr(epl, "position", 0.0)) * el.length
                    if getattr(epl, "is_local", False):
                        p1 = float(getattr(epl, "fx", 0.0) or 0.0) * factor
                        p2 = float(getattr(epl, "fy", 0.0) or 0.0) * factor
                        p3 = float(getattr(epl, "fz", 0.0) or 0.0) * factor
                        m1 = float(getattr(epl, "mx", 0.0) or 0.0) * factor
                        m2 = float(getattr(epl, "my", 0.0) or 0.0) * factor
                        m3 = float(getattr(epl, "mz", 0.0) or 0.0) * factor
                        structure.add_load(ElementPointLoad(el, p1=p1, p2=p2, p3=p3, m1=m1, m2=m2, m3=m3, x=pos, coord_system="local"))
                    else:
                        px = float(getattr(epl, "fx", 0.0) or 0.0) * factor
                        py = float(getattr(epl, "fy", 0.0) or 0.0) * factor
                        pz = float(getattr(epl, "fz", 0.0) or 0.0) * factor
                        mx = float(getattr(epl, "mx", 0.0) or 0.0) * factor
                        my = float(getattr(epl, "my", 0.0) or 0.0) * factor
                        mz = float(getattr(epl, "mz", 0.0) or 0.0) * factor
                        structure.add_load(ElementPointLoad(el, px=px, py=py, pz=pz, mx=mx, my=my, mz=mz, x=pos, coord_system="global"))
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
                        st = float(getattr(dl, "start", 0.0) or 0.0)
                        en = float(getattr(dl, "end", 1.0) or 1.0)
                        if getattr(dl, "is_local", False):
                            w1 = float(getattr(dl, "w1", 0.0) or 0.0)
                            w2 = float(getattr(dl, "w2", 0.0) or 0.0)
                            w3 = float(getattr(dl, "w3", 0.0) or 0.0)
                            w1e = getattr(dl, "w1_end", None)
                            w2e = getattr(dl, "w2_end", None)
                            w3e = getattr(dl, "w3_end", None)
                            w1_end = float(w1 if w1e is None else w1e)
                            w2_end = float(w2 if w2e is None else w2e)
                            w3_end = float(w3 if w3e is None else w3e)
                            structure.add_load(DistributedLoad(el, w1=w1, w2=w2, w3=w3, w1_end=w1_end, w2_end=w2_end, w3_end=w3_end, start=st, end=en, coord_system="local"))
                        else:
                            wx = float(getattr(dl, "wx", 0.0) or 0.0)
                            wy = float(getattr(dl, "wy", 0.0) or 0.0)
                            wz = float(getattr(dl, "wz", 0.0) or 0.0)
                            wxe = getattr(dl, "wx_end", None)
                            wye = getattr(dl, "wy_end", None)
                            wze = getattr(dl, "wz_end", None)
                            wx_end = float(wx if wxe is None else wxe)
                            wy_end = float(wy if wye is None else wye)
                            wz_end = float(wz if wze is None else wze)
                            structure.add_load(DistributedLoad(el, wx=wx, wy=wy, wz=wz, wx_end=wx_end, wy_end=wy_end, wz_end=wz_end, start=st, end=en, coord_system="global"))

                for epl in getattr(lc, "element_point_loads", []):
                    eid = epl.element_id
                    if eid in structure.elements:
                        el = structure.elements[eid]
                        pos = float(getattr(epl, "position", 0.0)) * el.length
                        if getattr(epl, "is_local", False):
                            p1 = float(getattr(epl, "fx", 0.0) or 0.0)
                            p2 = float(getattr(epl, "fy", 0.0) or 0.0)
                            p3 = float(getattr(epl, "fz", 0.0) or 0.0)
                            m1 = float(getattr(epl, "mx", 0.0) or 0.0)
                            m2 = float(getattr(epl, "my", 0.0) or 0.0)
                            m3 = float(getattr(epl, "mz", 0.0) or 0.0)
                            structure.add_load(ElementPointLoad(el, p1=p1, p2=p2, p3=p3, m1=m1, m2=m2, m3=m3, x=pos, coord_system="local"))
                        else:
                            px = float(getattr(epl, "fx", 0.0) or 0.0)
                            py = float(getattr(epl, "fy", 0.0) or 0.0)
                            pz = float(getattr(epl, "fz", 0.0) or 0.0)
                            mx = float(getattr(epl, "mx", 0.0) or 0.0)
                            my = float(getattr(epl, "my", 0.0) or 0.0)
                            mz = float(getattr(epl, "mz", 0.0) or 0.0)
                            structure.add_load(ElementPointLoad(el, px=px, py=py, pz=pz, mx=mx, my=my, mz=mz, x=pos, coord_system="global"))

                if getattr(lc, "include_self_weight", False):
                    _apply_case_self_weight_3d(lc.id, factor=1.0)

    return structure


def model_to_core(
    source: Union[Structure, object],
    units: Optional[Union[str, Units, dict]] = None,
    default_load_case_id: str = "LIVE",
) -> Project:
    """
    Convert a fem3d Structure or SimpleFrame into a struct_core.Project.

    Parameters
    ----------
    source : Structure or SimpleFrame
        Source fem3d model.
    units : str, Units, or dict, optional
        Units specification (e.g. 'kN, m'). If None, inferred from structure.units.
    default_load_case_id : str, optional
        Target load case ID to store existing structure loads into. Defaults to 'LIVE'.

    Returns
    -------
    struct_core.Project
        A project instance wrapping the structural model and units metadata.
    """
    _check_struct_core()

    structure = source.structure if hasattr(source, "structure") else source

    units_spec = units or getattr(structure, "units", None)
    target_units = (parse_units(units_spec) if isinstance(units_spec, (str, dict)) else units_spec) or Metadata().units
    mass_factor = target_units.consistent_mass_factor

    model = StructuralModel(dimension="3D", vertical_axis="Z")

    # 1. Nodes
    for n in structure.nodes.values():
        model.nodes.append(
            ScNode(
                id=n.id,
                x=n.x,
                y=n.y,
                z=n.z,
                mass=float(n.mass[0]) / mass_factor,
                inertia=float(n.inertia[0]) / mass_factor,
            )
        )

    # 2. Supports — full 6-DOF mapping using the extended struct_core.Support schema.
    for n in structure.nodes.values():
        sup = getattr(n, "support", None) or [False] * 6
        springs = getattr(n, "springs", None) or [0.0] * 6
        if any(sup) or any(s > 0.0 for s in springs):
            model.supports.append(
                ScSupport(
                    node_id=n.id,
                    ux=bool(sup[0]),
                    uy=bool(sup[1]),
                    uz=bool(sup[2]),
                    rx=bool(sup[3]),
                    ry=bool(sup[4]),
                    rz=bool(sup[5]),
                    kx=float(springs[0]),
                    ky=float(springs[1]),
                    kz=float(springs[2]),
                    krx=float(springs[3]),
                    kry=float(springs[4]),
                    krz=float(springs[5]),
                )
            )

    # 3. Materials and sections deduplication (a section is keyed by its properties and its material)
    def _mat_key(mat):
        return (mat.E, mat.nu, mat.rho, getattr(mat, "unit_weight", None))

    def _sec_key(sec, material_id):
        as2_val = getattr(sec, "As2", 0.0) or 0.0
        as3_val = getattr(sec, "As3", 0.0) or 0.0
        return (sec.A, sec.I2, sec.I3, sec.J, as2_val, as3_val, material_id)

    mat_ids = {}
    sec_ids = {}
    framed = [el for el in structure.elements.values() if not isinstance(el, SpringElement)]
    for el in framed:
        mat = el.section.material
        if _mat_key(mat) not in mat_ids:
            mid = f"mat_{len(mat_ids)+1}"
            mat_ids[_mat_key(mat)] = mid
            model.materials.append(
                ScElasticMaterial(
                    id=mid,
                    E=mat.E,
                    nu=mat.nu,
                    rho=mat.rho / mass_factor,
                    unit_weight=getattr(mat, "unit_weight", None),
                )
            )

        sec = el.section
        sec_key = _sec_key(sec, mat_ids[_mat_key(mat)])
        if sec_key not in sec_ids:
            sid = f"sec_{len(sec_ids)+1}"
            sec_ids[sec_key] = sid
            as2_val, as3_val = sec_key[4], sec_key[5]
            model.sections.append(
                ScGeneralSection(
                    id=sid,
                    material_id=sec_key[6],
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
        if not isinstance(el, SpringElement):
            sid = sec_ids[_sec_key(el.section, mat_ids[_mat_key(el.section.material)])]

        if isinstance(el, TrussElement):
            model.elements.append(
                ScTrussElement(
                    id=el.id,
                    start_node=el.node_i.id,
                    end_node=el.node_j.id,
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
            rel_i = getattr(el, "releases_i", None)
            rel_j = getattr(el, "releases_j", None)
            rel_i_6d = [bool(x) for x in rel_i] if rel_i is not None and len(rel_i) >= 6 else [False]*6
            rel_j_6d = [bool(x) for x in rel_j] if rel_j is not None and len(rel_j) >= 6 else [False]*6
            model.elements.append(
                ScBeamElement(
                    id=el.id,
                    start_node=el.node_i.id,
                    end_node=el.node_j.id,
                    section_id=sid,
                    release_i=rel_i_6d,
                    release_j=rel_j_6d,
                    roll_angle=float(getattr(el, "roll_angle", 0.0)),
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

    return Project(model=model, metadata=Metadata(units=target_units))


def _member_stations(length: float, f_i: list, loads: list) -> list:
    """Internal forces at STANDARD_STATIONS from the start-node end forces and the member's loads.

    ``f_i`` = [f1, f2, f3, m1, m2, m3] are the local forces the start node exerts on the member.
    Equilibrium of the segment [0, x] gives the internal actions on its cut face, with uniform/varying/partial
    (DistributedLoad) and concentrated (ElementPointLoad) member loads in local axes:
        P(x)  = -(f1 + R1(x) + sum p1)              T(x)  = -(m1 + sum m1p)
        V2(x) = -(f2 + R2(x) + sum p2)              V3(x) = -(f3 + R3(x) + sum p3)
        M3(x) = -m3 + f2 x + M2_res(x) + sum[p2 (x - a) - m3p]
        M2(x) = -m2 - f3 x - M3_res(x) - sum[p3 (x - a) + m2p]
    At x = L these equal the end forces at node j.
    """
    from struct_core.analysis_result import STANDARD_STATIONS
    from ..loads import ElementPointLoad

    f1, f2, f3, m1, m2, m3 = f_i
    dist_loads = []
    points = []
    for load in loads:
        if isinstance(load, DistributedLoad):
            dist_loads.append(load)
        elif isinstance(load, ElementPointLoad):
            points.append(load)
        else:
            raise TypeError(f"Unsupported member load {type(load).__name__} for internal force stations.")

    stations = []
    for s in STANDARD_STATIONS:
        x = s * length
        r1_tot = 0.0
        r2_tot = 0.0
        r3_tot = 0.0
        m2_res_tot = 0.0
        m3_res_tot = 0.0

        for dl in dist_loads:
            x1 = dl.start * length
            x2 = dl.end * length
            if x > x1 and x2 > x1:
                x_cut = min(x, x2)
                delta = x_cut - x1
                l_span = x2 - x1
                m1_slope = (dl.w1_end - dl.w1) / l_span
                m2_slope = (dl.w2_end - dl.w2) / l_span
                m3_slope = (dl.w3_end - dl.w3) / l_span

                r1_tot += dl.w1 * delta + 0.5 * m1_slope * delta**2
                r2 = dl.w2 * delta + 0.5 * m2_slope * delta**2
                r2_tot += r2
                m2_res_tot += (x - x1) * r2 - (0.5 * dl.w2 * delta**2 + (1.0 / 3.0) * m2_slope * delta**3)

                r3 = dl.w3 * delta + 0.5 * m3_slope * delta**2
                r3_tot += r3
                m3_res_tot += (x - x1) * r3 - (0.5 * dl.w3 * delta**2 + (1.0 / 3.0) * m3_slope * delta**3)

        acting = [p for p in points if p.x < x or (s == 1.0 and p.x <= length)]
        stations.append(BeamSectionForce(
            station=s,
            P=-(f1 + r1_tot + sum(p.p1 for p in acting)),
            V2=-(f2 + r2_tot + sum(p.p2 for p in acting)),
            V3=-(f3 + r3_tot + sum(p.p3 for p in acting)),
            T=-(m1 + sum(p.m1 for p in acting)),
            M2=-m2 - f3 * x - m3_res_tot - sum(p.p3 * (x - p.x) + p.m2 for p in acting),
            M3=-m3 + f2 * x + m2_res_tot + sum(p.p2 * (x - p.x) - p.m3 for p in acting),
        ))
    return stations


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

    structure = results.structure
    loads_by_element: dict = {}
    for load in getattr(structure, "loads", []) or []:
        el = getattr(load, "element", None)
        if el is not None:
            loads_by_element.setdefault(el.id, []).append(load)

    element_results: dict = {}
    for _, row in forces_df.iterrows():
        eid = row["element"]
        f_i = [float(row[k]) for k in ("f1_i", "f2_i", "f3_i", "m1_i", "m2_i", "m3_i")]
        stations = _member_stations(structure.elements[eid].length, f_i, loads_by_element.get(eid, []))
        # Envelopes: the signed station value with the largest magnitude (tension-positive axial force),
        # consistent with fem2d and struct_core superposition
        element_results[eid] = ElementResult(
            element_id=eid,
            forces=stations,
            max_axial=max((s.P for s in stations), key=abs),
            max_moment=max((m for s in stations for m in (s.M3, s.M2)), key=abs),
            max_shear=max((v for s in stations for v in (s.V2, s.V3)), key=abs),
        )

    return AnalysisResult(
        analysis_case_id="fem3d_linear_static",
        node_results=node_results,
        element_results=element_results,
    )
