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
    load_case_id: Optional[Union[int, str]] = None,
    load_combination_id: Optional[str] = None,
) -> Structure:
    """
    Convert a struct_core.StructuralModel or struct_core.Project into a fem3d.Structure.
    """
    _check_struct_core()

    if isinstance(source, Project):
        if source.model is None:
            raise ValueError("Project contains no structural model.")
        model = source.model
    else:
        model = source

    structure = Structure()

    # 1. Nodes
    for n in model.nodes:
        node = Node(nid=n.id, x=n.x, y=n.y, z=getattr(n, "z", 0.0))
        m = getattr(n, "mass", 0.0)
        it = getattr(n, "inertia", 0.0)
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
        rho = getattr(m, "rho", 0.0)
        mat_map[m.id] = ElasticMaterial(E=E, nu=nu, rho=rho)

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
            sec_map[sec.id] = Section(A=A, I2=I2, I3=I3, J=J)

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
            fem_el = FrameElement(eid=elem.id, node_i=n_i, node_j=n_j, material=mat, section=sec)

        structure.add_element(fem_el)

    # 6. Loads
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
                continue
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
            for dl in list(getattr(lc, "element_loads", []) or getattr(lc, "distributed_loads", [])):
                eid = dl.element_id
                if eid in structure.elements:
                    el = structure.elements[eid]
                    w1 = float(getattr(dl, "w1", getattr(dl, "wx", 0.0)) or 0.0) * factor
                    w2 = float(getattr(dl, "w2", getattr(dl, "wy", 0.0)) or 0.0) * factor
                    w3 = float(getattr(dl, "w3", getattr(dl, "wz", 0.0)) or 0.0) * factor
                    cs = "local" if getattr(dl, "is_local", False) else "global"
                    structure.add_load(DistributedLoad(el, w1=w1, w2=w2, w3=w3, coord_system=cs))
    else:
        load_cases = getattr(model, "load_cases", [])
        if load_cases:
            if load_case_id is not None and str(load_case_id).upper() not in ("ALL", "TOTAL", "ENVELOPE"):
                cases_to_apply = [lc for lc in load_cases if str(lc.id).upper() == str(load_case_id).upper()]
                if not cases_to_apply:
                    cases_to_apply = [lc for lc in load_cases if str(getattr(lc, "name", "")).upper() == str(load_case_id).upper()]
                if not cases_to_apply:
                    active_cases = [
                        lc for lc in load_cases
                        if lc.point_loads or getattr(lc, "element_loads", []) or getattr(lc, "distributed_loads", [])
                    ]
                    cases_to_apply = active_cases if active_cases else [load_cases[0]]
            else:
                cases_to_apply = load_cases

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

                for dl in list(getattr(lc, "element_loads", []) or getattr(lc, "distributed_loads", [])):
                    eid = dl.element_id
                    if eid in structure.elements:
                        el = structure.elements[eid]
                        w1 = float(getattr(dl, "w1", getattr(dl, "wx", 0.0)) or 0.0)
                        w2 = float(getattr(dl, "w2", getattr(dl, "wy", 0.0)) or 0.0)
                        w3 = float(getattr(dl, "w3", getattr(dl, "wz", 0.0)) or 0.0)
                        cs = "local" if getattr(dl, "is_local", False) else "global"
                        structure.add_load(DistributedLoad(el, w1=w1, w2=w2, w3=w3, coord_system=cs))

        # Legacy backward compatibility with model.loads
        for ld in getattr(model, "loads", []):
            if getattr(ld, "type", "") == "nodal_point":
                nid = ld.node_id
                if nid in structure.nodes:
                    fx = getattr(ld, "fx", 0.0)
                    fy = getattr(ld, "fy", 0.0)
                    fz = getattr(ld, "fz", 0.0)
                    mx = getattr(ld, "mx", 0.0)
                    my = getattr(ld, "my", 0.0)
                    mz = getattr(ld, "mz", 0.0)
                    structure.add_load(PointLoad(structure.nodes[nid], fx=fx, fy=fy, fz=fz, mx=mx, my=my, mz=mz))

    return structure


def model_to_core(source: Union[Structure, object]) -> StructuralModel:
    """
    Convert a fem3d Structure or SimpleFrame into a struct_core.StructuralModel.
    """
    _check_struct_core()

    structure = source.structure if hasattr(source, "structure") else source

    model = StructuralModel()

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
            mat_key = (el.material.E, getattr(el.material, "nu", 0.3), el.material.rho)
            if mat_key not in mat_ids:
                mid = f"mat_{len(mat_ids)+1}"
                mat_ids[mat_key] = mid
                model.materials.append(
                    ScElasticMaterial(
                        id=mid,
                        E=el.material.E,
                        nu=getattr(el.material, "nu", 0.3),
                        rho=el.material.rho,
                    )
                )

        if hasattr(el, "section") and el.section is not None:
            sec = el.section
            sec_key = (sec.A, sec.I2, sec.I3, sec.J)
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
                    )
                )

    # 4. Elements
    for el in structure.elements.values():
        if isinstance(el, TrussElement):
            mid = mat_ids.get((el.material.E, getattr(el.material, "nu", 0.3), el.material.rho))
            sid = sec_ids.get((el.section.A, el.section.I2, el.section.I3, el.section.J))
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
            mid = mat_ids.get((el.material.E, getattr(el.material, "nu", 0.3), el.material.rho))
            sid = sec_ids.get((el.section.A, el.section.I2, el.section.I3, el.section.J))
            model.elements.append(
                ScBeamElement(
                    id=el.id,
                    start_node=el.node_i.id,
                    end_node=el.node_j.id,
                    material_id=mid,
                    section_id=sid,
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
