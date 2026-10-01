"""
SimpleFrame module providing a convenient, high-level API for creating and solving 3D frame and truss models.
"""

from typing import Iterable, Optional, Tuple, Union
import numpy as np

from ..nodes import Node
from ..materials.elastic import ElasticMaterial
from ..sections.section import Section
from ..elements.frame import FrameElement
from ..elements.truss import TrussElement
from ..elements.spring import SpringElement
from ..loads import DistributedLoad, ElementPointLoad, PointLoad
from ..structure import Structure


class SimpleFrame:
    """
    High-level fluent wrapper for constructing, solving, and evaluating 3D finite element structures.

    Coordinate System & Orientation Overview
    -----------------------------------------
    - **Global system (X, Y, Z)**: Right-handed Cartesian triad where **+Z is vertical upward**.
    - **Local axes (1, 2, 3)**:
      - 1 (v1): Centroidal axis directed from node_i to node_j.
      - 2 (v2): Cross-sectional axis. Resists bending in local 1-2 plane via I3 = ∫ x2^2 dA.
      - 3 (v3): Cross-sectional axis. Resists bending in local 1-3 plane via I2 = ∫ x3^2 dA.
      - By default (roll_angle=0, web_vector=None):
        - For horizontal/sloped members: v3 is in the global XY horizontal plane,
          and v2 is in the vertical plane containing the member (pointing upward).
        - For vertical members along +Z: v2 points along global +Y, v3 points along global -X.
      - `roll_angle` (degrees): Rotates (v2, v3) about v1 following right-hand rule.
      - `web_vector`: Directs the local 1-2 plane toward a specific 3D direction (e.g. web plane).

    Attributes
    ----------
    structure : Structure
        The underlying Structure instance.
    """

    def __init__(self):
        """Initialize a SimpleFrame wrapper with an empty Structure."""
        self.structure = Structure()

    def add_node(
        self,
        id: Union[int, str],
        x: float,
        y: float,
        z: float = 0.0,
    ) -> Node:
        """
        Add a 3D node to the structure.

        Parameters
        ----------
        id : int or str
            Unique node identifier.
        x : float
            X coordinate.
        y : float
            Y coordinate.
        z : float, optional
            Z coordinate. Defaults to 0.0.

        Returns
        -------
        Node
            The created Node instance.
        """
        node = Node(id, x, y, z)
        self.structure.add_node(node)
        return node

    def add_frame(
        self,
        id: Union[int, str],
        node_i_id: Union[int, str],
        node_j_id: Union[int, str],
        E: float,
        A: float,
        I2: Optional[float] = None,
        I3: Optional[float] = None,
        J: Optional[float] = None,
        roll_angle: float = 0.0,
        web_vector: Optional[Iterable] = None,
        nu: float = 0.3,
        G: Optional[float] = None,
        extra_mass: float = 0.0,
    ) -> FrameElement:
        """
        Add an elastic 3D beam-column frame element.

        Parameters
        ----------
        id : int or str
            Unique element identifier.
        node_i_id : int or str
            Start node identifier (node i).
        node_j_id : int or str
            End node identifier (node j).
        E : float
            Young's modulus.
        A : float
            Cross-sectional area.
        I2 : float, optional
            Second moment of area about local 2-axis (resists bending in local 1-3 plane).
            If omitted and I3 is given, set equal to I3.
        I3 : float, optional
            Second moment of area about local 3-axis (resists bending in local 1-2 plane).
            If omitted and I2 is given, set equal to I2.
        J : float, optional
            St. Venant torsional constant about local 1-axis. If omitted, approximated as I2 + I3.
        roll_angle : float, optional
            Member roll angle in degrees about local 1-axis (v1).
            Rotates the local (2, 3) axes. Defaults to 0.0.
        web_vector : iterable, optional
            Reference orientation vector pointing in the local 1-2 plane (e.g. web direction / v2).
            If provided, overrides default orientation. Defaults to None.
        nu : float, optional
            Poisson's ratio. Defaults to 0.3.
        G : float, optional
            Shear modulus. If None, derived as E / (2*(1+nu)).
        extra_mass : float, optional
            Distributed non-structural mass per unit length. Defaults to 0.0.

        Returns
        -------
        FrameElement
            The created FrameElement instance.
        """
        node_i = self.structure.nodes[node_i_id]
        node_j = self.structure.nodes[node_j_id]

        i2_val = I2
        i3_val = I3

        if i3_val is None and i2_val is not None:
            i3_val = float(i2_val)
        elif i2_val is None and i3_val is not None:
            i2_val = float(i3_val)
        elif i2_val is None and i3_val is None:
            i2_val = 0.0
            i3_val = 0.0
        else:
            i2_val = float(i2_val)
            i3_val = float(i3_val)

        if J is None:
            J_val = i2_val + i3_val if (i2_val + i3_val) > 0 else 0.0
        else:
            J_val = float(J)

        mat = ElasticMaterial(E=E, nu=nu, G=G)
        sec = Section(A=A, I2=i2_val, I3=i3_val, J=J_val)

        elem = FrameElement(
            eid=id,
            node_i=node_i,
            node_j=node_j,
            material=mat,
            section=sec,
            roll_angle=roll_angle,
            web_vector=np.asarray(web_vector, dtype=float) if web_vector is not None else None,
            extra_mass=extra_mass,
        )
        self.structure.add_element(elem)
        return elem

    def add_truss(
        self,
        id: Union[int, str],
        node_i_id: Union[int, str],
        node_j_id: Union[int, str],
        E: float,
        A: float,
        extra_mass: float = 0.0,
    ) -> TrussElement:
        """
        Add an elastic 3D space truss element.

        Parameters
        ----------
        id : int or str
            Unique element identifier.
        node_i_id : int or str
            Start node identifier.
        node_j_id : int or str
            End node identifier.
        E : float
            Young's modulus.
        A : float
            Cross-sectional area.
        extra_mass : float, optional
            Distributed mass. Defaults to 0.0.

        Returns
        -------
        TrussElement
            The created TrussElement instance.
        """
        node_i = self.structure.nodes[node_i_id]
        node_j = self.structure.nodes[node_j_id]
        mat = ElasticMaterial(E=E)
        sec = Section(A=A)
        elem = TrussElement(
            eid=id,
            node_i=node_i,
            node_j=node_j,
            material=mat,
            section=sec,
            extra_mass=extra_mass,
        )
        self.structure.add_element(elem)
        return elem

    def add_spring(
        self,
        id: Union[int, str],
        node_i_id: Union[int, str],
        node_j_id: Union[int, str],
        kx: float = 0.0,
        ky: float = 0.0,
        kz: float = 0.0,
        krx: float = 0.0,
        kry: float = 0.0,
        krz: float = 0.0,
    ) -> SpringElement:
        """Add a 3D spring element."""
        node_i = self.structure.nodes[node_i_id]
        node_j = self.structure.nodes[node_j_id]
        elem = SpringElement(
            eid=id,
            node_i=node_i,
            node_j=node_j,
            kx=kx,
            ky=ky,
            kz=kz,
            krx=krx,
            kry=kry,
            krz=krz,
        )
        self.structure.add_element(elem)
        return elem

    def add_support(
        self,
        node_id: Union[int, str],
        fixity: Union[Iterable, bool],
        *extra_flags,
    ):
        """
        Apply boundary support conditions to a node.

        Can be called with a 6-element list/tuple:
            frame.add_support(1, [1, 1, 1, 1, 1, 1])  # Fixed base in 3D
            frame.add_support(1, [1, 1, 1, 0, 0, 0])  # Pinned base in 3D
        or with positional flags:
            frame.add_support(1, 1, 1, 1, 0, 0, 0)
        """
        node = self.structure.nodes[node_id]
        if isinstance(fixity, (list, tuple)):
            node.set_support(fixity)
        elif len(extra_flags) > 0:
            flags = [fixity] + list(extra_flags)
            node.set_support(flags)
        else:
            node.set_support(fixity)

    def add_node_load(
        self,
        node_id: Union[int, str],
        load: Union[Iterable, float],
        *extra_loads,
    ):
        """
        Apply concentrated forces and moments to a node.

        Can be called with a 6-element list/tuple:
            frame.add_node_load(2, [10.0, 0.0, -5.0, 0.0, 20.0, 0.0])
        or positional floats:
            frame.add_node_load(2, 10.0, 0.0, -5.0, 0.0, 20.0, 0.0)
        """
        node = self.structure.nodes[node_id]
        if isinstance(load, (list, tuple)):
            node.set_load(load)
        elif len(extra_loads) > 0:
            loads = [load] + list(extra_loads)
            node.set_load(loads)
        else:
            node.set_load(load)

    def add_distributed_load(
        self,
        element_id: Union[int, str],
        w1: float = 0.0,
        w2: float = 0.0,
        w3: float = 0.0,
        coord_system: str = "local",
        wx: Optional[float] = None,
        wy: Optional[float] = None,
        wz: Optional[float] = None,
    ) -> DistributedLoad:
        """Apply uniformly distributed load along an element."""
        element = self.structure.elements[element_id]
        dload = DistributedLoad(
            element=element,
            w1=w1,
            w2=w2,
            w3=w3,
            coord_system=coord_system,
            wx=wx,
            wy=wy,
            wz=wz,
        )
        self.structure.add_load(dload)
        return dload

    def add_element_point_load(
        self,
        element_id: Union[int, str],
        p1: float = 0.0,
        p2: float = 0.0,
        p3: float = 0.0,
        m1: float = 0.0,
        m2: float = 0.0,
        m3: float = 0.0,
        x: float = 0.0,
        coord_system: str = "local",
        px: Optional[float] = None,
        py: Optional[float] = None,
        pz: Optional[float] = None,
        mx: Optional[float] = None,
        my: Optional[float] = None,
        mz: Optional[float] = None,
    ) -> ElementPointLoad:
        """Apply concentrated point load acting at distance x along an element."""
        element = self.structure.elements[element_id]
        pload = ElementPointLoad(
            element=element,
            p1=p1,
            p2=p2,
            p3=p3,
            m1=m1,
            m2=m2,
            m3=m3,
            x=x,
            coord_system=coord_system,
            px=px,
            py=py,
            pz=pz,
            mx=mx,
            my=my,
            mz=mz,
        )
        self.structure.add_load(pload)
        return pload

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the linear structural system."""
        return self.structure.solve()

    def draw(self, **kwargs):
        """
        Draw the 3D structure using DrawStructure.

        See DrawStructure.draw for detailed parameters.
        """
        from ..visualization import DrawStructure
        drawer = DrawStructure(self.structure)
        return drawer.draw(**kwargs)

    def plot_mode_shape(self, mode: int = 1, **kwargs):
        """
        Plot a 3D vibration mode shape.
        """
        from ..visualization import DrawStructure
        drawer = DrawStructure(self.structure)
        return drawer.draw_mode_shape(mode=mode, **kwargs)

    def plot_buckling_mode(self, mode: int = 1, **kwargs):
        """
        Plot a 3D elastic buckling mode shape.
        """
        from ..visualization import DrawStructure
        drawer = DrawStructure(self.structure)
        return drawer.draw_buckling_mode(mode=mode, **kwargs)


# Convenient alias
SimpleFrame3D = SimpleFrame

