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
        Iy: Optional[float] = None,
        Iz: Optional[float] = None,
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
            Start node identifier.
        node_j_id : int or str
            End node identifier.
        E : float
            Young's modulus.
        A : float
            Cross-sectional area.
        Iy : float, optional
            Weak-axis moment of inertia. If omitted and Iz is given, set equal to Iz.
        Iz : float, optional
            Strong-axis moment of inertia. If omitted and Iy is given, set equal to Iy.
        J : float, optional
            Torsional constant. If omitted, approximated as Iy + Iz.
        roll_angle : float, optional
            Roll angle in degrees about element axis. Defaults to 0.0.
        web_vector : iterable, optional
            Reference orientation vector. Defaults to None.
        nu : float, optional
            Poisson's ratio. Defaults to 0.3.
        G : float, optional
            Shear modulus. If None, derived as E / (2*(1+nu)).
        extra_mass : float, optional
            Distributed non-structural mass. Defaults to 0.0.

        Returns
        -------
        FrameElement
            The created FrameElement instance.
        """
        node_i = self.structure.nodes[node_i_id]
        node_j = self.structure.nodes[node_j_id]

        # Handle flexibility in moment of inertia arguments
        if Iz is None and Iy is not None:
            Iz_val = float(Iy)
            Iy_val = float(Iy)
        elif Iy is None and Iz is not None:
            Iy_val = float(Iz)
            Iz_val = float(Iz)
        elif Iy is None and Iz is None:
            # Default zero or caller provided Section
            Iy_val = 0.0
            Iz_val = 0.0
        else:
            Iy_val = float(Iy)
            Iz_val = float(Iz)

        if J is None:
            J_val = Iy_val + Iz_val if (Iy_val + Iz_val) > 0 else 0.0
        else:
            J_val = float(J)

        mat = ElasticMaterial(E=E, nu=nu, G=G)
        sec = Section(A=A, Iy=Iy_val, Iz=Iz_val, J=J_val)

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
        wx: float = 0.0,
        wy: float = 0.0,
        wz: float = 0.0,
        coord_system: str = "local",
    ) -> DistributedLoad:
        """Apply uniformly distributed load along an element."""
        element = self.structure.elements[element_id]
        dload = DistributedLoad(
            element=element,
            wx=wx,
            wy=wy,
            wz=wz,
            coord_system=coord_system,
        )
        self.structure.add_load(dload)
        return dload

    def add_element_point_load(
        self,
        element_id: Union[int, str],
        px: float = 0.0,
        py: float = 0.0,
        pz: float = 0.0,
        mx: float = 0.0,
        my: float = 0.0,
        mz: float = 0.0,
        x: float = 0.0,
        coord_system: str = "local",
    ) -> ElementPointLoad:
        """Apply concentrated point load acting at distance x along an element."""
        element = self.structure.elements[element_id]
        pload = ElementPointLoad(
            element=element,
            px=px,
            py=py,
            pz=pz,
            mx=mx,
            my=my,
            mz=mz,
            x=x,
            coord_system=coord_system,
        )
        self.structure.add_load(pload)
        return pload

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the linear structural system."""
        return self.structure.solve()


# Convenient alias
SimpleFrame3D = SimpleFrame
