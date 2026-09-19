"""
Nodes module representing 3D spatial points in the structural model.
"""

from typing import Iterable, Union


class Node:
    """
    Represents a node in a 3D finite element model.

    Each node possesses 6 degrees of freedom in global coordinates:
    [ux, uy, uz, rx, ry, rz]
    where:
    - ux, uy, uz are translations along global X, Y, Z.
    - rx, ry, rz are rotations about global X, Y, Z (right-hand rule).

    Attributes
    ----------
    id : int or str
        Unique identifier of the node.
    x : float
        Global X-coordinate of the node.
    y : float
        Global Y-coordinate of the node.
    z : float
        Global Z-coordinate of the node.
    support : list of bool
        Fixity condition of the node's degrees of freedom:
        [ux, uy, uz, rx, ry, rz].
    load : list of float
        Nodal concentrated force and moment values:
        [Fx, Fy, Fz, Mx, My, Mz].
    dofs : list of int or None
        Global degrees of freedom indices assigned to this node (length 6).
    mass : list of float
        Translational lumped mass [mx, my, mz].
    inertia : list of float
        Rotational lumped inertia [Ix, Iy, Iz].
    """

    def __init__(self, nid: Union[int, str], x: float, y: float, z: float = 0.0):
        """
        Initialize a 3D Node object.

        Parameters
        ----------
        nid : int or str
            Unique identifier of the node.
        x : float
            X-coordinate.
        y : float
            Y-coordinate.
        z : float, optional
            Z-coordinate. Defaults to 0.0.
        """
        self.id = nid
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        # 6 DOFs: [ux, uy, uz, rx, ry, rz]
        self.support = [False, False, False, False, False, False]
        self.load = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.dofs = None  # Assigned by Structure.number_dofs()

        # Lumped mass [mx, my, mz] and rotational inertia [Ix, Iy, Iz]
        self.mass = [0.0, 0.0, 0.0]
        self.inertia = [0.0, 0.0, 0.0]

    def set_support(
        self,
        ux_fixed: Union[bool, int, Iterable] = False,
        uy_fixed: Union[bool, int] = False,
        uz_fixed: Union[bool, int] = False,
        rx_fixed: Union[bool, int] = False,
        ry_fixed: Union[bool, int] = False,
        rz_fixed: Union[bool, int] = False,
    ):
        """
        Set support fixity conditions for the 6 degrees of freedom.

        Can be called with keyword flags::

            node.set_support(ux_fixed=True, uy_fixed=True, uz_fixed=True)

        or with a 6-element iterable/sequence `(ux, uy, uz, rx, ry, rz)`::

            node.set_support([1, 1, 1, 1, 1, 1])
            node.set_support([1, 1, 1, 0, 0, 0])  # Pinned base in 3D

        or with 6 positional arguments::

            node.set_support(1, 1, 1, 0, 0, 0)
        """
        if isinstance(ux_fixed, (list, tuple)):
            flags = list(ux_fixed)
            if len(flags) == 3:
                # In 3D, 3 flags represent the three translational DOFs [ux, uy, uz] (e.g. 3D pin/ball joint)
                flags = [flags[0], flags[1], flags[2], False, False, False]
            elif len(flags) != 6:
                raise ValueError(
                    f"Support fixity list must have 6 elements [ux, uy, uz, rx, ry, rz], got {len(flags)}."
                )
            self.support = [bool(f) for f in flags]
        else:
            self.support = [
                bool(ux_fixed),
                bool(uy_fixed),
                bool(uz_fixed),
                bool(rx_fixed),
                bool(ry_fixed),
                bool(rz_fixed),
            ]

    def set_load(
        self,
        fx: Union[float, Iterable] = 0.0,
        fy: float = 0.0,
        fz: float = 0.0,
        mx: float = 0.0,
        my: float = 0.0,
        mz: float = 0.0,
    ):
        """
        Apply concentrated forces and moments directly to the node.

        Can be called with 6 positional/keyword floats or a 6-element iterable.

        Parameters
        ----------
        fx : float or iterable
            Force in global X-direction (or [fx, fy, fz, mx, my, mz] list).
        fy : float, optional
            Force in global Y-direction. Defaults to 0.0.
        fz : float, optional
            Force in global Z-direction. Defaults to 0.0.
        mx : float, optional
            Moment about global X-axis. Defaults to 0.0.
        my : float, optional
            Moment about global Y-axis. Defaults to 0.0.
        mz : float, optional
            Moment about global Z-axis. Defaults to 0.0.
        """
        if isinstance(fx, (list, tuple)):
            vals = list(fx)
            if len(vals) == 3:
                # Legacy 2D [Fx, Fy, Mz]
                vals = [vals[0], vals[1], 0.0, 0.0, 0.0, vals[2]]
            elif len(vals) != 6:
                raise ValueError(
                    f"Load list must have 6 elements [fx, fy, fz, mx, my, mz], got {len(vals)}."
                )
            self.load = [float(v) for v in vals]
        else:
            self.load = [
                float(fx),
                float(fy),
                float(fz),
                float(mx),
                float(my),
                float(mz),
            ]

    def set_mass(
        self,
        mass: Union[float, Iterable] = 0.0,
        inertia: Union[float, Iterable] = 0.0,
    ):
        """
        Set translational lumped mass and rotational lumped inertia.

        Parameters
        ----------
        mass : float or iterable of 3 floats
            Translational lumped mass. If scalar, applied equally to mx, my, mz.
        inertia : float or iterable of 3 floats
            Rotational lumped inertia. If scalar, applied equally to Ix, Iy, Iz.
        """
        if isinstance(mass, (list, tuple)):
            self.mass = [float(m) for m in mass]
        else:
            m = float(mass)
            self.mass = [m, m, m]

        if isinstance(inertia, (list, tuple)):
            self.inertia = [float(i) for i in inertia]
        else:
            it = float(inertia)
            self.inertia = [it, it, it]

    def __repr__(self) -> str:
        return f"Node(id={self.id}, x={self.x}, y={self.y}, z={self.z})"
