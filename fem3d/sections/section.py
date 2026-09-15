"""
Cross-section definitions for 3D finite element analysis.
"""

import math
from typing import Optional


class Section:
    """
    Represents a cross-section of a 3D structural element.

    Attributes
    ----------
    A : float
        Cross-sectional area.
    Iy : float
        Second moment of area about the local y'-axis (weak or minor axis).
    Iz : float
        Second moment of area about the local z'-axis (strong or major axis).
    J : float
        St. Venant torsional constant.
    Asy : float, optional
        Effective shear area in the local y'-direction.
    Asz : float, optional
        Effective shear area in the local z'-direction.
    """

    def __init__(
        self,
        A: float,
        Iy: float = 0.0,
        Iz: float = 0.0,
        J: float = 0.0,
        Asy: Optional[float] = None,
        Asz: Optional[float] = None,
    ):
        """
        Initialize a 3D Section object.

        Parameters
        ----------
        A : float
            Cross-sectional area.
        Iy : float, optional
            Second moment of area about local y'-axis. Defaults to 0.0.
        Iz : float, optional
            Second moment of area about local z'-axis. Defaults to 0.0.
        J : float, optional
            Torsional constant. Defaults to 0.0.
        Asy : float, optional
            Shear area along local y'. Defaults to None.
        Asz : float, optional
            Shear area along local z'. Defaults to None.
        """
        self.A = float(A)
        self.Iy = float(Iy)
        self.Iz = float(Iz)
        self.J = float(J)
        self.Asy = float(Asy) if Asy is not None else 0.0
        self.Asz = float(Asz) if Asz is not None else 0.0

    @property
    def I(self) -> float:
        """Alias for strong-axis moment of inertia Iz (2D compatibility)."""
        return self.Iz

    @classmethod
    def from_rectangle(cls, width: float, depth: float) -> "Section":
        """
        Create a rectangular cross-section.

        Local coordinates: width along local y', depth along local z'.
        Iz is strong axis (about z'), Iy is weak axis (about y').

        Parameters
        ----------
        width : float
            Dimension along local y' (b).
        depth : float
            Dimension along local z' (h).
        """
        b = float(width)
        h = float(depth)
        A = b * h
        Iz = (b * h**3) / 12.0
        Iy = (h * b**3) / 12.0

        # Torsional constant J for rectangle:
        # J = a * c^3 * (1/3 - 0.21*(c/a)*(1 - c^4/(12*a^4))) where a >= c
        a = max(b, h)
        c = min(b, h)
        J = a * (c**3) * (1.0 / 3.0 - 0.21 * (c / a) * (1.0 - (c**4) / (12.0 * a**4)))

        # Shear areas for rectangular section: 5/6 * A
        Asy = (5.0 / 6.0) * A
        Asz = (5.0 / 6.0) * A

        return cls(A=A, Iy=Iy, Iz=Iz, J=J, Asy=Asy, Asz=Asz)

    @classmethod
    def from_circle(cls, diameter: float) -> "Section":
        """
        Create a solid circular cross-section.

        Parameters
        ----------
        diameter : float
            Diameter of the circular section.
        """
        d = float(diameter)
        r = d / 2.0
        A = math.pi * r**2
        I_val = math.pi * (d**4) / 64.0
        J_val = math.pi * (d**4) / 32.0
        # Shear area for solid circle: 9/10 * A
        As_val = 0.9 * A
        return cls(A=A, Iy=I_val, Iz=I_val, J=J_val, Asy=As_val, Asz=As_val)

    @classmethod
    def from_pipe(cls, outer_d: float, thickness: float) -> "Section":
        """
        Create a hollow circular pipe cross-section.

        Parameters
        ----------
        outer_d : float
            Outer diameter.
        thickness : float
            Wall thickness.
        """
        d_o = float(outer_d)
        t = float(thickness)
        d_i = d_o - 2.0 * t
        if d_i <= 0.0:
            raise ValueError("Thickness must be less than half the outer diameter.")

        A = math.pi * (d_o**2 - d_i**2) / 4.0
        I_val = math.pi * (d_o**4 - d_i**4) / 64.0
        J_val = math.pi * (d_o**4 - d_i**4) / 32.0
        As_val = 0.5 * A
        return cls(A=A, Iy=I_val, Iz=I_val, J=J_val, Asy=As_val, Asz=As_val)

    @classmethod
    def from_tube(
        cls, width: float, depth: float, thickness: float
    ) -> "Section":
        """
        Create a hollow rectangular box / structural tube (HSS) cross-section.

        Parameters
        ----------
        width : float
            Outer dimension along local y'.
        depth : float
            Outer dimension along local z'.
        thickness : float
            Wall thickness.
        """
        b = float(width)
        h = float(depth)
        t = float(thickness)
        b_i = b - 2.0 * t
        h_i = h - 2.0 * t
        if b_i <= 0.0 or h_i <= 0.0:
            raise ValueError("Wall thickness is too large for the outer dimensions.")

        A = b * h - b_i * h_i
        Iz = (b * h**3 - b_i * h_i**3) / 12.0
        Iy = (h * b**3 - h_i * b_i**3) / 12.0

        # Bredt's formula for thin-walled single-cell tube: J = 4*A_m^2 / oint(ds/t)
        # where A_m = (b - t)*(h - t)
        A_m = (b - t) * (h - t)
        perimeter_m = 2.0 * ((b - t) + (h - t))
        J = 4.0 * (A_m**2) / (perimeter_m / t)

        return cls(A=A, Iy=Iy, Iz=Iz, J=J)

    @classmethod
    def from_i_shape(
        cls, d: float, bf: float, tf: float, tw: float
    ) -> "Section":
        """
        Create an I-beam / W-shape cross-section.

        Parameters
        ----------
        d : float
            Total depth of the section along local z'.
        bf : float
            Flange width along local y'.
        tf : float
            Flange thickness.
        tw : float
            Web thickness.
        """
        d = float(d)
        bf = float(bf)
        tf = float(tf)
        tw = float(tw)
        hw = d - 2.0 * tf

        A = 2.0 * bf * tf + hw * tw
        # Major axis bending Iz (about local z')
        Iz = (bf * d**3 - (bf - tw) * hw**3) / 12.0
        # Minor axis bending Iy (about local y')
        Iy = 2.0 * (tf * bf**3) / 12.0 + (hw * tw**3) / 12.0
        # Torsion constant J for open thin-walled shape: J = sum(1/3 * b_i * t_i^3)
        J = (2.0 * bf * tf**3 + hw * tw**3) / 3.0
        Asy = 2.0 * bf * tf * (5.0 / 6.0)
        Asz = d * tw

        return cls(A=A, Iy=Iy, Iz=Iz, J=J, Asy=Asy, Asz=Asz)

    def __repr__(self) -> str:
        return f"Section(A={self.A}, Iy={self.Iy}, Iz={self.Iz}, J={self.J})"
