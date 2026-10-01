"""
Cross-section definitions for 3D finite element analysis.
"""

import math
from typing import Optional


class Section:
    """
    Represents a cross-section of a 3D structural element.

    Moments of Inertia & Local Axes Convention
    -------------------------------------------
    - Local 1-axis is directed along the member centroidal axis (from node i to node j).
    - Local 2-axis and 3-axis lie in the cross-section plane forming a right-handed
      orthogonal system (v1 × v2 = v3):
      - Local 2-axis is the web/depth direction.
      - Local 3-axis is the flange/width direction.
    - I2 is the second moment of area about the local 2-axis:
          I2 = ∫ x3^2 dA
      It resists flexure in the local 1-3 plane (transverse displacement u_3, rotation theta_2).
    - I3 is the second moment of area about the local 3-axis:
          I3 = ∫ x2^2 dA
      It resists flexure in the local 1-2 plane (transverse displacement u_2, rotation theta_3).

    Attributes
    ----------
    A : float
        Cross-sectional area.
    I2 : float
        Second moment of area about local 2-axis (resists bending in local 1-3 plane).
    I3 : float
        Second moment of area about local 3-axis (resists bending in local 1-2 plane).
    J : float
        St. Venant torsional constant about local 1-axis.
    As2 : float, optional
        Effective shear area in the local 2-direction.
    As3 : float, optional
        Effective shear area in the local 3-direction.
    """

    def __init__(
        self,
        A: float,
        I2: float = 0.0,
        I3: float = 0.0,
        J: float = 0.0,
        As2: Optional[float] = None,
        As3: Optional[float] = None,
    ):
        """
        Initialize a 3D Section object.

        Parameters
        ----------
        A : float
            Cross-sectional area.
        I2 : float, optional
            Second moment of area about local 2-axis (resists bending in local 1-3 plane). Defaults to 0.0.
        I3 : float, optional
            Second moment of area about local 3-axis (resists bending in local 1-2 plane). Defaults to 0.0.
        J : float, optional
            Torsional constant about local 1-axis. Defaults to 0.0.
        As2 : float, optional
            Effective shear area along local 2-axis. Defaults to None.
        As3 : float, optional
            Effective shear area along local 3-axis. Defaults to None.
        """
        self.A = float(A)
        self.I2 = float(I2)
        self.I3 = float(I3)
        self.J = float(J)
        self.As2 = float(As2 if As2 is not None else 0.0)
        self.As3 = float(As3 if As3 is not None else 0.0)

    @classmethod
    def from_rectangle(cls, width: float, depth: float) -> "Section":
        """
        Create a rectangular cross-section.

        Local coordinates convention:
        - `depth` (h) is the dimension along the local 2-axis.
        - `width` (b) is the dimension along the local 3-axis.
        - I3 = (width * depth^3) / 12 resists flexure in the local 1-2 plane (about 3-axis).
        - I2 = (depth * width^3) / 12 resists flexure in the local 1-3 plane (about 2-axis).

        Parameters
        ----------
        width : float
            Cross-section dimension along local 3-axis (b).
        depth : float
            Cross-section dimension along local 2-axis (h).
        """
        b = float(width)
        h = float(depth)
        A = b * h
        I3 = (b * h**3) / 12.0
        I2 = (h * b**3) / 12.0

        # Torsional constant J for rectangle:
        # J = a * c^3 * (1/3 - 0.21*(c/a)*(1 - c^4/(12*a^4))) where a >= c
        a = max(b, h)
        c = min(b, h)
        J = a * (c**3) * (1.0 / 3.0 - 0.21 * (c / a) * (1.0 - (c**4) / (12.0 * a**4)))

        # Shear areas for rectangular section: 5/6 * A
        As2 = (5.0 / 6.0) * A
        As3 = (5.0 / 6.0) * A

        return cls(A=A, I2=I2, I3=I3, J=J, As2=As2, As3=As3)

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
        return cls(A=A, I2=I_val, I3=I_val, J=J_val, As2=As_val, As3=As_val)

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
        return cls(A=A, I2=I_val, I3=I_val, J=J_val, As2=As_val, As3=As_val)

    @classmethod
    def from_tube(
        cls, width: float, depth: float, thickness: float
    ) -> "Section":
        """
        Create a hollow rectangular box / structural tube (HSS) cross-section.

        Local coordinates convention:
        - `depth` (h) is the outer dimension along the local 2-axis (web).
        - `width` (b) is the outer dimension along the local 3-axis (flange).
        - I3 = (width * depth^3 - b_i * h_i^3) / 12 resists flexure in local 1-2 plane (about 3-axis).
        - I2 = (depth * width^3 - h_i * b_i^3) / 12 resists flexure in local 1-3 plane (about 2-axis).

        Parameters
        ----------
        width : float
            Outer dimension along local 3-axis (b).
        depth : float
            Outer dimension along local 2-axis (h).
        thickness : float
            Wall thickness (t).
        """
        b = float(width)
        h = float(depth)
        t = float(thickness)
        b_i = b - 2.0 * t
        h_i = h - 2.0 * t
        if b_i <= 0.0 or h_i <= 0.0:
            raise ValueError("Wall thickness is too large for the outer dimensions.")

        A = b * h - b_i * h_i
        I3 = (b * h**3 - b_i * h_i**3) / 12.0
        I2 = (h * b**3 - h_i * b_i**3) / 12.0

        # Bredt's formula for thin-walled single-cell tube: J = 4*A_m^2 / oint(ds/t)
        # where A_m = (b - t)*(h - t)
        A_m = (b - t) * (h - t)
        perimeter_m = 2.0 * ((b - t) + (h - t))
        J = 4.0 * (A_m**2) / (perimeter_m / t)

        return cls(A=A, I2=I2, I3=I3, J=J)

    @classmethod
    def from_i_shape(
        cls, d: float, bf: float, tf: float, tw: float
    ) -> "Section":
        """
        Create an I-beam / W-shape cross-section.

        Local coordinates convention:
        - `d` is the total depth along the local 2-axis (parallel to the web).
        - `bf` is the flange width along the local 3-axis.
        - I3 resists flexure in the web plane 1-2 (bending about local 3-axis).
        - I2 resists flexure across the web plane 1-3 (bending about local 2-axis).

        Parameters
        ----------
        d : float
            Total depth of the section along local 2-axis (parallel to web).
        bf : float
            Flange width along local 3-axis.
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
        # Bending in web plane 1-2 about local 3-axis
        I3 = (bf * d**3 - (bf - tw) * hw**3) / 12.0
        # Bending across web plane 1-3 about local 2-axis
        I2 = 2.0 * (tf * bf**3) / 12.0 + (hw * tw**3) / 12.0
        # Torsion constant J for open thin-walled shape: J = sum(1/3 * b_i * t_i^3)
        J = (2.0 * bf * tf**3 + hw * tw**3) / 3.0
        As2 = d * tw
        As3 = 2.0 * bf * tf * (5.0 / 6.0)

        return cls(A=A, I2=I2, I3=I3, J=J, As2=As2, As3=As3)

    def __repr__(self) -> str:
        return f"Section(A={self.A}, I2={self.I2}, I3={self.I3}, J={self.J})"
