"""
Unit tests for 3D Modal Analysis and Linear Elastic Buckling Analysis.
"""

import math
import numpy as np
import pytest

from fem3d import SimpleFrame, Structure, Node, ElasticMaterial, Section, FrameElement, buckling_analysis


def test_modal_analysis_cantilever():
    """
    Test modal analysis for 3D cantilever beam.
    Theoretical fundamental bending frequency (Euler-Bernoulli):
    f_1 = (1.8751^2 / (2 * pi * L^2)) * sqrt(E * I / (rho * A))
    """
    struct = Structure()
    n1 = Node(1, 0.0, 0.0, 0.0)
    n2 = Node(2, 120.0, 0.0, 0.0)
    n1.set_support([1, 1, 1, 1, 1, 1])

    struct.add_node(n1)
    struct.add_node(n2)

    E = 30000.0   # ksi
    rho = 0.00073 # kips*s^2 / in^4 (steel density ~ 490 pcf = 0.2836 lb/in^3 / 386.4)
    mat = ElasticMaterial(E=E, rho=rho)

    A = 10.0
    Iy = 80.0
    Iz = 200.0
    J = 50.0
    sec = Section(A=A, Iy=Iy, Iz=Iz, J=J)

    elem = FrameElement(1, n1, n2, material=mat, section=sec)
    struct.add_element(elem)

    # Free tip dummy solve to initialize DOFs
    struct.solve()

    modal = struct.modal_analysis(num_modes=4)
    frequencies = modal["frequencies"]
    assert len(frequencies) > 0
    assert frequencies[0] > 0.0

    # Theoretical frequency for weak axis bending (Iy = 80):
    L = 120.0
    f_theory_y = ((1.875104**2) / (2.0 * math.pi * (L**2))) * math.sqrt((E * Iy) / (rho * A))

    # Our single-element discretization with consistent mass matches within a few percent
    # (typically within ~5% for a single cubic beam element)
    rel_diff = abs(frequencies[0] - f_theory_y) / f_theory_y
    assert rel_diff < 0.10


def test_buckling_analysis_cantilever():
    """
    Test linear elastic buckling of a 3D cantilever column:
    Theoretical Euler buckling load: P_cr = pi^2 * E * I / (4 * L^2)
    """
    struct = Structure()
    n1 = Node(1, 0.0, 0.0, 0.0)
    n2 = Node(2, 0.0, 0.0, 100.0)  # vertical column along Z
    n1.set_support([1, 1, 1, 1, 1, 1])

    struct.add_node(n1)
    struct.add_node(n2)

    E = 29000.0
    A = 10.0
    Iy = 50.0   # weak axis
    Iz = 150.0  # strong axis
    J = 30.0
    sec = Section(A=A, Iy=Iy, Iz=Iz, J=J)
    mat = ElasticMaterial(E=E)

    elem = FrameElement(1, n1, n2, material=mat, section=sec)
    struct.add_element(elem)

    # Reference axial compressive load at top (downward in -Z direction)
    P_ref = 10.0
    n2.set_load(fz=-P_ref)

    struct.solve()

    factors, modes = buckling_analysis(struct, num_modes=1)
    P_cr_fem = factors[0] * P_ref

    # Theoretical Euler buckling load for weak axis (Iy):
    L = 100.0
    P_cr_exact = (math.pi**2) * E * Iy / (4.0 * (L**2))

    # Single-element cubic formulation approximates Euler load within ~3%
    rel_error = abs(P_cr_fem - P_cr_exact) / P_cr_exact
    assert rel_error < 0.05
