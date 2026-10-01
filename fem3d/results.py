"""
Results module for extracting, post-processing, and reporting 3D structural analysis results.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


class Results:
    """
    Post-processor to extract analysis results (displacements, reactions, member forces)
    into pandas DataFrames and generate comprehensive textbook-style reports.

    Attributes
    ----------
    structure : Structure
        The solved Structure instance.
    """

    def __init__(self, obj):
        """
        Initialize the Results post-processor.

        Parameters
        ----------
        obj : Structure or SimpleFrame
            The analyzed structure or SimpleFrame wrapper.
        """
        if hasattr(obj, "structure") and hasattr(obj.structure, "nodes"):
            self.structure = obj.structure
        else:
            self.structure = obj

    def node_displacements(self) -> pd.DataFrame:
        """
        Return nodal displacements and rotations in a pandas DataFrame.

        Columns:
        - 'node': Node identifier
        - 'ux', 'uy', 'uz': Translational displacements along global X, Y, Z
        - 'rx', 'ry', 'rz': Rotations about global X, Y, Z (radians)
        """
        if self.structure.disp is None:
            self.structure.solve()

        data = []
        for node in self.structure.nodes.values():
            d = self.structure.disp[node.dofs]
            data.append(
                {
                    "node": node.id,
                    "ux": d[0],
                    "uy": d[1],
                    "uz": d[2],
                    "rx": d[3],
                    "ry": d[4],
                    "rz": d[5],
                }
            )
        return pd.DataFrame(data)

    def reactions(self) -> pd.DataFrame:
        """
        Return support reaction forces and moments in a pandas DataFrame.

        Columns:
        - 'node': Node identifier
        - 'Fx', 'Fy', 'Fz': Reaction forces along global X, Y, Z
        - 'Mx', 'My', 'Mz': Reaction moments about global X, Y, Z
        """
        if self.structure.reactions is None:
            self.structure.solve()

        data = []
        for node in self.structure.nodes.values():
            r = self.structure.reactions[node.dofs]
            data.append(
                {
                    "node": node.id,
                    "Fx": r[0],
                    "Fy": r[1],
                    "Fz": r[2],
                    "Mx": r[3],
                    "My": r[4],
                    "Mz": r[5],
                }
            )
        return pd.DataFrame(data)

    def element_forces(self) -> pd.DataFrame:
        """
        Return local end forces and moments for all elements in a pandas DataFrame.

        Columns:
        - 'element': Element identifier
        - 'f1_i', 'f2_i', 'f3_i': Forces at start node i in local coords (1, 2, 3)
        - 'm1_i', 'm2_i', 'm3_i': Moments at start node i in local coords (1, 2, 3)
        - 'f1_j', 'f2_j', 'f3_j': Forces at end node j in local coords (1, 2, 3)
        - 'm1_j', 'm2_j', 'm3_j': Moments at end node j in local coords (1, 2, 3)
        - 'P_i', 'V2_i', 'V3_i', 'T_i', 'M2_i', 'M3_i'
        - 'P_j', 'V2_j', 'V3_j', 'T_j', 'M2_j', 'M3_j'
        """
        if self.structure.disp is None:
            self.structure.solve()

        data = []
        for el in self.structure.elements.values():
            f = el.get_local_forces()
            data.append(
                {
                    "element": el.id,
                    "f1_i": f[0],
                    "f2_i": f[1],
                    "f3_i": f[2],
                    "m1_i": f[3],
                    "m2_i": f[4],
                    "m3_i": f[5],
                    "f1_j": f[6],
                    "f2_j": f[7],
                    "f3_j": f[8],
                    "m1_j": f[9],
                    "m2_j": f[10],
                    "m3_j": f[11],
                    # Standard engineering nomenclature
                    "P_i": f[0],
                    "V2_i": f[1],
                    "V3_i": f[2],
                    "T_i": f[3],
                    "M2_i": f[4],
                    "M3_i": f[5],
                    "P_j": f[6],
                    "V2_j": f[7],
                    "V3_j": f[8],
                    "T_j": f[9],
                    "M2_j": f[10],
                    "M3_j": f[11],
                }
            )
        return pd.DataFrame(data)

    @staticmethod
    def _format_array(array: Optional[np.ndarray]) -> str:
        """Format a numpy array nicely."""
        if array is None:
            return "Not computed"
        return np.array2string(np.asarray(array), precision=6, suppress_small=False)

    def create_report(self, print_report: bool = True) -> str:
        """
        Create a detailed structural analysis report for classroom and engineering verification.

        Parameters
        ----------
        print_report : bool, optional
            Whether to print to stdout. Defaults to True.

        Returns
        -------
        str
            Full report text.
        """
        if self.structure.disp is None:
            self.structure.solve()

        lines = []
        lines.append("3D Structural Analysis Report (fem3d)")
        lines.append("=" * 80)
        lines.append(f"Nodes: {len(self.structure.nodes)}")
        lines.append(f"Elements: {len(self.structure.elements)}")
        lines.append(f"Total DOFs: {self.structure.neq}")
        lines.append(f"Fixed DOFs ({len(self.structure.fixed_dofs)}): {self.structure.fixed_dofs}")
        lines.append(f"Free DOFs ({len(self.structure.free_dofs)}): {self.structure.free_dofs}")
        lines.append("")

        lines.append("Node Support & Load Conditions")
        lines.append("-" * 80)
        for node in self.structure.nodes.values():
            lines.append(
                f"Node {node.id}: coords=({node.x:.3f}, {node.y:.3f}, {node.z:.3f}), "
                f"support={node.support}, dofs={node.dofs}, load={node.load}"
            )
        lines.append("")

        lines.append("Element Formulations & Reports")
        lines.append("-" * 80)
        for el in self.structure.elements.values():
            lines.append(
                f"Element {el.id} ({type(el).__name__}): nodes=({el.node_i.id}, {el.node_j.id}), L={el.length:.4f}"
            )
            lines.append(f"  DOFs: {el.node_i.dofs + el.node_j.dofs}")
            lines.append(f"  Orientation v1: {np.round(el.v1, 4)}")
            lines.append(f"  Orientation v2: {np.round(el.v2, 4)}")
            lines.append(f"  Orientation v3: {np.round(el.v3, 4)}")
            lines.append("  Local Stiffness Matrix (12x12):")
            lines.append(self._format_array(el.local_stiffness()))
            lines.append("  Transformation Matrix (12x12):")
            lines.append(self._format_array(el.transformation_matrix()))
            lines.append("  Global Stiffness Matrix (12x12):")
            lines.append(self._format_array(el.global_stiffness()))
            lines.append("")

        lines.append("Global Structure Stiffness Matrix K")
        lines.append("-" * 80)
        lines.append(self._format_array(self.structure.K))
        lines.append("")

        lines.append("Global Structure Load Vector F")
        lines.append("-" * 80)
        lines.append(self._format_array(self.structure.F))
        lines.append("")

        lines.append("Displacements [ux, uy, uz, rx, ry, rz]")
        lines.append("-" * 80)
        disp_df = self.node_displacements()
        lines.append(disp_df.to_string(index=False))
        lines.append("")

        lines.append("Reactions [Fx, Fy, Fz, Mx, My, Mz]")
        lines.append("-" * 80)
        reac_df = self.reactions()
        lines.append(reac_df.to_string(index=False))
        lines.append("")

        lines.append("Element Local End Forces")
        lines.append("-" * 80)
        forces_df = self.element_forces()
        lines.append(forces_df.to_string(index=False))

        report = "\n".join(lines)
        if print_report:
            print(report)
        return report
