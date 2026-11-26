import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "applications/scripts/cone_grouping/batch_cone_grouping.py"
)


def quarter_turns():
    import numpy as np

    quarter = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return np.stack([np.eye(3), quarter, np.diag([-1, -1, 1]), quarter.T])


class ConeGroupingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("numpy") is None:
            raise unittest.SkipTest("Cone grouping tests require NumPy")

    def test_nearest_resolves_symmetry_mirrors_and_ties(self):
        import numpy as np
        from xmippPyModules.coneGroupingTools.cone_grouping import group_projection_directions

        x, y, z = np.eye(3)
        directions = np.array([y, -x, z])
        references = np.array([x, x, z])
        symmetries = quarter_turns()

        for consider_mirrors, expected_symmetries in (
            (False, [1, 2, 0]),
            (True, [1, 0, 0]),
        ):
            with self.subTest(consider_mirrors=consider_mirrors):
                result = group_projection_directions(
                    directions,
                    references,
                    symmetries,
                    consider_mirrors=consider_mirrors,
                    batch_size=2,
                )
                np.testing.assert_array_equal(result.particle_indices, [0, 1, 2])
                np.testing.assert_array_equal(result.group_indices, [0, 0, 2])
                np.testing.assert_array_equal(
                    result.symmetry_indices, expected_symmetries
                )
                np.testing.assert_allclose(result.angular_distances, 0, atol=1e-12)
                self.assertFalse(result.fallback.any())

    def test_threshold_preserves_overlap_and_flags_only_uncovered_particles(self):
        import numpy as np
        from xmippPyModules.coneGroupingTools.cone_grouping import group_projection_directions

        x, y, z = np.eye(3)
        args = (
            np.array([(x + y) / np.sqrt(2), -x, z]),
            np.array([x, y]),
            np.eye(3)[None, ...],
        )
        options = dict(mode="threshold", angular_radius=np.deg2rad(50), batch_size=2)

        with self.assertRaisesRegex(ValueError, "1/3 particles.*90"):
            group_projection_directions(*args, **options)

        result = group_projection_directions(
            *args, **options, unassigned_policy="nearest"
        )
        np.testing.assert_array_equal(result.particle_indices, [0, 0, 1, 2])
        np.testing.assert_array_equal(result.group_indices, [0, 1, 0, 0])
        np.testing.assert_allclose(np.rad2deg(result.angular_distances), [45, 45, 0, 90])
        np.testing.assert_array_equal(result.fallback, [False, False, False, True])

    def test_deduplication_uses_symmetry_and_only_kept_references(self):
        import numpy as np
        from xmippPyModules.coneGroupingTools.cone_grouping import deduplicate_with_symmetry

        # With C2 these directions lie at 0, 5 and 10 degrees in the same sector.
        # The discarded middle direction must leave the final direction alone.
        azimuths = np.deg2rad([0, 185, 190])
        directions = np.column_stack(
            [np.cos(azimuths), np.sin(azimuths), np.zeros_like(azimuths)]
        )
        keep = deduplicate_with_symmetry(
            directions,
            quarter_turns()[::2],
            angular_distance_threshold=np.deg2rad(7),
            batch_size=1,
            consider_mirrors=False,
        )
        np.testing.assert_array_equal(keep, [True, False, True])

    def test_alignment_transforms_each_assignment_from_the_original_particle(self):
        import numpy as np
        from xmippPyModules.coneGroupingTools.cone_grouping import (
            Assignments,
            EulerAngles,
            align_assignments,
            euler_zyz_to_matrix,
        )

        particle_matrices = euler_zyz_to_matrix(
            EulerAngles(
                rot=np.zeros(3),
                tilt=np.deg2rad([0, 180, 0]),
                psi=np.deg2rad([90, 0, 90]),
            )
        )
        reference_matrices = euler_zyz_to_matrix(
            EulerAngles(np.zeros(2), np.zeros(2), np.array([0, np.pi / 2]))
        )
        assignments = Assignments(
            particle_indices=np.array([0, 0, 1, 2]),
            group_indices=np.array([0, 1, 0, 0]),
            symmetry_indices=np.array([0, 0, 0, 1]),
            angular_distances=np.zeros(4),
            fallback=np.zeros(4, dtype=bool),
        )

        psi, flip, shifts = align_assignments(
            particle_matrices,
            np.tile([2.0, 3.0], (3, 1)),
            reference_matrices,
            assignments,
            quarter_turns(),
            batch_size=3,
        )
        np.testing.assert_allclose(np.rad2deg(psi), [-90, 0, 0, 0], atol=1e-12)
        np.testing.assert_array_equal(flip, [False, False, True, False])
        np.testing.assert_allclose(shifts, [[-3, 2], [2, 3], [-2, 3], [2, 3]])


class XmippConeSymmetryTest(unittest.TestCase):
    def test_symmetries_are_identity_first_closed_rotation_groups(self):
        import numpy as np
        from xmippPyModules.coneGroupingTools.symmetry import generate_symmetries

        groups = {
            None: 1, " c4 ": 4, "D1": 2, "D3": 6, "T": 12, "O": 24,
            "I": 60, "I1": 60, "I2": 60, "I3": 60, "I4": 60,
        }
        identity = np.eye(3)
        for symbol, count in groups.items():
            with self.subTest(symbol=symbol):
                matrices = generate_symmetries(symbol)
                self.assertEqual(matrices.shape, (count, 3, 3))
                np.testing.assert_array_equal(matrices[0], identity)
                np.testing.assert_allclose(
                    matrices @ matrices.transpose(0, 2, 1),
                    np.broadcast_to(identity, matrices.shape),
                    atol=1e-12,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    np.linalg.det(matrices), 1, atol=1e-12, rtol=0
                )
                for rotation in matrices:
                    products = rotation @ matrices
                    distances = np.max(
                        np.abs(products[:, None] - matrices[None, :]), axis=(2, 3)
                    )
                    self.assertTrue((distances.min(axis=1) < 1e-12).all())

        np.testing.assert_array_equal(generate_symmetries("I"), generate_symmetries("I2"))
        np.testing.assert_allclose(
            generate_symmetries("D1")[1], np.diag([1, -1, -1]), atol=1e-6
        )

    def test_polyhedral_groups_match_xmipp_coordinate_conventions(self):
        if importlib.util.find_spec("xmippLib") is None:
            self.skipTest("Convention comparison requires the Xmipp Python binding")
        import numpy as np
        import xmippLib
        from xmippPyModules.coneGroupingTools.symmetry import generate_symmetries

        with tempfile.TemporaryDirectory() as tmpdir:
            tetrahedral = Path(tmpdir) / "tetrahedral.sym"
            tetrahedral.write_text(
                "rot_axis 3 0 0 1\n"
                "rot_axis 2 0 0.816496580927726 0.577350269189626\n"
            )
            for symbol in ("T", "O", "I1", "I2", "I3", "I4"):
                with self.subTest(symbol=symbol):
                    definition = str(tetrahedral) if symbol == "T" else symbol
                    expected = np.asarray(
                        xmippLib.SymList().getSymmetryMatrices(definition)
                    )
                    rotations = generate_symmetries(symbol)
                    self.assertEqual(rotations.shape, expected.shape)
                    distances = np.max(
                        np.abs(rotations[:, None] - expected[None, :]), axis=(2, 3)
                    )
                    np.testing.assert_array_less(distances.min(axis=0), 1e-6)
                    np.testing.assert_array_less(distances.min(axis=1), 1e-6)

    def test_tetrahedral_match_recovers_the_reference_orientation(self):
        import numpy as np
        from xmippPyModules.coneGroupingTools.cone_grouping import (
            EulerAngles,
            align_assignments,
            euler_zyz_to_matrix,
            group_projection_directions,
        )
        from xmippPyModules.coneGroupingTools.symmetry import generate_symmetries

        reference = euler_zyz_to_matrix(EulerAngles(*np.deg2rad([31, 43, -17])))
        # In Xmipp's convention T has a threefold axis along Z.
        sine = np.sqrt(3) / 2
        threefold = np.array([[-0.5, -sine, 0], [sine, -0.5, 0], [0, 0, 1]])
        particle = reference @ threefold.T
        symmetries = generate_symmetries("T")
        result = group_projection_directions(
            particle[None, 2, :], reference[None, 2, :], symmetries,
            consider_mirrors=False,
        )
        selected = symmetries[result.symmetry_indices[0]]
        np.testing.assert_allclose(selected, threefold, atol=1e-6)
        np.testing.assert_allclose(particle @ selected, reference, atol=1e-6)

        shifts = np.array([[2.0, 3.0]])
        psi, flip, aligned_shifts = align_assignments(
            particle[None], shifts, reference[None], result, symmetries
        )
        np.testing.assert_allclose(psi, 0, atol=1e-6)
        self.assertFalse(flip.any())
        np.testing.assert_allclose(aligned_shifts, shifts, atol=1e-6)

    def test_vectorized_euler_conversion_matches_xmipp(self):
        if importlib.util.find_spec("xmippLib") is None:
            self.skipTest("Euler comparison requires the Xmipp Python binding")
        import numpy as np
        import xmippLib
        from xmippPyModules.coneGroupingTools.cone_grouping import (
            EulerAngles,
            euler_zyz_to_matrix,
        )

        degrees = np.array([[31, 43, -17], [-73, 0, 32], [21, 180, 110]])
        expected = np.stack([xmippLib.Euler_angles2matrix(*row) for row in degrees])
        actual = euler_zyz_to_matrix(EulerAngles(*np.deg2rad(degrees).T))
        np.testing.assert_allclose(actual, expected, atol=1e-12)


class ConeGroupingScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [
            name for name in ("numpy", "pandas", "starfile")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise unittest.SkipTest(f"Cone grouping CLI requires {', '.join(missing)}")

    def test_cli_preserves_particle_ids_and_counts_overlapping_memberships(self):
        import numpy as np
        import pandas as pd
        import starfile

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            source = directory / "particles.xmd"
            grouped = directory / "grouped.star"
            references = directory / "references.star"
            # With three references, the 45-degree particle matches the
            # equatorial and polar cones. The middle cone stays empty.
            starfile.write(
                pd.DataFrame(
                    {
                        "itemId": [70, 3],
                        "image": ["2@particles.mrcs", "1@particles.mrcs"],
                        "angleRot": [0, 0],
                        "angleTilt": [45, 90],
                        "anglePsi": [0, 90],
                        "shiftX": [0, 2],
                        "shiftY": [0, 3],
                    }
                ),
                source,
            )

            process = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--input-xmd", str(source),
                    "--out-star", str(grouped),
                    "--out-reference-md", str(references),
                    "--n-groups", "3",
                    "--grouping-batch-size", "1",
                    "--grouping-mode", "threshold",
                    "--angular-radius-deg", "50",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            assigned = starfile.read(grouped)
            cones = starfile.read(references)

        self.assertEqual(assigned["itemId"].tolist(), [70, 70, 3])
        self.assertEqual(
            assigned["image"].tolist(),
            ["2@particles.mrcs", "2@particles.mrcs", "1@particles.mrcs"],
        )
        self.assertEqual(assigned["cone_group"].tolist(), [1, 3, 1])
        self.assertEqual(assigned["assignment_fallback"].tolist(), [0, 0, 0])
        np.testing.assert_allclose(assigned["angular_distance_deg"], [45, 45, 0])
        np.testing.assert_allclose(
            assigned.loc[2, ["anglePsi", "flip", "shiftX", "shiftY"]].to_numpy(dtype=float),
            [-90, 0, -3, 2],
        )
        self.assertEqual(cones["cone_group"].tolist(), [1, 2, 3])
        self.assertEqual(cones["n_particles"].tolist(), [2, 0, 1])


class XmippConeGeometryTest(unittest.TestCase):
    def test_native_geometry_applies_cone_rotation_shift_and_flip(self):
        program = shutil.which("xmipp_transform_geometry")
        if program is None or importlib.util.find_spec("mrcfile") is None:
            self.skipTest("Cone geometry test requires xmipp_transform_geometry and mrcfile")

        import mrcfile
        import numpy as np
        from xmippPyModules.coneGroupingTools.cone_grouping import (
            EulerAngles,
            align_assignments,
            euler_zyz_to_matrix,
            group_projection_directions,
        )

        particles = euler_zyz_to_matrix(
            EulerAngles(np.zeros(2), np.deg2rad([0, 180]), np.deg2rad([90, 0]))
        )
        reference = np.eye(3)[None, ...]
        assignments = group_projection_directions(
            particles[:, 2, :], reference[:, 2, :], reference
        )
        psi, flip, shifts = align_assignments(
            particles, np.tile([2., 3.], (2, 1)), reference, assignments, reference
        )
        marker = np.zeros((9, 9), dtype=np.float32)
        marker[4, 5] = 2
        marker[3, 3] = 7
        expected = np.zeros((2, 9, 9), dtype=np.float32)
        expected[:, 7, 1] = 2
        expected[0, 5, 2] = 7
        expected[1, 6, 3] = 7

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images = root / "input.mrcs"
            metadata = root / "grouped.xmd"
            output = root / "aligned.mrcs"
            mrcfile.write(images, np.stack([marker, marker]))
            rows = [
                f"{index}@{images} {angle} {int(mirror)} {sx} {sy}"
                for index, (angle, mirror, (sx, sy)) in enumerate(
                    zip(np.rad2deg(psi), flip, shifts), start=1
                )
            ]
            metadata.write_text(
                "# XMIPP_STAR_1 *\ndata_particles\nloop_\n"
                " _image\n _anglePsi\n _flip\n _shiftX\n _shiftY\n"
                + "\n".join(rows) + "\n"
            )
            process = subprocess.run(
                [program, "-i", str(metadata), "-o", str(output),
                 "--apply_transform", "--interp", "linear"],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with mrcfile.open(output) as aligned:
                np.testing.assert_allclose(aligned.data, expected, atol=1e-6)
