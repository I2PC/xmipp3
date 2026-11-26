import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "applications/scripts/cone_grouping/batch_cone_grouping.py"
)


class ConeGroupingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("numpy") is None:
            raise unittest.SkipTest("Cone grouping tests require NumPy")

    def test_nearest_resolves_symmetry_mirrors_and_ties(self):
        import numpy as np
        from xmippPyModules.coneGroupingTools.cone_grouping import (
            generate_symmetries,
            group_projection_directions,
        )

        x, y, z = np.eye(3)
        directions = np.array([y, -x, z])
        references = np.array([x, x, z])
        symmetries = generate_symmetries("c4")

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
        from xmippPyModules.coneGroupingTools.cone_grouping import (
            generate_symmetries,
            group_projection_directions,
        )

        x, y, z = np.eye(3)
        args = (
            np.array([(x + y) / np.sqrt(2), -x, z]),
            np.array([x, y]),
            generate_symmetries("c1"),
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
        from xmippPyModules.coneGroupingTools.cone_grouping import (
            deduplicate_with_symmetry,
            generate_symmetries,
        )

        # With C2 these directions lie at 0, 5 and 10 degrees in the same sector.
        # The discarded middle direction must leave the final direction alone.
        azimuths = np.deg2rad([0, 185, 190])
        directions = np.column_stack(
            [np.cos(azimuths), np.sin(azimuths), np.zeros_like(azimuths)]
        )
        keep = deduplicate_with_symmetry(
            directions,
            generate_symmetries("c2"),
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
            generate_symmetries,
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
            generate_symmetries("c4"),
            batch_size=3,
        )
        np.testing.assert_allclose(np.rad2deg(psi), [-90, 0, 0, 0], atol=1e-12)
        np.testing.assert_array_equal(flip, [False, False, True, False])
        np.testing.assert_allclose(shifts, [[-3, 2], [2, 3], [-2, 3], [2, 3]])


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
