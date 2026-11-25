import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "applications/scripts/gmm_average_estimation/batch_gmm_average_estimation.py"
)


def require_dependencies(*modules):
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise unittest.SkipTest(f"Missing GMM test dependencies: {', '.join(missing)}")


class GmmAverageScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_dependencies("mrcfile", "numpy", "pandas", "starfile", "torch")

    def run_cli(self, *args):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--device", "cpu", *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_writes_corrected_averages_in_class_order(self):
        import mrcfile
        import numpy as np
        import pandas as pd
        import starfile

        class_2 = np.array([[4, 1, 2], [1, 3, 1], [2, 1, 4]], dtype=np.float32)
        class_9 = np.array([[1, 2, 3], [2, 4, 2], [3, 2, 1]], dtype=np.float32)
        # Positive multiples within each class have equal Tagare weights.
        images = np.stack([class_9, 3 * class_2, 5 * class_9, class_2])

        with tempfile.TemporaryDirectory() as tmpdir:
            stack = Path(tmpdir) / "particles.mrcs"
            metadata = Path(tmpdir) / "particles.xmd"
            output = Path(tmpdir) / "averages.mrcs"
            mrcfile.write(stack, images)
            starfile.write(
                pd.DataFrame(
                    {
                        "image": [f"{index}@{stack}" for index in (3, 4, 1, 2)],
                        "itemId": [103, 104, 101, 102],
                        "ref": [9, 2, 9, 2],
                    }
                ),
                metadata,
            )

            self.run_cli(
                "--input-xmd", str(metadata), "--out-corrected-avgs", str(output), "irls"
            )
            with mrcfile.open(output) as averages:
                np.testing.assert_allclose(
                    averages.data,
                    np.stack([2 * class_2, 3 * class_9]),
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_cli_gmm_weights_reproduce_the_unmasked_corrected_average(self):
        import mrcfile
        import numpy as np
        import pandas as pd
        import starfile

        indices = np.arange(80)
        outliers = indices % 5 == 0
        images = np.zeros((len(indices), 5, 5), dtype=np.float32)
        images[:, 2, 2] = 1.0
        images[:, 2, 3] = np.linspace(-0.03, 0.03, len(indices))
        images[outliers, 2, 3] += 3.0 * (-1.0) ** (indices[outliers] // 5)
        # Signal outside the circular fitting mask still belongs in the average.
        images[:, 0, 0] = 0.25 + 2.0 * outliers
        item_ids = 1000 + indices
        order = np.random.default_rng(0).permutation(indices)

        with tempfile.TemporaryDirectory() as tmpdir:
            stack = Path(tmpdir) / "particles.mrcs"
            metadata = Path(tmpdir) / "particles.xmd"
            output_metadata = Path(tmpdir) / "weights.star"
            output_average = Path(tmpdir) / "corrected.mrcs"
            mrcfile.write(stack, images)
            starfile.write(
                pd.DataFrame(
                    {
                        "image": [f"{index + 1}@{stack}" for index in order],
                        "itemId": item_ids[order],
                        "ref": 2,
                    }
                ),
                metadata,
            )

            self.run_cli(
                "--input-xmd", str(metadata),
                "--out-star", str(output_metadata),
                "--out-corrected-avgs", str(output_average),
                "--gmm", "--gmm-initial-bad-weight", "0.2",
                "--gmm-initial-bad-quantile", "0.1",
                "irls",
            )

            results = starfile.read(output_metadata).set_index("itemId").loc[item_ids]
            weights = results["wRobustGmm"].to_numpy()
            self.assertLess(weights[outliers].mean(), weights[~outliers].mean())
            expected = np.average(images, axis=0, weights=weights)
            with mrcfile.open(output_average) as averages:
                np.testing.assert_allclose(
                    averages.data.reshape(expected.shape), expected, rtol=1e-5, atol=1e-6
                )

    def test_metadata_keeps_all_weights_from_each_gmm_winner(self):
        import numpy as np
        import pandas as pd

        with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
            from batch_gmm_average_estimation import write_weights_to_dataframe

        output = pd.DataFrame(
            {
                "itemId": [17, 42],
                "ref": [-1, -1],
                "wRobust": np.nan,
                "wRobustStd": np.nan,
                "wRobustGmm": np.nan,
            }
        ).set_index("itemId", drop=False)
        # The raw robust scores and GMM scores rank the assignments oppositely.
        write_weights_to_dataframe(
            output,
            robust_weights_np=np.array([0.90, 0.20]),
            gmm_weights_np=np.array([0.25, 0.75]),
            item_ids=np.array([42, 17]),
            group_by_column="ref",
            group_by_value=2,
        )
        write_weights_to_dataframe(
            output,
            robust_weights_np=np.array([0.95, 0.35]),
            gmm_weights_np=np.array([0.20, 0.85]),
            item_ids=np.array([17, 42]),
            group_by_column="ref",
            group_by_value=9,
        )
        # Equal GMM scores keep the earlier assignment and its weights.
        write_weights_to_dataframe(
            output,
            robust_weights_np=np.array([0.99, 0.45]),
            gmm_weights_np=np.array([0.85, 0.75]),
            item_ids=np.array([42, 17]),
            group_by_column="ref",
            group_by_value=12,
        )

        expected = pd.DataFrame(
            {
                "itemId": [17, 42],
                "ref": [2, 9],
                "wRobust": [0.20, 0.35],
                "wRobustStd": [-1.0, -1.0],
                "wRobustGmm": [0.75, 0.85],
            },
            index=pd.Index([17, 42], name="itemId"),
        )
        pd.testing.assert_frame_equal(output, expected, check_dtype=False)


class RecursiveGmmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_dependencies("torch")

    @staticmethod
    def distance_from_reference(images, reference):
        return (images - reference).flatten(1).norm(dim=1)

    def test_separated_outliers_get_lower_weights_and_less_influence(self):
        import torch
        from xmippPyModules.gmmAverageTools.gmm_estimator import RecursiveGMMEstimator

        clean = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        # Interleave outliers so their weights must be returned in input order.
        outliers = torch.arange(80) % 5 == 0
        offsets = torch.linspace(-0.2, 0.2, 80) + 8 * outliers
        images = clean + offsets[:, None, None]
        estimator = RecursiveGMMEstimator(
            distance_function=self.distance_from_reference,
            initial_bad_mean_quantile=0.9,
            initial_bad_weight=0.2,
            max_iter=5,
            random_state=0,
        )

        result = estimator.fit(images)

        weights = result.weights.flatten()
        self.assertLess(weights[outliers].max().item(), weights[~outliers].min().item())
        corrected_error = (result.estimate - clean).norm().item()
        ordinary_error = (images.mean(dim=0) - clean).norm().item()
        self.assertLess(corrected_error, ordinary_error)

    def test_constant_distances_fall_back_to_the_ordinary_mean(self):
        import torch
        from xmippPyModules.gmmAverageTools.gmm_estimator import RecursiveGMMEstimator

        reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        # Both offsets are equally far from the initial reference, but the mean shifts.
        offsets = torch.tensor([1.0] * 48 + [-1.0] * 16)
        images = reference + offsets[:, None, None]
        estimator = RecursiveGMMEstimator(distance_function=self.distance_from_reference)

        result = estimator.fit(images, reference=reference)

        torch.testing.assert_close(result.estimate, reference + 0.5)
        torch.testing.assert_close(result.weights.flatten(), torch.ones(len(images)))


class IrlsSolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_dependencies("numpy", "torch")

    def test_irls_recovers_the_weighted_least_squares_estimate_with_ctf(self):
        import torch
        from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator

        images = torch.tensor(
            [
                [[7.0, 8.0], [9.0, 10.0]],
                [[1.0, -1.0], [-3.0, 7.0]],
                [[30.0, 20.0], [10.0, 40.0]],
            ]
        )
        ctf = torch.tensor([1.0, -2.0, 0.0])[:, None, None]

        def fixed_weights(_images, _reference, _std):
            return torch.tensor([1.0, 0.5, 0.25])

        solver = IRLSMEstimator(weight_function=fixed_weights, max_iter=2, tol=1e-6)
        result = solver.fit(images, ctf=ctf)

        # The normal equation is 3 * estimate = first image - second image.
        # The third image has zero CTF and contributes no information.
        torch.testing.assert_close(result.estimate, torch.tensor([[2.0, 3.0], [4.0, 1.0]]))

    def test_fourier_mask_limits_weighting_and_preserves_full_bandwidth(self):
        import torch
        from xmippPyModules.gmmAverageTools.fourier_irls_estimator import JointIRLSFourier
        from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator
        from xmippPyModules.gmmAverageTools.masks import create_lowpass_rfft_mask

        phase = torch.arange(7, dtype=torch.float64) * (2 * torch.pi / 7)
        low = phase.cos().repeat(3, 1)
        high = (3 * phase).sin().repeat(3, 1)
        images = torch.stack([low + 3 * high, 2 * low + high])
        mask = torch.from_numpy(create_lowpass_rfft_mask(low.shape, cutoff=1.0))

        def spectral_weights(spectra, reference, _std):
            residual_norm = (spectra - reference).flatten(1).norm(dim=1)
            relative_residual = residual_norm / reference.norm()
            return 1 / (1 + relative_residual.square())

        irls = IRLSMEstimator(weight_function=spectral_weights, max_iter=1, tol=1e-6)
        solver = JointIRLSFourier(irls, weight_approach="per-image", mask=mask)
        result = solver.fit(images, reference=low)

        # In the retained band the relative residuals are 0 and 1, yielding
        # weights of 1 and 1/2. Apply those weights to both frequency components.
        torch.testing.assert_close(result.estimate, (4 * low + 7 * high) / 3)

    def test_fourier_single_particle_reproduces_the_image_with_unit_weight(self):
        from functools import partial

        import torch
        from xmippPyModules.gmmAverageTools.fourier_irls_estimator import JointIRLSFourier
        from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator
        from xmippPyModules.gmmAverageTools.weights import smooth_redescending_weights_modulus

        image = torch.arange(15.0).reshape(3, 5)
        irls = IRLSMEstimator(
            weight_function=partial(smooth_redescending_weights_modulus, delta=1.5),
            max_iter=5,
            tol=1e-6,
        )
        result = JointIRLSFourier(irls).fit(image[None])

        torch.testing.assert_close(result.estimate, image)
        torch.testing.assert_close(result.weights.flatten(), torch.ones(1))

    def test_masked_fourier_prior_uses_inferred_full_spectrum_variance(self):
        import torch
        from xmippPyModules.gmmAverageTools.fourier_irls_estimator import JointIRLSFourier
        from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator
        from xmippPyModules.gmmAverageTools.masks import create_lowpass_rfft_mask

        impulse = torch.zeros((3, 5), dtype=torch.float64)
        impulse[0, 0] = 1.0
        images = torch.stack([impulse, 3 * impulse])
        mask = torch.from_numpy(create_lowpass_rfft_mask(impulse.shape, cutoff=1.0))

        def unit_weights(spectra, _reference, _std):
            return spectra.real.new_ones(len(spectra))

        irls = IRLSMEstimator(weight_function=unit_weights, max_iter=2, tol=1e-6)
        solver = JointIRLSFourier(irls, weight_approach="per-image", mask=mask)
        result = solver.fit(images, prior_mean=6 * impulse, prior_variance=1.0)

        # Each Fourier coefficient has sample variance 2. The two observations
        # together have precision 1, so their mean and the prior have equal influence.
        torch.testing.assert_close(result.estimate, 4 * impulse)


class AdmmSolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_dependencies("torch")

    def test_admm_recovers_the_common_weighted_mean(self):
        import torch
        from xmippPyModules.gmmAverageTools.admm_estimator import ADMMEstimator
        from xmippPyModules.gmmAverageTools.fourier_irls_estimator import JointIRLSFourier
        from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator

        first = torch.arange(15, dtype=torch.float64).reshape(3, 5)
        second = first.flip(-1) + 5
        images = torch.stack([first, second])

        def fixed_weights(observations, _reference, _std):
            return observations.real.new_tensor([1.0, 0.5])

        real_solver = IRLSMEstimator(fixed_weights, max_iter=5, tol=1e-9)
        fourier_solver = JointIRLSFourier(
            IRLSMEstimator(fixed_weights, max_iter=5, tol=1e-9),
            weight_approach="per-image",
        )
        solver = ADMMEstimator(
            real_solver,
            fourier_solver,
            max_iter=100,
            initial_mu=1.0,
            fourier_multiplier=1.0,
            atol=1e-9,
            rtol=1e-9,
        )
        result = solver.fit(
            images,
            reference=torch.zeros_like(first),
            image_variance_real=first.new_tensor(1.0),
            image_variance_fourier=first.new_tensor(2.0),
        )

        torch.testing.assert_close(result.estimate, (2 * first + second) / 3)
