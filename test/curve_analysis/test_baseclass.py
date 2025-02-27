# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Test curve fitting base class."""
from test.base import QiskitExperimentsTestCase
from test.fake_experiment import FakeExperiment

import numpy as np

from lmfit.models import ExpressionModel
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.curve_analysis import (
    CurveAnalysis,
    CompositeCurveAnalysis,
    fit_function,
)
from qiskit_experiments.curve_analysis.curve_data import (
    SeriesDef,
    CurveFitResult,
    ParameterRepr,
    FitOptions,
)
from qiskit_experiments.data_processing import DataProcessor, Probability
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import (
    ExperimentData,
    AnalysisResultData,
    CompositeAnalysis,
)


class CurveAnalysisTestCase(QiskitExperimentsTestCase):
    """Base class for testing Curve Analysis subclasses."""

    @staticmethod
    def single_sampler(x, y, shots=10000, seed=123, **metadata):
        """Prepare fake experiment data."""
        rng = np.random.default_rng(seed=seed)
        counts = rng.binomial(shots, y)

        circuit_results = [
            {
                "counts": {"0": shots - count, "1": count},
                "metadata": {"xval": xi, **metadata},
            }
            for xi, count in zip(x, counts)
        ]
        expdata = ExperimentData(experiment=FakeExperiment())
        expdata.add_data(circuit_results)
        expdata.metadata["meas_level"] = MeasLevel.CLASSIFIED

        return expdata

    @staticmethod
    def parallel_sampler(x, y1, y2, shots=10000, seed=123, **metadata):
        """Prepare fake parallel experiment data."""
        rng = np.random.default_rng(seed=seed)

        circuit_results = []
        for xi, p1, p2 in zip(x, y1, y2):
            cs = rng.multinomial(
                shots, [(1 - p1) * (1 - p2), p1 * (1 - p2), (1 - p1) * p2, p1 * p2]
            )
            circ_data = {
                "counts": {"00": cs[0], "01": cs[1], "10": cs[2], "11": cs[3]},
                "metadata": {
                    "composite_index": [0, 1],
                    "composite_metadata": [
                        {"xval": xi, **metadata},
                        {"xval": xi, **metadata},
                    ],
                    "composite_qubits": [[0], [1]],
                    "composite_clbits": [[0], [1]],
                },
            }
            circuit_results.append(circ_data)

        expdata = ExperimentData(experiment=FakeExperiment())
        expdata.add_data(circuit_results)
        expdata.metadata["meas_level"] = MeasLevel.CLASSIFIED

        return expdata


class TestCurveAnalysis(CurveAnalysisTestCase):
    """A collection of CurveAnalysis unit tests and integration tests."""

    def test_roundtrip_serialize(self):
        """A testcase for serializing analysis instance."""
        analysis = CurveAnalysis(
            models=[ExpressionModel(expr="par0 * x + par1", name="test")]
        )
        self.assertRoundTripSerializable(analysis, check_func=self.json_equiv)

    def test_parameters(self):
        """A testcase for getting fit parameters with attribute."""
        analysis = CurveAnalysis(
            models=[ExpressionModel(expr="par0 * x + par1", name="test")]
        )
        self.assertListEqual(analysis.parameters, ["par0", "par1"])

        analysis.set_options(fixed_parameters={"par0": 1.0})
        self.assertListEqual(analysis.parameters, ["par1"])

    def test_combine_funcs_with_different_parameters(self):
        """A testcase for composing two objectives with different signature."""
        analysis = CurveAnalysis(
            models=[
                ExpressionModel(expr="par0 * x + par1", name="test1"),
                ExpressionModel(expr="par2 * x + par1", name="test2"),
            ]
        )
        self.assertListEqual(analysis.parameters, ["par0", "par1", "par2"])

    def test_data_extraction(self):
        """A testcase for extracting data."""
        x = np.linspace(0, 1, 10)
        y1 = 0.1 * x + 0.3
        y2 = 0.2 * x + 0.4
        expdata1 = self.single_sampler(x, y1, shots=1000000, series=1)
        expdata2 = self.single_sampler(x, y2, shots=1000000, series=2)

        analysis = CurveAnalysis(
            models=[
                ExpressionModel(
                    expr="par0 * x + par1",
                    name="s1",
                    data_sort_key={"series": 1},
                ),
                ExpressionModel(
                    expr="par2 * x + par3",
                    name="s2",
                    data_sort_key={"series": 2},
                ),
            ]
        )
        analysis.set_options(
            data_processor=DataProcessor("counts", [Probability("1")]),
        )

        curve_data = analysis._run_data_processing(
            raw_data=expdata1.data() + expdata2.data(),
            models=analysis._models,
        )
        self.assertListEqual(curve_data.labels, ["s1", "s2"])

        # check data of series1
        sub1 = curve_data.get_subset_of("s1")
        self.assertListEqual(sub1.labels, ["s1"])
        np.testing.assert_array_equal(sub1.x, x)
        np.testing.assert_array_almost_equal(sub1.y, y1, decimal=3)
        np.testing.assert_array_equal(sub1.data_allocation, np.full(x.size, 0))

        # check data of series2
        sub2 = curve_data.get_subset_of("s2")
        self.assertListEqual(sub2.labels, ["s2"])
        np.testing.assert_array_equal(sub2.x, x)
        np.testing.assert_array_almost_equal(sub2.y, y2, decimal=3)
        np.testing.assert_array_equal(sub2.data_allocation, np.full(x.size, 1))

    def test_create_result(self):
        """A testcase for creating analysis result data from fit data."""
        analysis = CurveAnalysis(
            models=[ExpressionModel(expr="par0 * x + par1", name="s1")]
        )
        analysis.set_options(
            result_parameters=["par0", ParameterRepr("par1", "Param1", "SomeUnit")]
        )

        covar = np.diag([0.1**2, 0.2**2])

        fit_data = CurveFitResult(
            method="some_method",
            model_repr={"s1": "par0 * x + par1"},
            success=True,
            params={"par0": 0.3, "par1": 0.4},
            var_names=["par0", "par1"],
            covar=covar,
            reduced_chisq=1.5,
        )

        result_data = analysis._create_analysis_results(
            fit_data, quality="good", test="hoge"
        )

        # entry name
        self.assertEqual(result_data[0].name, "par0")
        self.assertEqual(result_data[1].name, "Param1")

        # entry value
        self.assertEqual(result_data[0].value.nominal_value, 0.3)
        self.assertEqual(result_data[0].value.std_dev, 0.1)
        self.assertEqual(result_data[1].value.nominal_value, 0.4)
        self.assertEqual(result_data[1].value.std_dev, 0.2)

        # other metadata
        self.assertEqual(result_data[1].quality, "good")
        self.assertEqual(result_data[1].chisq, 1.5)
        ref_meta = {
            "test": "hoge",
            "unit": "SomeUnit",
        }
        self.assertDictEqual(result_data[1].extra, ref_meta)

    def test_invalid_type_options(self):
        """A testcase for failing with invalid options."""
        analysis = CurveAnalysis()

        class InvalidClass:
            """Dummy class."""

            pass

        with self.assertRaises(TypeError):
            analysis.set_options(data_processor=InvalidClass())

        with self.assertRaises(TypeError):
            analysis.set_options(curve_drawer=InvalidClass())

    def test_end_to_end_single_function(self):
        """Integration test for single function."""
        analysis = CurveAnalysis(
            models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")]
        )
        analysis.set_options(
            data_processor=DataProcessor(
                input_key="counts", data_actions=[Probability("1")]
            ),
            p0={"amp": 0.5, "tau": 0.3},
            result_parameters=["amp", "tau"],
            plot=False,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y = amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y)
        result = analysis.run(test_data).block_for_results()

        self.assertAlmostEqual(
            result.analysis_results("amp").value.nominal_value, 0.5, delta=0.1
        )
        self.assertAlmostEqual(
            result.analysis_results("tau").value.nominal_value, 0.3, delta=0.1
        )

    def test_end_to_end_multi_objective(self):
        """Integration test for multi objective function."""
        analysis = CurveAnalysis(
            models=[
                ExpressionModel(
                    expr="amp * cos(2 * pi * freq * x + phi) + base",
                    name="m1",
                    data_sort_key={"series": "cos"},
                ),
                ExpressionModel(
                    expr="amp * sin(2 * pi * freq * x + phi) + base",
                    name="m2",
                    data_sort_key={"series": "sin"},
                ),
            ]
        )
        analysis.set_options(
            data_processor=DataProcessor(
                input_key="counts", data_actions=[Probability("1")]
            ),
            p0={"amp": 0.5, "freq": 2.1, "phi": 0.3, "base": 0.1},
            result_parameters=["amp", "freq", "phi", "base"],
            plot=False,
        )
        amp = 0.3
        freq = 2.1
        phi = 0.3
        base = 0.4

        x = np.linspace(0, 1, 100)
        y1 = amp * np.cos(2 * np.pi * freq * x + phi) + base
        y2 = amp * np.sin(2 * np.pi * freq * x + phi) + base

        test_data1 = self.single_sampler(x, y1, series="cos")
        test_data2 = self.single_sampler(x, y2, series="sin")

        expdata = ExperimentData(experiment=FakeExperiment())
        expdata.add_data(test_data1.data())
        expdata.add_data(test_data2.data())
        expdata.metadata["meas_level"] = MeasLevel.CLASSIFIED

        result = analysis.run(expdata).block_for_results()

        self.assertAlmostEqual(
            result.analysis_results("amp").value.nominal_value, amp, delta=0.1
        )
        self.assertAlmostEqual(
            result.analysis_results("freq").value.nominal_value, freq, delta=0.1
        )
        self.assertAlmostEqual(
            result.analysis_results("phi").value.nominal_value, phi, delta=0.1
        )
        self.assertAlmostEqual(
            result.analysis_results("base").value.nominal_value, base, delta=0.1
        )

    def test_end_to_end_single_function_with_fixed_parameter(self):
        """Integration test for fitting with fixed parameter."""
        analysis = CurveAnalysis(
            models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")]
        )
        analysis.set_options(
            data_processor=DataProcessor(
                input_key="counts", data_actions=[Probability("1")]
            ),
            p0={"tau": 0.3},
            result_parameters=["amp", "tau"],
            fixed_parameters={"amp": 0.5},
            plot=False,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y = amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y)
        result = analysis.run(test_data).block_for_results()

        self.assertEqual(result.analysis_results("amp").value.nominal_value, 0.5)
        self.assertEqual(result.analysis_results("amp").value.std_dev, 0.0)
        self.assertAlmostEqual(
            result.analysis_results("tau").value.nominal_value, 0.3, delta=0.1
        )

    def test_end_to_end_compute_new_entry(self):
        """Integration test for computing new parameter with error propagation."""

        class CustomAnalysis(CurveAnalysis):
            """Custom analysis class to override result generation."""

            def __init__(self):
                super().__init__(
                    models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")]
                )

            def _create_analysis_results(self, fit_data, quality, **metadata):
                results = super()._create_analysis_results(
                    fit_data, quality, **metadata
                )
                u_amp = fit_data.ufloat_params["amp"]
                u_tau = fit_data.ufloat_params["tau"]
                results.append(
                    AnalysisResultData(
                        name="new_value",
                        value=u_amp + u_tau,
                    )
                )
                return results

        analysis = CustomAnalysis()
        analysis.set_options(
            data_processor=DataProcessor(
                input_key="counts", data_actions=[Probability("1")]
            ),
            p0={"amp": 0.5, "tau": 0.3},
            plot=False,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y = amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y)
        result = analysis.run(test_data).block_for_results()

        new_value = result.analysis_results("new_value").value

        # Use ufloat_params in @Parameters dataclass.
        # This dataclass stores UFloat values with correlation.
        fit_amp = result.analysis_results(0).value.ufloat_params["amp"]
        fit_tau = result.analysis_results(0).value.ufloat_params["tau"]

        self.assertEqual(new_value.n, fit_amp.n + fit_tau.n)

        # This is not equal because of fit parameter correlation
        self.assertNotEqual(new_value.s, np.sqrt(fit_amp.s**2 + fit_tau.s**2))
        self.assertEqual(new_value.s, (fit_amp + fit_tau).s)

    def test_end_to_end_create_model_at_run(self):
        """Integration test for dynamically generate model at run time."""

        class CustomAnalysis(CurveAnalysis):
            """Custom analysis class to override model generation."""

            @classmethod
            def _default_options(cls):
                options = super()._default_options()
                options.model_var = None

                return options

            def _initialize(self, experiment_data):
                super()._initialize(experiment_data)

                # Generate model with `model_var` option
                self._models = [
                    ExpressionModel(
                        expr=f"{self.options.model_var} * amp * exp(-x/tau)",
                        name="test",
                    )
                ]

        analysis = CustomAnalysis()
        analysis.set_options(
            data_processor=DataProcessor(
                input_key="counts", data_actions=[Probability("1")]
            ),
            p0={"amp": 0.5, "tau": 0.3},
            result_parameters=["amp", "tau"],
            plot=False,
            model_var=0.5,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y = 0.5 * amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y)
        result = analysis.run(test_data).block_for_results()

        self.assertAlmostEqual(
            result.analysis_results("amp").value.nominal_value, 0.5, delta=0.1
        )
        self.assertAlmostEqual(
            result.analysis_results("tau").value.nominal_value, 0.3, delta=0.1
        )

    def test_end_to_end_parallel_analysis(self):
        """Integration test for running two curve analyses in parallel."""

        analysis1 = CurveAnalysis(
            models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")]
        )
        analysis1.set_options(
            data_processor=DataProcessor(
                input_key="counts", data_actions=[Probability("1")]
            ),
            p0={"amp": 0.5, "tau": 0.3},
            result_parameters=["amp", "tau"],
            plot=False,
        )

        analysis2 = CurveAnalysis(
            models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")]
        )
        analysis2.set_options(
            data_processor=DataProcessor(
                input_key="counts", data_actions=[Probability("1")]
            ),
            p0={"amp": 0.7, "tau": 0.5},
            result_parameters=["amp", "tau"],
            plot=False,
        )

        composite = CompositeAnalysis([analysis1, analysis2], flatten_results=True)
        amp1 = 0.5
        tau1 = 0.3
        amp2 = 0.7
        tau2 = 0.5

        x = np.linspace(0, 1, 100)
        y1 = amp1 * np.exp(-x / tau1)
        y2 = amp2 * np.exp(-x / tau2)

        test_data = self.parallel_sampler(x, y1, y2)
        result = composite.run(test_data).block_for_results()

        amps = result.analysis_results("amp")
        taus = result.analysis_results("tau")

        self.assertAlmostEqual(amps[0].value.nominal_value, amp1, delta=0.1)
        self.assertAlmostEqual(amps[1].value.nominal_value, amp2, delta=0.1)

        self.assertAlmostEqual(taus[0].value.nominal_value, tau1, delta=0.1)
        self.assertAlmostEqual(taus[1].value.nominal_value, tau2, delta=0.1)

    def test_get_init_params(self):
        """Integration test for getting initial parameter from overview entry."""

        analysis = CurveAnalysis(
            models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")]
        )
        analysis.set_options(
            data_processor=DataProcessor(
                input_key="counts", data_actions=[Probability("1")]
            ),
            p0={"amp": 0.45, "tau": 0.25},
            plot=False,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y_true = amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y_true)
        result = analysis.run(test_data).block_for_results()

        overview = result.analysis_results(0).value

        self.assertDictEqual(overview.init_params, {"amp": 0.45, "tau": 0.25})

        y_ref = 0.45 * np.exp(-x / 0.25)
        y_reproduced = analysis.models[0].eval(x=x, **overview.init_params)
        np.testing.assert_array_almost_equal(y_ref, y_reproduced)

    def test_multi_composite_curve_analysis(self):
        """Integration test for composite curve analysis.

        This analysis consists of two curve fittings for cos and sin series.
        This experiment is executed twice for different setups of "A" and "B".
        """
        analyses = []

        group_names = ["group_A", "group_B"]
        setups = ["setup_A", "setup_B"]
        for group_name, setup in zip(group_names, setups):
            analysis = CurveAnalysis(
                models=[
                    ExpressionModel(
                        expr="amp * cos(2 * pi * freq * x) + b",
                        data_sort_key={"type": "cos"},
                    ),
                    ExpressionModel(
                        expr="amp * sin(2 * pi * freq * x) + b",
                        data_sort_key={"type": "sin"},
                    ),
                ],
                name=group_name,
            )
            analysis.set_options(
                filter_data={"setup": setup},
                result_parameters=["amp"],
                data_processor=DataProcessor(
                    input_key="counts", data_actions=[Probability("1")]
                ),
            )
            analyses.append(analysis)

        group_analysis = CompositeCurveAnalysis(analyses)
        group_analysis.analyses("group_A").set_options(
            p0={"amp": 0.3, "freq": 2.1, "b": 0.5}
        )
        group_analysis.analyses("group_B").set_options(
            p0={"amp": 0.5, "freq": 3.2, "b": 0.5}
        )
        group_analysis.set_options(plot=False)

        amp1 = 0.2
        amp2 = 0.4
        b1 = 0.5
        b2 = 0.5
        freq1 = 2.1
        freq2 = 3.2

        x = np.linspace(0, 1, 100)
        y1a = amp1 * np.cos(2 * np.pi * freq1 * x) + b1
        y2a = amp1 * np.sin(2 * np.pi * freq1 * x) + b1
        y1b = amp2 * np.cos(2 * np.pi * freq2 * x) + b2
        y2b = amp2 * np.sin(2 * np.pi * freq2 * x) + b2

        # metadata must contain key for filtering, specified in filter_data option.
        test_data1a = self.single_sampler(x, y1a, type="cos", setup="setup_A")
        test_data2a = self.single_sampler(x, y2a, type="sin", setup="setup_A")
        test_data1b = self.single_sampler(x, y1b, type="cos", setup="setup_B")
        test_data2b = self.single_sampler(x, y2b, type="sin", setup="setup_B")

        expdata = ExperimentData(experiment=FakeExperiment())
        expdata.add_data(test_data1a.data())
        expdata.add_data(test_data2a.data())
        expdata.add_data(test_data1b.data())
        expdata.add_data(test_data2b.data())
        expdata.metadata["meas_level"] = MeasLevel.CLASSIFIED

        result = group_analysis.run(expdata).block_for_results()
        amps = result.analysis_results("amp")

        # two entries are generated for group A and group B
        self.assertEqual(len(amps), 2)
        self.assertEqual(amps[0].extra["group"], "group_A")
        self.assertEqual(amps[1].extra["group"], "group_B")
        self.assertAlmostEqual(amps[0].value.n, 0.2, delta=0.1)
        self.assertAlmostEqual(amps[1].value.n, 0.4, delta=0.1)


class TestFitOptions(QiskitExperimentsTestCase):
    """Unittest for fit option object."""

    def test_empty(self):
        """Test if default value is automatically filled."""
        opt = FitOptions(["par0", "par1", "par2"])

        # bounds should be default to inf tuple. otherwise crashes the scipy fitter.
        ref_opts = {
            "p0": {"par0": None, "par1": None, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_create_option_with_dict(self):
        """Create option and fill with dictionary."""
        opt = FitOptions(
            ["par0", "par1", "par2"],
            default_p0={"par0": 0, "par1": 1, "par2": 2},
            default_bounds={"par0": (0, 1), "par1": (1, 2), "par2": (2, 3)},
        )

        ref_opts = {
            "p0": {"par0": 0.0, "par1": 1.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 1.0), "par1": (1.0, 2.0), "par2": (2.0, 3.0)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_create_option_with_array(self):
        """Create option and fill with array."""
        opt = FitOptions(
            ["par0", "par1", "par2"],
            default_p0=[0, 1, 2],
            default_bounds=[(0, 1), (1, 2), (2, 3)],
        )

        ref_opts = {
            "p0": {"par0": 0.0, "par1": 1.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 1.0), "par1": (1.0, 2.0), "par2": (2.0, 3.0)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_override_partial_dict(self):
        """Create option and override value with partial dictionary."""
        opt = FitOptions(["par0", "par1", "par2"])
        opt.p0.set_if_empty(par1=3)

        ref_opts = {
            "p0": {"par0": None, "par1": 3.0, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_cannot_override_assigned_value(self):
        """Test cannot override already assigned value."""
        opt = FitOptions(["par0", "par1", "par2"])
        opt.p0.set_if_empty(par1=3)
        opt.p0.set_if_empty(par1=5)

        ref_opts = {
            "p0": {"par0": None, "par1": 3.0, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_can_override_assigned_value_with_dict_access(self):
        """Test override already assigned value with direct dict access."""
        opt = FitOptions(["par0", "par1", "par2"])
        opt.p0["par1"] = 3
        opt.p0["par1"] = 5

        ref_opts = {
            "p0": {"par0": None, "par1": 5.0, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_cannot_override_user_option(self):
        """Test cannot override already assigned value."""
        opt = FitOptions(["par0", "par1", "par2"], default_p0={"par1": 3})
        opt.p0.set_if_empty(par1=5)

        ref_opts = {
            "p0": {"par0": None, "par1": 3, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_set_operation(self):
        """Test if set works and duplicated entry is removed."""
        opt1 = FitOptions(["par0", "par1"], default_p0=[0, 1])
        opt2 = FitOptions(["par0", "par1"], default_p0=[0, 1])
        opt3 = FitOptions(["par0", "par1"], default_p0=[0, 2])

        opts = set()
        opts.add(opt1)
        opts.add(opt2)
        opts.add(opt3)

        self.assertEqual(len(opts), 2)

    def test_detect_invalid_p0(self):
        """Test if invalid p0 raises Error."""
        with self.assertRaises(AnalysisError):
            # less element
            FitOptions(["par0", "par1", "par2"], default_p0=[0, 1])

    def test_detect_invalid_bounds(self):
        """Test if invalid bounds raises Error."""
        with self.assertRaises(AnalysisError):
            # less element
            FitOptions(["par0", "par1", "par2"], default_bounds=[(0, 1), (1, 2)])

        with self.assertRaises(AnalysisError):
            # not min-max tuple
            FitOptions(["par0", "par1", "par2"], default_bounds=[0, 1, 2])

        with self.assertRaises(AnalysisError):
            # max-min tuple
            FitOptions(
                ["par0", "par1", "par2"], default_bounds=[(1, 0), (2, 1), (3, 2)]
            )

    def test_detect_invalid_key(self):
        """Test if invalid key raises Error."""
        opt = FitOptions(["par0", "par1", "par2"])

        with self.assertRaises(AnalysisError):
            opt.p0.set_if_empty(par3=3)

    def test_set_extra_options(self):
        """Add extra fitter options."""
        opt = FitOptions(
            ["par0", "par1", "par2"],
            default_p0=[0, 1, 2],
            default_bounds=[(0, 1), (1, 2), (2, 3)],
        )
        opt.add_extra_options(ex1=0, ex2=1)

        ref_opts = {
            "p0": {"par0": 0.0, "par1": 1.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 1.0), "par1": (1.0, 2.0), "par2": (2.0, 3.0)},
            "ex1": 0,
            "ex2": 1,
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_complicated(self):
        """Test for realistic operations for algorithmic guess with user options."""
        user_p0 = {"par0": 1, "par1": None}
        user_bounds = {"par0": None, "par1": (-100, 100)}

        opt = FitOptions(
            ["par0", "par1", "par2"],
            default_p0=user_p0,
            default_bounds=user_bounds,
        )

        # similar computation in algorithmic guess

        opt.p0.set_if_empty(
            par0=5
        )  # this is ignored because user already provided initial guess
        opt.p0.set_if_empty(
            par1=opt.p0["par0"] * 2 + 3
        )  # user provided guess propagates

        opt.bounds.set_if_empty(par0=(0, 10))  # this will be set
        opt.add_extra_options(fitter="algo1")

        opt1 = opt.copy()  # copy options while keeping previous values
        opt1.p0.set_if_empty(par2=opt1.p0["par0"] + opt1.p0["par1"])

        opt2 = opt.copy()
        opt2.p0.set_if_empty(par2=opt2.p0["par0"] * 2)  # add another p2 value

        ref_opt1 = {
            "p0": {"par0": 1.0, "par1": 5.0, "par2": 6.0},
            "bounds": {
                "par0": (0.0, 10.0),
                "par1": (-100.0, 100.0),
                "par2": (-np.inf, np.inf),
            },
            "fitter": "algo1",
        }

        ref_opt2 = {
            "p0": {"par0": 1.0, "par1": 5.0, "par2": 2.0},
            "bounds": {
                "par0": (0.0, 10.0),
                "par1": (-100.0, 100.0),
                "par2": (-np.inf, np.inf),
            },
            "fitter": "algo1",
        }

        self.assertDictEqual(opt1.options, ref_opt1)
        self.assertDictEqual(opt2.options, ref_opt2)


class TestBackwardCompatibility(QiskitExperimentsTestCase):
    """Test case for backward compatibility."""

    def test_old_fixed_param_attributes(self):
        """Test if old class structure for fixed param is still supported."""

        with self.assertWarns(DeprecationWarning):

            class _DeprecatedAnalysis(CurveAnalysis):
                __series__ = [
                    SeriesDef(
                        fit_func=lambda x, par0, par1, par2, par3: fit_function.exponential_decay(
                            x, amp=par0, lamb=par1, x0=par2, baseline=par3
                        ),
                    )
                ]

                __fixed_parameters__ = ["par1"]

                @classmethod
                def _default_options(cls):
                    opts = super()._default_options()
                    opts.par1 = 2

                    return opts

        with self.assertWarns(DeprecationWarning):
            instance = _DeprecatedAnalysis()

        self.assertDictEqual(instance.options.fixed_parameters, {"par1": 2})

    def test_loading_data_with_deprecated_fixed_param(self):
        """Test loading old data with fixed parameters as standalone options."""

        with self.assertWarns(DeprecationWarning):

            class _DeprecatedAnalysis(CurveAnalysis):
                __series__ = [
                    SeriesDef(
                        fit_func=lambda x, par0, par1, par2, par3: fit_function.exponential_decay(
                            x, amp=par0, lamb=par1, x0=par2, baseline=par3
                        ),
                    )
                ]

        with self.assertWarns(DeprecationWarning):
            # old option data structure, i.e. fixed param as a standalone option
            # the analysis instance fixed parameters might be set via the experiment instance
            instance = _DeprecatedAnalysis.from_config({"options": {"par1": 2}})

        self.assertDictEqual(instance.options.fixed_parameters, {"par1": 2})

    def test_instantiating_series_def_in_old_format(self):
        """Test instantiating curve analysis with old series def format."""

        with self.assertWarns(DeprecationWarning):

            class _DeprecatedAnalysis(CurveAnalysis):
                __series__ = [
                    SeriesDef(
                        fit_func=lambda x, par0: fit_function.exponential_decay(
                            x, amp=par0
                        )
                    )
                ]

        with self.assertWarns(DeprecationWarning):
            instance = _DeprecatedAnalysis()

        # Still works.
        self.assertListEqual(instance.parameters, ["par0"])
