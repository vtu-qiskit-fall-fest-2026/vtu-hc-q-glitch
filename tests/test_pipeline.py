"""
Unit tests for Q-Glitch quantum and classical pipeline.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pytest
except ImportError:
    pytest = None

import numpy as np

from src.config import CONFIG, ProjectConfig
from src.data_generator import RareDiseaseDataGenerator, prepare_splits
from src.preprocessing import DataPreprocessor
from src.classical_models import ClassicalBaselines
from src.quantum_kernel import QuantumKernelClassifier
from src.quantum_vqc import VariationalQuantumClassifier
from src.hybrid_ensemble import HybridEnsemble
from src.evaluation import ModelEvaluator


class TestConfig(unittest.TestCase):
    """Test configuration module."""

    def test_default_config(self):
        cfg = ProjectConfig()
        self.assertEqual(cfg.project_name, "Q-Glitch")
        self.assertEqual(cfg.team_id, "VTU-QFF-HC-0190")
        self.assertEqual(cfg.quantum.n_qubits, 6)

    def test_quantum_config_defaults(self):
        self.assertEqual(CONFIG.quantum.feature_map_reps, 2)
        self.assertEqual(CONFIG.quantum.entanglement, "full")


class TestDataGenerator(unittest.TestCase):
    """Test synthetic data generation."""

    def test_generate_dataset(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        self.assertIn("clinical", data)
        self.assertIn("genetic", data)
        self.assertIn("phenotypic", data)
        self.assertIn("combined", data)
        self.assertIn("labels", data)
        self.assertEqual(len(data["labels"]), CONFIG.data.n_samples)

    def test_label_distribution(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        prevalence = data["labels"].mean()
        self.assertTrue(0.05 < prevalence < 0.35)

    def test_feature_dimensions(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        expected_total = (
            CONFIG.data.n_clinical_features
            + CONFIG.data.n_genetic_markers
            + CONFIG.data.n_phenotypic_features
        )
        self.assertEqual(data["combined"].shape[1], expected_total)

    def test_data_splits(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        splits = prepare_splits(data)
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)
        total = sum(len(s[0]) for s in splits.values())
        self.assertEqual(total, CONFIG.data.n_samples)


class TestPreprocessor(unittest.TestCase):
    """Test preprocessing pipeline."""

    def setUp(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        self.splits = prepare_splits(data)

    def test_fit_transform(self):
        X_train, y_train = self.splits["train"]
        preprocessor = DataPreprocessor()
        X_cl, X_q = preprocessor.fit_transform(X_train, y_train)
        self.assertEqual(X_cl.shape[0], X_train.shape[0])
        self.assertEqual(X_q.shape[1], CONFIG.quantum.n_qubits)

    def test_quantum_range(self):
        X_train, y_train = self.splits["train"]
        preprocessor = DataPreprocessor()
        _, X_q = preprocessor.fit_transform(X_train, y_train)
        self.assertGreaterEqual(X_q.min(), -1e-5)
        self.assertLessEqual(X_q.max(), np.pi + 0.05)


class TestClassicalModels(unittest.TestCase):
    """Test classical ML baselines."""

    def setUp(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        splits = prepare_splits(data)
        X_train, y_train = splits["train"]
        X_test, y_test = splits["test"]

        preprocessor = DataPreprocessor()
        X_train_cl, _ = preprocessor.fit_transform(X_train, y_train)
        X_test_cl = preprocessor.transform_classical(X_test)

        self.classical = ClassicalBaselines()
        self.classical.train_all(X_train_cl, y_train)
        self.X_test_cl = X_test_cl
        self.y_test = y_test

    def test_models_created(self):
        self.assertEqual(len(self.classical.models), 3)

    def test_evaluation(self):
        results = self.classical.evaluate_all(self.X_test_cl, self.y_test)
        for name, metrics in results.items():
            self.assertIn("accuracy", metrics)
            self.assertTrue(0 <= metrics["accuracy"] <= 1)
            self.assertIn("f1_score", metrics)


class TestQuantumModels(unittest.TestCase):
    """Test quantum model initialization and fallbacks."""

    def test_qsvm_init(self):
        qsvm = QuantumKernelClassifier()
        self.assertEqual(qsvm.n_qubits, CONFIG.quantum.n_qubits)
        self.assertEqual(qsvm.optimal_threshold, 0.5)

    def test_vqc_init(self):
        vqc = VariationalQuantumClassifier()
        self.assertEqual(vqc.n_qubits, CONFIG.quantum.n_qubits)
        self.assertEqual(vqc.ansatz_reps, CONFIG.quantum.ansatz_reps)
        self.assertEqual(vqc.optimal_threshold, 0.5)


class TestQuantumNoise(unittest.TestCase):
    """Test quantum hardware noise simulation and readout error mitigation."""

    def test_noise_benchmark_execution(self):
        from src.quantum_noise import QuantumNoiseBenchmark
        bench = QuantumNoiseBenchmark(n_qubits=4, depolarizing_rates=[0.01, 0.05])
        res = bench.run_noise_benchmark()
        self.assertIn("avg_tvd_reduction_percent", res)
        self.assertTrue(res["avg_tvd_reduction_percent"] > 0)
        self.assertEqual(len(res["fidelities_unmitigated"]), 2)
        self.assertEqual(len(res["fidelities_mitigated"]), 2)
        self.assertGreaterEqual(res["fidelities_mitigated"][1], res["fidelities_unmitigated"][1])


class TestEvaluator(unittest.TestCase):
    """Test evaluation and reporting."""

    def test_comparison_table(self):
        evaluator = ModelEvaluator()
        evaluator.add_result("Model A", {"accuracy": 0.9, "f1_score": 0.85})
        evaluator.add_result("Model B", {"accuracy": 0.8, "f1_score": 0.75})
        df = evaluator.get_comparison_table()
        self.assertEqual(len(df), 2)

    def test_best_model(self):
        evaluator = ModelEvaluator()
        evaluator.add_result("Model A", {"f1_score": 0.85})
        evaluator.add_result("Model B", {"f1_score": 0.95})
        self.assertEqual(evaluator.get_best_model("f1_score"), "Model B")


class TestDifferentialDiagnosis(unittest.TestCase):
    """Test rare-disease categorization engine."""

    def test_laminopathy_classification(self):
        from src.differential_diagnosis import RareDiseaseCategorizer
        features = {
            "lmna_mutation_score": 9.5,
            "progerin_level": 8.0,
            "telomere_length": 2.5,
            "growth_retardation_score": 0.9,
            "skin_elasticity": 0.1,
            "joint_mobility": 0.3,
        }
        res = RareDiseaseCategorizer.classify_category(features)
        self.assertIn("Laminopathy", res["primary_category"])
        self.assertTrue(res["confidence"] > 0.25)
        self.assertIn("LMNA mutation", res["primary_biomarkers"])

    def test_categories_metadata(self):
        from src.differential_diagnosis import RareDiseaseCategorizer
        self.assertIn("laminopathy", RareDiseaseCategorizer.CATEGORIES)
        self.assertIn("lysosomal", RareDiseaseCategorizer.CATEGORIES)
        self.assertIn("connective_tissue", RareDiseaseCategorizer.CATEGORIES)
        self.assertIn("metabolic", RareDiseaseCategorizer.CATEGORIES)


class TestModelStore(unittest.TestCase):
    """Test model persistence manager."""

    def test_save_and_load(self):
        from src.model_store import save_trained_pipeline, load_trained_pipeline
        bundle = {"test_key": "test_value", "accuracy": 0.95}
        test_path = "/tmp/test_bundle.pkl"
        save_trained_pipeline(bundle, test_path)
        loaded = load_trained_pipeline(test_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["test_key"], "test_value")
        if os.path.exists(test_path):
            os.remove(test_path)


class TestHybridEnsemble(unittest.TestCase):
    """Unit tests for the Hybrid Quantum-Classical Ensemble."""

    @classmethod
    def setUpClass(cls):
        """Prepare shared minimal data and trained components for ensemble tests."""
        np.random.seed(42)
        cls.n = 40
        cls.X_q = np.random.uniform(0.1, np.pi - 0.1, size=(cls.n, 6))
        cls.X_c = np.random.randn(cls.n, 10)
        cls.y = np.array([1] * 8 + [0] * 32)

        # Train minimal classical models for ensemble
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        cls.rf = RandomForestClassifier(n_estimators=10, random_state=42)
        cls.rf.fit(cls.X_c, cls.y)
        cls.svm = SVC(kernel="rbf", probability=True, random_state=42)
        cls.svm.fit(cls.X_c, cls.y)

    def _make_ensemble(self):
        ens = HybridEnsemble(
            quantum_classifier=None,   # No QSVC — tests graceful fallback
            classical_rf=self.rf,
            classical_svm=self.svm,
        )
        ens.is_trained = True
        return ens

    def test_ensemble_instantiation(self):
        """Ensemble initialises with correct default weights."""
        ens = self._make_ensemble()
        self.assertAlmostEqual(ens.base_weights.sum(), 1.0, places=6)
        self.assertAlmostEqual(ens.borderline_weights.sum(), 1.0, places=6)

    def test_predict_proba_shape(self):
        """predict_proba_from_features returns correct shape and valid probabilities."""
        ens = self._make_ensemble()
        probs = ens.predict_proba_from_features(self.X_q, self.X_c)
        self.assertEqual(probs.shape, (self.n,))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_calibrate_and_evaluate(self):
        """End-to-end calibrate + evaluate returns all expected metric keys."""
        ens = self._make_ensemble()
        ens.calibrate_threshold(self.X_q, self.X_c, self.y)
        metrics = ens.evaluate(self.X_q, self.X_c, self.y)
        for key in ["accuracy", "f1_score", "precision", "recall", "roc_auc", "quantum_arbitration_rate"]:
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_arbitration_report(self):
        """Arbitration report contains expected section headers."""
        ens = self._make_ensemble()
        ens.predict_proba_from_features(self.X_q, self.X_c)
        report = ens.get_arbitration_report()
        self.assertIn("Hybrid Ensemble Arbitration Report", report)
        self.assertIn("Quantum arbitration path", report)


if __name__ == "__main__":
    if pytest is not None and "pytest" in sys.modules:
        pytest.main([__file__, "-v"])
    else:
        unittest.main(verbosity=2)
