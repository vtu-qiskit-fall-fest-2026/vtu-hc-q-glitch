"""
Main entry point for Q-Glitch pipeline.

Orchestrates the full workflow: data generation, preprocessing,
model training (classical + quantum), evaluation, and reporting.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import logging
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG
from src.data_generator import RareDiseaseDataGenerator, prepare_splits
from src.preprocessing import DataPreprocessor
from src.classical_models import ClassicalBaselines
from src.quantum_kernel import QuantumKernelClassifier
from src.quantum_vqc import VariationalQuantumClassifier
from src.hybrid_ensemble import HybridEnsemble
from src.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("q-glitch")


def print_banner():
    """Display the Q-Glitch project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║   ⚛️  Q-GLITCH                                       ║
    ║   Multimodal Quantum AI for Rare-Disease Prediction  ║
    ║                                                      ║
    ║   Team: Q-Glitch | ID: VTU-QFF-HC-0190               ║
    ║   Track: Quantum + AI & ML                           ║
    ║   VTU-Qiskit Fall Fest 2026                          ║
    ║                                                      ║
    ║   Authors: Mallikarjun.R, Mohammed Adil,             ║
    ║            Mallanagouda M                             ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def run_pipeline():
    """Execute the full Q-Glitch training and evaluation pipeline."""
    print_banner()
    start_time = time.time()

    # ──────────────────────────────────────────────
    # Step 1: Generate synthetic multimodal dataset
    # ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 1: Generating multimodal dataset")
    logger.info("=" * 50)

    generator = RareDiseaseDataGenerator()
    dataset = generator.generate()
    generator.save_dataset(dataset)

    logger.info(
        f"Dataset: {dataset['combined'].shape[0]} samples, "
        f"{dataset['combined'].shape[1]} features "
        f"(clinical={dataset['clinical'].shape[1]}, "
        f"genetic={dataset['genetic'].shape[1]}, "
        f"phenotypic={dataset['phenotypic'].shape[1]})"
    )

    # ──────────────────────────────────────────────
    # Step 2: Preprocess and split data
    # ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 2: Preprocessing and splitting data")
    logger.info("=" * 50)

    splits = prepare_splits(dataset)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    preprocessor = DataPreprocessor()
    X_train_classical, X_train_quantum = preprocessor.fit_transform(X_train, y_train)
    X_val_classical = preprocessor.transform_classical(X_val)
    X_val_quantum = preprocessor.transform_quantum(X_val)
    X_test_classical = preprocessor.transform_classical(X_test)
    X_test_quantum = preprocessor.transform_quantum(X_test)

    logger.info(f"Classical features: {X_train_classical.shape[1]}")
    logger.info(f"Quantum features: {X_train_quantum.shape[1]} (= n_qubits)")

    # ──────────────────────────────────────────────
    # Step 3: Train classical baselines
    # ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 3: Training classical ML baselines")
    logger.info("=" * 50)

    classical = ClassicalBaselines()
    classical.train_all(X_train_classical, y_train)
    classical_results = classical.evaluate_all(X_test_classical, y_test)

    # ──────────────────────────────────────────────
    # Step 4: Train Quantum Kernel Classifier (QSVM)
    # ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 4: Training Quantum Kernel Classifier (QSVM)")
    logger.info("=" * 50)

    qsvm = QuantumKernelClassifier()
    qsvm.train(X_train_quantum, y_train)
    qsvm.calibrate_threshold(X_val_quantum, y_val)
    qsvm_metrics = qsvm.evaluate(X_test_quantum, y_test)

    circuit_text = qsvm.get_circuit_diagram()
    if circuit_text:
        logger.info(f"QSVM Feature Map Circuit:\n{circuit_text}")

    # ──────────────────────────────────────────────
    # Step 5: Train Variational Quantum Classifier
    # ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 5: Training Variational Quantum Classifier (VQC)")
    logger.info("=" * 50)

    vqc = VariationalQuantumClassifier()
    vqc.train(X_train_quantum, y_train)
    vqc.calibrate_threshold(X_val_quantum, y_val)
    vqc_metrics = vqc.evaluate(X_test_quantum, y_test)

    # ──────────────────────────────────────────────
    # Step 6: Hybrid Quantum-Classical Ensemble
    # ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 6: Building Hybrid Quantum-Classical Ensemble")
    logger.info("=" * 50)

    ensemble = HybridEnsemble.from_pipeline(
        classical_baselines=classical,
        quantum_kernel_clf=qsvm,
    )
    ensemble.calibrate_threshold(X_val_quantum, X_val_classical, y_val)
    ensemble_metrics = ensemble.evaluate(X_test_quantum, X_test_classical, y_test)
    logger.info(ensemble.get_arbitration_report())

    # ──────────────────────────────────────────────
    # Step 7: Quantum Hardware Noise Simulation & Mitigation
    # ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 7: Quantum Hardware Noise & Error Mitigation Benchmark")
    logger.info("=" * 50)

    from src.quantum_noise import QuantumNoiseBenchmark
    noise_bench = QuantumNoiseBenchmark()
    noise_results = noise_bench.run_noise_benchmark(X_test_quantum)
    noise_chart_path = noise_bench.plot_benchmark()

    # ──────────────────────────────────────────────
    # Step 8: Compare and report results
    # ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 8: Comprehensive Evaluation & Reporting")
    logger.info("=" * 50)

    evaluator = ModelEvaluator()

    # Register classical results
    for model_name, metrics in classical_results.items():
        evaluator.add_result(model_name, metrics)

    # Register quantum and ensemble results
    evaluator.add_result("Quantum Kernel (QSVM)", qsvm_metrics)
    evaluator.add_result("VQC (Variational)", vqc_metrics)
    evaluator.add_result("★ Hybrid Ensemble (Q+RF+SVM)", ensemble_metrics)

    # Generate comparison chart
    evaluator.plot_comparison_bar()

    # Plot VQC training convergence
    vqc_history = vqc.get_training_history()
    if vqc_history:
        evaluator.plot_training_history(vqc_history, "VQC")

    # Print final report
    report = evaluator.generate_report()
    print("\n" + report)

    from src.model_store import save_trained_pipeline
    bundle = {
        "evaluator": evaluator,
        "classical": classical,
        "qsvm": qsvm,
        "vqc": vqc,
        "ensemble": ensemble,
        "ensemble_metrics": ensemble_metrics,
        "noise_benchmark": noise_results,
        "preprocessor": preprocessor,
        "dataset": dataset,
        "vqc_history": vqc_history,
    }
    save_trained_pipeline(bundle)

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed:.1f} seconds")

    return bundle


if __name__ == "__main__":
    results = run_pipeline()
