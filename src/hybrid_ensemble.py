"""
Hybrid Quantum-Classical Ensemble Classifier for Rare-Disease Prediction.

Implements a weighted consensus arbitration engine that combines:
  - Quantum Kernel Classifier (QSVC with ZZFeatureMap fidelity kernel)
  - Classical Random Forest (ensemble diversity, interpretability)
  - Classical SVM RBF (high-margin boundary for routine cases)

The ensemble uses calibrated probability averaging with dynamic quantum
weighting: the QSVC receives elevated weight on borderline samples where
classical confidence < 0.7, exploiting Hilbert-space geometry to resolve
ambiguous rare-disease cases.

Clinical significance: quantum arbitration is invoked precisely on the
"diagnostic grey zone" — the narrow probability band where classical
models disagree — matching real clinical utility of quantum computing.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
Event: VTU-Qiskit Fall Fest Hackathon 2026
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from .config import CONFIG

logger = logging.getLogger(__name__)


class HybridEnsemble:
    """
    Quantum-Classical Weighted Consensus Ensemble.

    Architecture:
        Soft-vote averaging across:
          w_q * P_QSVC(x) + w_rf * P_RF(x) + w_svm * P_SVM(x)
        where w_q is dynamically increased for borderline samples
        (classical confidence 0.35–0.65) to exploit quantum Hilbert-space
        separation for difficult edge cases.

    This directly implements the hybrid clinical workflow described in
    the Q-Glitch architecture: classical fast-path for routine predictions,
    quantum arbitration for rare-disease boundary cases.
    """

    # Default base weights (QSVC, RF, SVM)
    BASE_WEIGHTS = (0.40, 0.35, 0.25)
    # Elevated quantum weight on borderline samples
    BORDERLINE_WEIGHTS = (0.60, 0.25, 0.15)
    BORDERLINE_BAND = (0.30, 0.70)

    def __init__(
        self,
        quantum_classifier=None,
        classical_rf=None,
        classical_svm=None,
        base_weights: Optional[Tuple[float, float, float]] = None,
        borderline_weights: Optional[Tuple[float, float, float]] = None,
        dynamic_quantum_arbitration: bool = True,
    ):
        """
        Args:
            quantum_classifier: Trained QuantumKernelClassifier instance (QSVC).
            classical_rf: Trained RandomForestClassifier (or ClassicalBaselines model).
            classical_svm: Trained SVC with RBF kernel.
            base_weights: (w_quantum, w_rf, w_svm) for routine samples.
            borderline_weights: Weights used when classical confidence is ambiguous.
            dynamic_quantum_arbitration: If True, increase quantum weight on borderlines.
        """
        self.qsvc = quantum_classifier
        self.rf = classical_rf
        self.svm = classical_svm
        self.base_weights = np.array(base_weights or self.BASE_WEIGHTS, dtype=float)
        self.base_weights /= self.base_weights.sum()
        self.borderline_weights = np.array(borderline_weights or self.BORDERLINE_WEIGHTS, dtype=float)
        self.borderline_weights /= self.borderline_weights.sum()
        self.dynamic_qa = dynamic_quantum_arbitration
        self.optimal_threshold = 0.5
        self.is_trained = False
        self._arbitration_stats: Dict[str, int] = {"routine": 0, "borderline_quantum": 0}

    @classmethod
    def from_pipeline(
        cls,
        classical_baselines,
        quantum_kernel_clf,
        **kwargs,
    ) -> "HybridEnsemble":
        """
        Factory: build ensemble directly from the Q-Glitch pipeline objects.

        Args:
            classical_baselines: ClassicalBaselines instance (already trained).
            quantum_kernel_clf: QuantumKernelClassifier instance (already trained).
        """
        rf_model = classical_baselines.models.get("Random Forest")
        svm_model = classical_baselines.models.get("SVM (RBF)")

        instance = cls(
            quantum_classifier=quantum_kernel_clf,
            classical_rf=rf_model,
            classical_svm=svm_model,
            **kwargs,
        )
        instance.is_trained = True
        logger.info(
            "HybridEnsemble assembled from pipeline: "
            f"QSVC(w={instance.base_weights[0]:.2f}) + "
            f"RF(w={instance.base_weights[1]:.2f}) + "
            f"SVM(w={instance.base_weights[2]:.2f})"
        )
        return instance

    def _get_qsvc_probas(self, X_quantum: np.ndarray) -> np.ndarray:
        """Get QSVC probabilities for quantum-encoded features."""
        if self.qsvc is not None and self.qsvc.is_trained:
            return self.qsvc.predict_proba(X_quantum)[:, 1]
        return np.full(len(X_quantum), 0.5)

    def _get_rf_probas(self, X_classical: np.ndarray) -> np.ndarray:
        """Get Random Forest probabilities for classical features."""
        if self.rf is not None:
            return self.rf.predict_proba(X_classical)[:, 1]
        return np.full(len(X_classical), 0.5)

    def _get_svm_probas(self, X_classical: np.ndarray) -> np.ndarray:
        """Get SVM probabilities for classical features."""
        if self.svm is not None:
            return self.svm.predict_proba(X_classical)[:, 1]
        return np.full(len(X_classical), 0.5)

    def predict_proba_from_features(
        self,
        X_quantum: np.ndarray,
        X_classical: np.ndarray,
    ) -> np.ndarray:
        """
        Compute hybrid ensemble probability predictions.

        For each sample:
          - Compute base RF + SVM average (classical confidence).
          - If classical confidence is in borderline band [0.30, 0.70],
            apply elevated QSVC weight (quantum arbitration mode).
          - Otherwise apply base weights.

        Returns:
            Array of shape (n_samples,) with P(rare_disease | x).
        """
        p_q = self._get_qsvc_probas(X_quantum)
        p_rf = self._get_rf_probas(X_classical)
        p_svm = self._get_svm_probas(X_classical)

        n = len(p_q)
        p_ensemble = np.zeros(n)
        self._arbitration_stats = {"routine": 0, "borderline_quantum": 0}

        lo, hi = self.BORDERLINE_BAND

        for i in range(n):
            # Classical consensus signal
            p_classical_avg = 0.5 * (p_rf[i] + p_svm[i])

            if self.dynamic_qa and lo <= p_classical_avg <= hi:
                # QUANTUM ARBITRATION: classical models disagree — elevate QSVC weight
                w = self.borderline_weights
                self._arbitration_stats["borderline_quantum"] += 1
            else:
                w = self.base_weights
                self._arbitration_stats["routine"] += 1

            p_ensemble[i] = w[0] * p_q[i] + w[1] * p_rf[i] + w[2] * p_svm[i]

        return np.clip(p_ensemble, 0.0, 1.0)

    def calibrate_threshold(
        self,
        X_quantum_val: np.ndarray,
        X_classical_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """
        Calibrate ensemble decision threshold on validation set to maximize F1.

        Returns:
            Optimal threshold value.
        """
        p_ensemble = self.predict_proba_from_features(X_quantum_val, X_classical_val)
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 81):
            preds = (p_ensemble >= t).astype(int)
            f1 = float(f1_score(y_val, preds, zero_division=0))
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.optimal_threshold = best_t
        logger.info(
            f"Hybrid Ensemble optimal threshold: {self.optimal_threshold:.3f} "
            f"(Validation F1: {best_f1:.3f})"
        )
        return self.optimal_threshold

    def predict(
        self,
        X_quantum: np.ndarray,
        X_classical: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Binary rare-disease predictions using calibrated threshold."""
        t = threshold if threshold is not None else self.optimal_threshold
        return (self.predict_proba_from_features(X_quantum, X_classical) >= t).astype(int)

    def evaluate(
        self,
        X_quantum_test: np.ndarray,
        X_classical_test: np.ndarray,
        y_test: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Evaluate hybrid ensemble and return comprehensive metrics.

        Returns:
            Dict with accuracy, f1_score, precision, recall, roc_auc,
            and quantum_arbitration_rate (fraction of samples resolved by QSVC).
        """
        t = threshold if threshold is not None else self.optimal_threshold
        p_ensemble = self.predict_proba_from_features(X_quantum_test, X_classical_test)
        y_pred = (p_ensemble >= t).astype(int)

        n_total = len(y_test)
        n_borderline = self._arbitration_stats.get("borderline_quantum", 0)
        qa_rate = n_borderline / max(n_total, 1)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, p_ensemble)),
            "quantum_arbitration_rate": round(qa_rate, 3),
        }

        logger.info(
            f"  Hybrid Quantum-Classical Ensemble: "
            f"Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"AUC={metrics['roc_auc']:.4f} | "
            f"Quantum Arbitration: {n_borderline}/{n_total} samples "
            f"({qa_rate*100:.1f}%) (threshold={t:.3f})"
        )

        logger.info(
            f"\n{classification_report(y_test, y_pred, target_names=['No RD', 'Rare Disease'], zero_division=0)}"
        )

        return metrics

    def get_arbitration_report(self) -> str:
        """Return a human-readable arbitration breakdown."""
        total = sum(self._arbitration_stats.values())
        n_q = self._arbitration_stats.get("borderline_quantum", 0)
        n_r = self._arbitration_stats.get("routine", 0)
        return (
            f"Hybrid Ensemble Arbitration Report\n"
            f"  Total samples evaluated : {total}\n"
            f"  Routine (classical path) : {n_r} ({100*n_r/max(total,1):.1f}%)\n"
            f"  Quantum arbitration path : {n_q} ({100*n_q/max(total,1):.1f}%)\n"
            f"  QSVC invoked on border   : [{self.BORDERLINE_BAND[0]:.2f}, {self.BORDERLINE_BAND[1]:.2f}] classical confidence band\n"
            f"  Base weights (Q/RF/SVM)  : {self.base_weights.round(3).tolist()}\n"
            f"  Border weights (Q/RF/SVM): {self.borderline_weights.round(3).tolist()}"
        )
