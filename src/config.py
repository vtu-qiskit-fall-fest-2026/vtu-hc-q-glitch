"""
Configuration settings for Q-Glitch pipeline.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""
    n_samples: int = 500
    n_clinical_features: int = 12
    n_genetic_markers: int = 8
    n_phenotypic_features: int = 6
    rare_disease_prevalence: float = 0.15
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")


@dataclass
class QuantumConfig:
    """Configuration for quantum circuit parameters."""
    n_qubits: int = 6
    feature_map_reps: int = 2
    ansatz_reps: int = 3
    entanglement: str = "full"
    optimizer_name: str = "cobyla"
    max_iter: int = 150
    shots: int = 1024
    backend: str = "aer_simulator"


@dataclass
class ClassicalConfig:
    """Configuration for classical ML baselines."""
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    xgb_n_estimators: int = 100
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6


@dataclass
class DashboardConfig:
    """Configuration for Streamlit dashboard."""
    page_title: str = "Q-Glitch — Rare Disease Predictor"
    page_icon: str = "⚛️"
    layout: str = "wide"
    theme_primary: str = "#7C3AED"
    theme_secondary: str = "#06B6D4"


@dataclass
class ProjectConfig:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    project_name: str = "Q-Glitch"
    team_id: str = "VTU-QFF-HC-0190"
    version: str = "2.0.0"


# Global config instance
CONFIG = ProjectConfig()
