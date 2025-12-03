"""
交互式可视化模块
包含所有交互式组件的统一接口
"""

from .gradient_descent import InteractiveGradientDescent
from .optimizer import InteractiveOptimizer
from .regularization import InteractiveRegularization
from .convolution import InteractiveConvolution
from .matrix import InteractiveMatrix
from .svm import InteractiveSVM
from .lagrange import InteractiveLagrange
from .vcdim import InteractiveVCDim
from .calculus import InteractiveCalculus
from .probability import InteractiveProbability
from .ml_curves import InteractiveMLCurves
from .noise import InteractiveNoise
from .training_dynamics import InteractiveTrainingDynamics
from .multimodal_geometry import InteractiveMultimodalGeometry
from .vcdim_derivation import InteractiveVCDimDerivation
from .signal_processing import InteractiveSignalProcessing
from .cnn_math_foundations import InteractiveCNNMathFoundations
from .neural_geometry import InteractiveNeuralGeometry
from .hilbert_space import InteractiveHilbertSpace
from .diffusion_model import InteractiveDiffusionModel
from .kernel_regression import InteractiveKernelRegression
from .neuroevolution import InteractiveNeuroevolution
from .probabilistic_programming import InteractiveProbabilisticProgramming
from .mdp import InteractiveMDP
from .information_geometry import InteractiveInformationGeometry
from .kernel_regression import InteractiveKernelRegression
from .gcn import InteractiveGCN
from .causation import InteractiveCausation
from .optimal_transport import InteractiveOptimalTransport
from .game_theory import InteractiveGameTheory
from .scaling_laws import InteractiveScalingLaws
from .classification_optimization import InteractiveClassificationOptimization
from .dimensions_parameters import InteractiveDimensionsParameters
from .loss_function import InteractiveLossFunction


# 模块注册表
INTERACTIVE_MODULES = {
    "loss": {
        "loss_function": InteractiveLossFunction,
        "gradient_descent": InteractiveGradientDescent,
    },
    "convolution": {
        "convolution_demo": InteractiveConvolution,
    },
    "matrix": {
        "matrix_transform": InteractiveMatrix,
    },
    "svm": {
        "svm_classifier": InteractiveSVM,
    },
    "optimizer": {
        "optimizer_comparison": InteractiveOptimizer,
    },
    "regularization": {
        "l1_l2_comparison": InteractiveRegularization,
    },
    "lagrange": {
        "constraint_optimization": InteractiveLagrange,
    },
    "vcdim": {
        "vc_theory": InteractiveVCDim,
    },
    "calculus": {
        "calculus_fundamentals": InteractiveCalculus,
    },
    "probability": {
        "probability_info": InteractiveProbability,
    },
    "ml_curves": {
        "important_curves": InteractiveMLCurves,
    },
    "noise": {
        "noise_theory": InteractiveNoise,
    },
    "training_dynamics": {
        "training_dynamics": InteractiveTrainingDynamics,
    },
    "multimodal_geometry": {
        "multimodal_geometry": InteractiveMultimodalGeometry,
    },
    "vcdim_derivation": {
        "vcdim_derivation": InteractiveVCDimDerivation,
    },
    "signal_processing": {
        "signal_processing": InteractiveSignalProcessing,
    },
    "cnn_math_foundations": {
        "cnn_math_foundations": InteractiveCNNMathFoundations,
    },
    "neural_geometry": {
        "dimension_analysis": InteractiveNeuralGeometry,
    },
    "hilbert_space": {
        "fourier_analysis": InteractiveHilbertSpace,
    },
    "diffusion_model": {
        "sde_dynamics": InteractiveDiffusionModel,
    },
    "kernel_regression": {
        "attention_mechanism": InteractiveKernelRegression,
    },
    "neuroevolution": {
        "zero_order_optimization": InteractiveNeuroevolution,
    },
    "probabilistic_programming": {
        "uncertainty_quantification": InteractiveProbabilisticProgramming,
    },
    "mdp": {
        "reinforcement_learning": InteractiveMDP,
    },
    "information_geometry": {
        "natural_optimization": InteractiveInformationGeometry,
    },
    "gcn": {
        "graph_neural_network": InteractiveGCN,
    },
    "causation": {
        "causal_inference": InteractiveCausation,
    },
    "optimal_transport": {
        "transport_theory": InteractiveOptimalTransport,
    },
    "game_theory": {
        "strategic_reasoning": InteractiveGameTheory,
    },
    "scaling_laws": {
        "scaling_analysis": InteractiveScalingLaws,
    },
    "classification_optimization": {
        "optimization_logic": InteractiveClassificationOptimization,
    },
    "dimensions_parameters": {
        "calculator": InteractiveDimensionsParameters,
    }
}


__all__ = [
    'InteractiveGradientDescent',
    'InteractiveOptimizer',
    'InteractiveRegularization',
    'InteractiveConvolution',
    'InteractiveMatrix',
    'InteractiveSVM',
    'InteractiveLagrange',
    'InteractiveVCDim',
    'InteractiveCalculus',
    'InteractiveProbability',
    'InteractiveMLCurves',
    'InteractiveNoise',
    'InteractiveTrainingDynamics',
    'InteractiveMultimodalGeometry',
    'InteractiveVCDimDerivation',
    'InteractiveSignalProcessing',
    'InteractiveCNNMathFoundations',
    'InteractiveNeuralGeometry',
    'InteractiveHilbertSpace',
    'InteractiveDiffusionModel',
    'InteractiveKernelRegression',
    'InteractiveNeuroevolution',
    'InteractiveProbabilisticProgramming',
    'InteractiveMDP',
    'InteractiveInformationGeometry',
    'InteractiveGCN',
    'InteractiveCausation',
    'InteractiveOptimalTransport',
    'InteractiveGameTheory',
    'InteractiveScalingLaws',
    'InteractiveClassificationOptimization',
    'InteractiveDimensionsParameters',
    'InteractiveLossFunction',
    'INTERACTIVE_MODULES'
]
