"""Machine learning family: quantum_kernel, data_reuploading, qsvm, qcnn,
qgan_generator, quantum_autoencoder."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "quantum_kernel", "machine_learning", (2, 8),
    tags=["feature_map", "zz_interaction", "kernel_method"],
)
def make_quantum_kernel(n_qubits=4, **kwargs) -> QuantumCircuit:
    """ZZFeatureMap-style quantum kernel circuit.

    Encodes classical data with H + RZ (single-qubit) + CX-RZ-CX (ZZ interaction).
    Two repetitions of the feature map.
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(123)

    for _rep in range(2):
        # Single-qubit encoding
        for i in range(n):
            qc.h(i)
            qc.rz(rng.uniform(0, 2 * np.pi), i)
        # ZZ entangling feature map
        for i in range(n - 1):
            qc.cx(i, i + 1)
            qc.rz(rng.uniform(0, 2 * np.pi), i + 1)
            qc.cx(i, i + 1)
    return qc


@register_algorithm(
    "data_reuploading", "machine_learning", (2, 6),
    tags=["classifier", "reuploading", "variational"],
)
def make_data_reuploading(n_qubits=2, layers=3, **kwargs) -> QuantumCircuit:
    """Data re-uploading classifier: layered RY/RZ + CX.

    Each layer re-encodes data via RY/RZ rotations, interleaved with CX.
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(456)

    for _layer in range(layers):
        # Data encoding + trainable rotations
        for i in range(n):
            qc.ry(rng.uniform(0, 2 * np.pi), i)
            qc.rz(rng.uniform(0, 2 * np.pi), i)
        # Entangling layer
        for i in range(n - 1):
            qc.cx(i, i + 1)
    return qc


@register_algorithm(
    "qsvm", "machine_learning", (2, 8),
    tags=["kernel_method", "classification"],
)
def make_qsvm(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Quantum SVM feature map (ZZ feature map + measurement-ready).

    Implements the ZZFeatureMap structure used in quantum kernel methods
    for support vector classification.

    Args:
        n_qubits: Number of qubits / features (minimum 2).

    Tags: kernel_method, classification
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(2024)

    # Simulate classical data values x_i in [0, 2pi)
    data = rng.uniform(0, 2 * np.pi, n)

    for _rep in range(2):
        # Step 1: Hadamard on all qubits
        for i in range(n):
            qc.h(i)

        # Step 2: Single-qubit data encoding RZ(x_i)
        for i in range(n):
            qc.rz(data[i], i)

        # Step 3: ZZ entangling feature map on adjacent pairs
        for i in range(n - 1):
            phi_ij = (np.pi - data[i]) * (np.pi - data[i + 1])
            qc.cx(i, i + 1)
            qc.rz(2 * phi_ij, i + 1)
            qc.cx(i, i + 1)

    return qc


@register_algorithm(
    "qcnn", "machine_learning", (4, 8),
    tags=["neural_network", "classification"],
)
def make_qcnn(n_qubits=8, **kwargs) -> QuantumCircuit:
    """Quantum convolutional neural network.

    Alternating convolutional and pooling layers that progressively
    reduce the number of active qubits, mirroring the structure of a
    classical CNN.

    Args:
        n_qubits: Number of qubits, must be power of 2 (minimum 4).

    Tags: neural_network, classification
    """
    # Ensure power of 2, minimum 4
    n = max(4, n_qubits)
    # Round up to next power of 2
    n = 1 << (n - 1).bit_length()

    qc = QuantumCircuit(n)
    rng = np.random.default_rng(808)

    active = list(range(n))

    while len(active) > 1:
        # Convolutional layer: RY + CX on adjacent pairs
        for i in range(0, len(active) - 1, 2):
            q0 = active[i]
            q1 = active[i + 1]
            qc.ry(rng.uniform(0, 2 * np.pi), q0)
            qc.ry(rng.uniform(0, 2 * np.pi), q1)
            qc.cx(q0, q1)

        # Also entangle odd-even boundary pairs for full coverage
        for i in range(1, len(active) - 1, 2):
            q0 = active[i]
            q1 = active[i + 1]
            qc.ry(rng.uniform(0, 2 * np.pi), q0)
            qc.ry(rng.uniform(0, 2 * np.pi), q1)
            qc.cx(q0, q1)

        # Pooling layer: CX from even to odd, discard odd
        surviving = []
        for i in range(0, len(active) - 1, 2):
            q_even = active[i]
            q_odd = active[i + 1]
            qc.cx(q_even, q_odd)
            surviving.append(q_even)

        active = surviving

    return qc


@register_algorithm(
    "qgan_generator", "machine_learning", (2, 8),
    tags=["generative", "adversarial"],
)
def make_qgan_generator(n_qubits=4, layers=3, **kwargs) -> QuantumCircuit:
    """QGAN generator circuit.

    Parameterised ansatz structured as a quantum generator network for
    a quantum generative adversarial network.

    Args:
        n_qubits: Number of qubits (minimum 2).
        layers: Number of generator layers (default 3).

    Tags: generative, adversarial
    """
    layers = kwargs.get("layers", layers)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(606)

    for _layer in range(layers):
        # Single-qubit rotations: RY for amplitude, RZ for phase
        for i in range(n):
            qc.ry(rng.uniform(0, 2 * np.pi), i)
            qc.rz(rng.uniform(0, 2 * np.pi), i)

        # Entangling layer: linear CX chain
        for i in range(n - 1):
            qc.cx(i, i + 1)

    # Final rotation layer for output expressibility
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), i)

    return qc


@register_algorithm(
    "quantum_autoencoder", "machine_learning", (4, 8),
    tags=["compression", "autoencoder"],
)
def make_quantum_autoencoder(n_qubits=6, **kwargs) -> QuantumCircuit:
    """Quantum autoencoder circuit.

    Compresses an n-qubit quantum state into a smaller latent register.

    Args:
        n_qubits: Total number of qubits (minimum 4, even preferred).

    Tags: compression, autoencoder
    """
    n = max(4, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(909)

    n_input = n // 2
    n_latent = n - n_input

    input_qubits = list(range(n_input))
    latent_qubits = list(range(n_input, n))

    # Encoder: parametrised layers compressing input -> latent
    # Layer 1: RY rotations on all qubits
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), i)

    # Layer 2: CX entangling input with latent (cross-register coupling)
    for i in range(min(n_input, n_latent)):
        qc.cx(input_qubits[i], latent_qubits[i])

    # Layer 3: More RY + intra-register entangling
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), i)
    for i in range(n_input - 1):
        qc.cx(input_qubits[i], input_qubits[i + 1])
    for i in range(n_latent - 1):
        qc.cx(latent_qubits[i], latent_qubits[i + 1])

    # Layer 4: Cross-register entangling again
    for i in range(min(n_input, n_latent)):
        qc.cx(input_qubits[i], latent_qubits[i])

    # Final RY for output expressibility
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), i)

    # SWAP test between trash (input) and reference (latent)
    n_pairs = min(n_input, n_latent)
    for i in range(n_pairs):
        qc.cx(input_qubits[i], latent_qubits[i])

    return qc
