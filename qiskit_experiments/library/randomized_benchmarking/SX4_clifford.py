from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Operator
import numpy as np
import qutip as qutip
from typing import *
from numpy.typing import NDArray
import qiskit.quantum_info as qi
from sympy import compose

SX4_indexes = np.array(
    [
        [0.0, -0.0, 0.0],
        [3.0, 4.0, 0.0],
        [-1.0, -2.0, -2.0],
        [2.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [-2.0, -4.0, -2.0],
        [4.0, 4.0, 0.0],
        [1.0, -0.0, 0.0],
        [3.0, 2.0, 2.0],
        [0.0, -2.0, 0.0],
        [1.0, -2.0, 0.0],
        [2.0, 0.0, 2.0],
        [2.0, 2.0, -2.0],
        [-1.0, 0.0, -4.0],
        [1.0, 2.0, -2.0],
        [-2.0, -2.0, -4.0],
        [-1.0, -4.0, -2.0],
        [0.0, 0.0, -2.0],
        [2.0, 4.0, 0.0],
        [-3.0, -4.0, -4.0],
        [5.0, 6.0, 2.0],
        [-4.0, -6.0, -4.0],
        [-1.0, -2.0, 0.0],
        [4.0, 4.0, 2.0],
    ]
)
multiply_map = np.array(
    [
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ],
        [
            1,
            0,
            23,
            22,
            9,
            8,
            19,
            18,
            5,
            4,
            15,
            14,
            13,
            12,
            11,
            10,
            21,
            20,
            7,
            6,
            17,
            16,
            3,
            2,
        ],
        [
            2,
            3,
            6,
            7,
            17,
            16,
            8,
            9,
            0,
            1,
            23,
            22,
            20,
            21,
            12,
            13,
            11,
            10,
            14,
            15,
            18,
            19,
            5,
            4,
        ],
        [
            3,
            2,
            4,
            5,
            1,
            0,
            15,
            14,
            16,
            17,
            13,
            12,
            21,
            20,
            22,
            23,
            19,
            18,
            9,
            8,
            10,
            11,
            7,
            6,
        ],
        [
            4,
            5,
            15,
            14,
            18,
            19,
            16,
            17,
            3,
            2,
            6,
            7,
            10,
            11,
            21,
            20,
            12,
            13,
            22,
            23,
            9,
            8,
            0,
            1,
        ],
        [
            5,
            4,
            1,
            0,
            2,
            3,
            23,
            22,
            19,
            18,
            20,
            21,
            11,
            10,
            7,
            6,
            8,
            9,
            17,
            16,
            13,
            12,
            14,
            15,
        ],
        [
            6,
            7,
            8,
            9,
            10,
            11,
            0,
            1,
            2,
            3,
            4,
            5,
            18,
            19,
            20,
            21,
            22,
            23,
            12,
            13,
            14,
            15,
            16,
            17,
        ],
        [
            7,
            6,
            17,
            16,
            3,
            2,
            13,
            12,
            11,
            10,
            21,
            20,
            19,
            18,
            5,
            4,
            15,
            14,
            1,
            0,
            23,
            22,
            9,
            8,
        ],
        [
            8,
            9,
            0,
            1,
            23,
            22,
            2,
            3,
            6,
            7,
            17,
            16,
            14,
            15,
            18,
            19,
            5,
            4,
            20,
            21,
            12,
            13,
            11,
            10,
        ],
        [
            9,
            8,
            10,
            11,
            7,
            6,
            21,
            20,
            22,
            23,
            19,
            18,
            15,
            14,
            16,
            17,
            13,
            12,
            3,
            2,
            4,
            5,
            1,
            0,
        ],
        [
            10,
            11,
            21,
            20,
            12,
            13,
            22,
            23,
            9,
            8,
            0,
            1,
            4,
            5,
            15,
            14,
            18,
            19,
            16,
            17,
            3,
            2,
            6,
            7,
        ],
        [
            11,
            10,
            7,
            6,
            8,
            9,
            17,
            16,
            13,
            12,
            14,
            15,
            5,
            4,
            1,
            0,
            2,
            3,
            23,
            22,
            19,
            18,
            20,
            21,
        ],
        [
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ],
        [
            13,
            12,
            11,
            10,
            21,
            20,
            7,
            6,
            17,
            16,
            3,
            2,
            1,
            0,
            23,
            22,
            9,
            8,
            19,
            18,
            5,
            4,
            15,
            14,
        ],
        [
            14,
            15,
            18,
            19,
            5,
            4,
            20,
            21,
            12,
            13,
            11,
            10,
            8,
            9,
            0,
            1,
            23,
            22,
            2,
            3,
            6,
            7,
            17,
            16,
        ],
        [
            15,
            14,
            16,
            17,
            13,
            12,
            3,
            2,
            4,
            5,
            1,
            0,
            9,
            8,
            10,
            11,
            7,
            6,
            21,
            20,
            22,
            23,
            19,
            18,
        ],
        [
            16,
            17,
            3,
            2,
            6,
            7,
            4,
            5,
            15,
            14,
            18,
            19,
            22,
            23,
            9,
            8,
            0,
            1,
            10,
            11,
            21,
            20,
            12,
            13,
        ],
        [
            17,
            16,
            13,
            12,
            14,
            15,
            11,
            10,
            7,
            6,
            8,
            9,
            23,
            22,
            19,
            18,
            20,
            21,
            5,
            4,
            1,
            0,
            2,
            3,
        ],
        [
            18,
            19,
            20,
            21,
            22,
            23,
            12,
            13,
            14,
            15,
            16,
            17,
            6,
            7,
            8,
            9,
            10,
            11,
            0,
            1,
            2,
            3,
            4,
            5,
        ],
        [
            19,
            18,
            5,
            4,
            15,
            14,
            1,
            0,
            23,
            22,
            9,
            8,
            7,
            6,
            17,
            16,
            3,
            2,
            13,
            12,
            11,
            10,
            21,
            20,
        ],
        [
            20,
            21,
            12,
            13,
            11,
            10,
            14,
            15,
            18,
            19,
            5,
            4,
            2,
            3,
            6,
            7,
            17,
            16,
            8,
            9,
            0,
            1,
            23,
            22,
        ],
        [
            21,
            20,
            22,
            23,
            19,
            18,
            9,
            8,
            10,
            11,
            7,
            6,
            3,
            2,
            4,
            5,
            1,
            0,
            15,
            14,
            16,
            17,
            13,
            12,
        ],
        [
            22,
            23,
            9,
            8,
            0,
            1,
            10,
            11,
            21,
            20,
            12,
            13,
            16,
            17,
            3,
            2,
            6,
            7,
            4,
            5,
            15,
            14,
            18,
            19,
        ],
        [
            23,
            22,
            19,
            18,
            20,
            21,
            5,
            4,
            1,
            0,
            2,
            3,
            17,
            16,
            13,
            12,
            14,
            15,
            11,
            10,
            7,
            6,
            8,
            9,
        ],
    ]
).astype(int)

inverse_map = np.array(
    [
        0,
        1,
        8,
        5,
        22,
        3,
        6,
        19,
        2,
        23,
        10,
        15,
        12,
        13,
        14,
        11,
        16,
        21,
        18,
        7,
        20,
        17,
        4,
        9,
    ]
).astype(int)


def reduce_angle(angle: float) -> int:
    """Reduce angle in range of (-pi, pi]

    Args:
        angle (float): Bloch angle

    Returns:
        int: Bloch angle
    """
    if angle > np.pi:
        angle -= 2 * np.pi
        return reduce_angle(angle)
    elif angle <= -np.pi:
        angle += 2 * np.pi
        return reduce_angle(angle)
    else:
        return angle


def in_sin_cos_range(x: float) -> float:
    """Preprocessing before putting value to arccos or arcsin function to preventing NaN

    Args:
        x (float): _description_

    Returns:
        float: _description_
    """
    if abs(x - 1.0) < 1e-6:
        return 1.0
    elif abs(x + 1.0) < 1e-6:
        return -1.0
    else:
        return x


def decompose_standard(op: NDArray) -> Tuple[float]:
    """Decompose Operator to standard angles, which are defined as eq. 4 in arXiv:2105.02398

    Args:
        op (NDArray): Single qubit operator

    Returns:
        Tuple[float]: standard angles
    """
    phase_factor = 0.5 * np.arctan2(np.imag(np.linalg.det(op)), np.real(np.linalg.det(op)))
    gamma_tilde = np.arccos(in_sin_cos_range(abs(op[0, 0])))
    alpha_tilde = np.angle(op[0, 0]) - phase_factor
    beta_tilde = np.angle(op[1, 0]) - phase_factor
    return alpha_tilde, beta_tilde, gamma_tilde


def standard_to_SX4(alpha_tilde: float, beta_tilde: float, gamma_tilde: float) -> Tuple[float]:
    """Standard angles to SX4 angles defined as Op = SX(beta)SX(theta)SX(theta)SX(alpha)

    Args:
        alpha_tilde (float): standard angle alpha
        beta_tilde (float): standard angle beta
        gamma_tilde (float): standard angle gamma

    Returns:
        Tuple[float]: SX4 angles theta, alpha, beta for op = SX(beta)SX(theta)SX(theta)SX(alpha)
    """
    alpha = -alpha_tilde - beta_tilde
    beta = alpha_tilde - beta_tilde
    theta = gamma_tilde - beta_tilde
    return theta, alpha, beta


def SX4_to_standard(theta: float, alpha: float, beta: float):
    """SX4 angles to standard angles

    Args:
        theta (float): SX4 angle theta
        alpha (float): SX4 angle alpha
        beta (float): SX4 angle beta

    Returns:
       Tuple[float]: Standard angles alpha, beta, gamma
    """
    alpha_tilde = (beta - alpha) / 2
    beta_tilde = -(alpha + beta) / 2
    gamma_tilde = beta_tilde + theta
    return alpha_tilde, beta_tilde, gamma_tilde


def decompose_SX4(op: NDArray) -> Tuple[float]:
    """Decompose operator to SX4 angles

    Args:
        op (NDArray): Target operator

    Returns:
        Tuple[float]: SX4 angles
    """
    return standard_to_SX4(*decompose_standard(op))


def standard_to_bloch(alpha_tilde: float, beta_tilde: float, gamma_tilde: float) -> Tuple[float]:
    """Convert standard angles to Bloch angles

    Args:
        alpha_tilde (float): standard angle alpha
        beta_tilde (float): standard angle beta
        gamma_tilde (float): standard angle gamma

    Returns:
        Tuple[float]: Bloch angles
    """
    rotation = in_sin_cos_range(2 * np.arccos(np.cos(alpha_tilde) * np.cos(gamma_tilde)))
    sin_rot = np.sin(rotation / 2)
    if np.abs(sin_rot) < 1e-12:
        if np.abs(reduce_angle(rotation)) < 1e-12:
            return 0, 0, 0

        else:
            nx = in_sin_cos_range(-np.sin(beta_tilde) * np.sin(gamma_tilde))
            ny = in_sin_cos_range(np.cos(beta_tilde) * np.sin(gamma_tilde))
            nz = in_sin_cos_range(-np.sin(alpha_tilde) * np.cos(gamma_tilde))
            sin_rot = np.linalg.norm([nx, ny, nz])
            nx /= sin_rot
            ny /= sin_rot
            nz /= sin_rot
            rotation = in_sin_cos_range(2 * np.arcsin(sin_rot))
            theta = in_sin_cos_range(np.arccos(nz))
            sin_theta = np.sin(theta)
            if np.abs(reduce_angle(rotation)) < 1e-6:
                return 0, 0, 0
            if np.abs(sin_theta) < 1e-6:
                if theta >= (np.pi - 1e-6):
                    theta = reduce_angle(theta - np.pi)
                    rotation = reduce_angle(-1 * rotation)
                return reduce_angle(rotation), reduce_angle(theta), 1e-12
            phi = np.arctan2(ny / sin_theta, nx / sin_theta)

    else:
        if np.abs(reduce_angle(rotation)) < 1e-6:
            return 0, 0, 0
        nx = in_sin_cos_range(-np.sin(beta_tilde) * np.sin(gamma_tilde) / sin_rot)
        ny = in_sin_cos_range(np.cos(beta_tilde) * np.sin(gamma_tilde) / sin_rot)
        nz = in_sin_cos_range(-np.sin(alpha_tilde) * np.cos(gamma_tilde) / sin_rot)
        theta = np.arccos(nz)
        sin_theta = np.sin(theta)
        if np.abs(sin_theta) < 1e-6:
            if theta >= (np.pi - 1e-6):
                theta = reduce_angle(theta - np.pi)
                rotation = reduce_angle(-1 * rotation)
            return reduce_angle(rotation), reduce_angle(theta), reduce_angle(1e-12)

        phi = np.arctan2(ny / sin_theta, nx / sin_theta)

    if phi < 0:
        phi = reduce_angle(phi + np.pi)
        theta = reduce_angle(np.pi - theta)
        rotation = reduce_angle(-1 * rotation)

    if phi >= (np.pi - 1e-6):
        phi = reduce_angle(phi + np.pi)
        theta = reduce_angle(np.pi - theta)
        rotation = reduce_angle(-1 * rotation)

    if theta >= (np.pi - 1e-6):
        theta = reduce_angle(theta - np.pi)
        rotation = reduce_angle(-1 * rotation)

    return reduce_angle(rotation), reduce_angle(theta), reduce_angle(phi)


def decompose_bloch(op: NDArray) -> Tuple[float]:
    """Decompose operator to Bloch angles

    Args:
        op (NDArray): Operator

    Returns:
        Tuple[float]: Bloch angles
    """
    return standard_to_bloch(*decompose_standard(op))


def SX4_gates(qc: QuantumCircuit, theta: float, alpha: float, beta: float) -> QuantumCircuit:
    """Add SX4 gate to the given quantum circuit with SX4 angles

    Args:
        qc (QuantumCircuit): Quantum Circuit
        theta (float): SX4 angle theta
        alpha (float): SX4 angle alpha
        beta (float): SX4 angle beta

    Returns:
        QuantumCircuit: Quantum Circuits
    """
    qc.rz(alpha, 0)
    qc.sx(0)
    qc.rz(-alpha, 0)

    qc.rz(theta, 0)
    qc.sx(0)
    qc.rz(-theta, 0)

    qc.rz(theta, 0)
    qc.sx(0)
    qc.rz(-theta, 0)

    qc.rz(beta, 0)
    qc.sx(0)
    qc.rz(-beta, 0)
    return qc


def qc_to_operator(qc: QuantumCircuit) -> NDArray:
    """Convert Quantum Circuit to Operator

    Args:
        qc (QuantumCircuit): Quantum circuit

    Returns:
        NDArray: Operator
    """
    return qi.Operator(qc).data


def clifford_num_to_circuit(qc: QuantumCircuit, num: int) -> QuantumCircuit:
    """Create Clifford gate circuit based on SX4 method

    Args:
        qc (QuantumCircuit): Quantum Circuit
        num (int): clifford number

    Returns:
        QuantumCircuit: Quantum Circuit
    """
    SX4_angles = SX4_indexes[num] * np.pi / 4
    return SX4_gates(qc, *SX4_angles)


class SX4Clifford:
    """Clifford gates with SX4 method"""

    def __init__(self, num):
        self.num = num
        self.qc = QuantumCircuit(1, name=f"Clifford-1Q({num})")
        self.qc = clifford_num_to_circuit(self.qc, self.num)

    def compose(self, rop):
        composed_num = multiply_map[rop.num, self.num]
        return SX4Clifford(composed_num)

    def adjoint(self):
        inv_num = inverse_map[self.num]
        return SX4Clifford(inv_num)

    def to_instruction(self):
        return self.qc.to_instruction()
