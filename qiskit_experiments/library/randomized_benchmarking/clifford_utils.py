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
"""
Utilities for using the Clifford group in randomized benchmarking
"""

from functools import lru_cache
from numbers import Integral
from typing import Optional, Union

import numpy as np
from numpy.random import Generator, default_rng

from qiskit.circuit import Gate, Instruction
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import SdgGate, HGate, SGate
from qiskit.quantum_info import Clifford, random_clifford

from .custom_clifford import CustomClifford

@lru_cache(maxsize=None)
def _clifford_1q_int_to_instruction(num: Integral) -> Instruction:
    ## >>> MODIFIED FROM
    ## return CliffordUtils.clifford_1_qubit_circuit(num).to_instruction()
    ## -- TO 
    return CliffordUtils.clifford_1_qubit_circuit(num)
    ## <<< END

@lru_cache(maxsize=11520)
def _clifford_2q_int_to_instruction(num: Integral) -> Instruction:
    return CliffordUtils.clifford_2_qubit_circuit(num).to_instruction()


class VGate(Gate):
    """V Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new V Gate."""
        super().__init__("v", 1, [])

    def _define(self):
        """V Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(SdgGate(), [q[0]], []), (HGate(), [q[0]], [])]
        self.definition = qc


class WGate(Gate):
    """W Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new W Gate."""
        super().__init__("w", 1, [])

    def _define(self):
        """W Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(HGate(), [q[0]], []), (SGate(), [q[0]], [])]
        self.definition = qc


class CliffordUtils:
    """Utilities for generating 1 and 2 qubit clifford circuits and elements"""

    NUM_CLIFFORD_1_QUBIT = 24
    NUM_CLIFFORD_2_QUBIT = 11520
    CLIFFORD_1_QUBIT_SIG = (2, 3, 4)
    CLIFFORD_2_QUBIT_SIGS = [
        (2, 2, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 4, 4),
    ]

    @classmethod
    @lru_cache(maxsize=24)
    def clifford_1_qubit(cls, num):
        """Return the 1-qubit clifford element corresponding to `num`
        where `num` is between 0 and 23.
        """
        ## >> MODIFIED FROM
        ## Clifford(cls.clifford_1_qubit_circuit(num), validate=False)
        ## -- TO 
        return cls.clifford_1_qubit_circuit(num) 
        ## << END

    @classmethod
    @lru_cache(maxsize=11520)
    def clifford_2_qubit(cls, num):
        """Return the 2-qubit clifford element corresponding to `num`
        where `num` is between 0 and 11519.
        """
        return Clifford(cls.clifford_2_qubit_circuit(num), validate=False)

    def random_cliffords(
        self, num_qubits: int, size: int = 1, rng: Optional[Union[int, Generator]] = None
    ):
        """Generate a list of random clifford elements"""
        if num_qubits > 2:
            return random_clifford(num_qubits, seed=rng)

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        if num_qubits == 1:
            samples = rng.integers(24, size=size)
            return [Clifford(self.clifford_1_qubit_circuit(i), validate=False) for i in samples]
        else:
            samples = rng.integers(11520, size=size)
            return [Clifford(self.clifford_2_qubit_circuit(i), validate=False) for i in samples]

    def random_clifford_circuits(
        self, num_qubits: int, size: int = 1, rng: Optional[Union[int, Generator]] = None
    ):
        """Generate a list of random clifford circuits"""
        if num_qubits > 2:
            return [random_clifford(num_qubits, seed=rng).to_circuit() for _ in range(size)]

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        if num_qubits == 1:
            samples = rng.integers(24, size=size)
            return [self.clifford_1_qubit_circuit(i) for i in samples]
        else:
            samples = rng.integers(11520, size=size)
            return [self.clifford_2_qubit_circuit(i) for i in samples]

    @classmethod
    @lru_cache(maxsize=24)
    def clifford_1_qubit_circuit(cls, num):
        """Return the 1-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 23.
        """
        ## >>> MODIFIED FROM 
        # unpacked = cls._unpack_num(num, (2, 3, 4))
        # i, j, p = unpacked[0], unpacked[1], unpacked[2]
        # qc = QuantumCircuit(1, name=f"Clifford-1Q({num})")
        # if i == 1:
        #     qc.h(0)
        # if j == 1:
        #     qc.sxdg(0)
        # if j == 2:
        #     qc.s(0)
        # if p == 1:
        #     qc.x(0)
        # if p == 2:
        #     qc.y(0)
        # if p == 3:
        #     qc.z(0)

        # return qc
        ## -- TO
        # Redefine 24 Clifford gates with custom gates with SX4 methods
        def SX4_gates(qc, theta, alpha, beta):
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
        if num == 0:
            theta, alpha, beta = 0.0 * np.pi/4, -0.0 * np.pi/4, 0.0 * np.pi/4
        if num == 1:
            theta, alpha, beta = 3.0 * np.pi/4, 4.0 * np.pi/4, 0.0 * np.pi/4
        if num == 2:
            theta, alpha, beta = -1.0 * np.pi/4, -2.0 * np.pi/4, -2.0 * np.pi/4
        if num == 3:
            theta, alpha, beta = 2.0 * np.pi/4, 2.0 * np.pi/4, 0.0 * np.pi/4
        if num == 4:
            theta, alpha, beta = 1.0 * np.pi/4, 2.0 * np.pi/4, 0.0 * np.pi/4
        if num == 5:
            theta, alpha, beta = -2.0 * np.pi/4, -4.0 * np.pi/4, -2.0 * np.pi/4
        if num == 6:
            theta, alpha, beta = 4.0 * np.pi/4, 4.0 * np.pi/4, 0.0 * np.pi/4
        if num == 7:
            theta, alpha, beta = 1.0 * np.pi/4, -0.0 * np.pi/4, 0.0 * np.pi/4
        if num == 8:
            theta, alpha, beta = 3.0 * np.pi/4, 2.0 * np.pi/4, 2.0 * np.pi/4
        if num == 9:
            theta, alpha, beta = 0.0 * np.pi/4, -2.0 * np.pi/4, 0.0 * np.pi/4
        if num == 10:
            theta, alpha, beta = 1.0 * np.pi/4, -2.0 * np.pi/4, 0.0 * np.pi/4
        if num == 11:
            theta, alpha, beta = 2.0 * np.pi/4, 0.0 * np.pi/4, 2.0 * np.pi/4
        if num == 12:
            theta, alpha, beta = 2.0 * np.pi/4, 2.0 * np.pi/4, -2.0 * np.pi/4
        if num == 13:
            theta, alpha, beta = -1.0 * np.pi/4, 0.0 * np.pi/4, -4.0 * np.pi/4
        if num == 14:
            theta, alpha, beta = 1.0 * np.pi/4, 2.0 * np.pi/4, -2.0 * np.pi/4
        if num == 15:
            theta, alpha, beta = -2.0 * np.pi/4, -2.0 * np.pi/4, -4.0 * np.pi/4
        if num == 16:
            theta, alpha, beta = -1.0 * np.pi/4, -4.0 * np.pi/4, -2.0 * np.pi/4
        if num == 17:
            theta, alpha, beta = 0.0 * np.pi/4, 0.0 * np.pi/4, -2.0 * np.pi/4
        if num == 18:
            theta, alpha, beta = 2.0 * np.pi/4, 4.0 * np.pi/4, 0.0 * np.pi/4
        if num == 19:
            theta, alpha, beta = -3.0 * np.pi/4, -4.0 * np.pi/4, -4.0 * np.pi/4
        if num == 20:
            theta, alpha, beta = 5.0 * np.pi/4, 6.0 * np.pi/4, 2.0 * np.pi/4
        if num == 21:
            theta, alpha, beta = -4.0 * np.pi/4, -6.0 * np.pi/4, -4.0 * np.pi/4
        if num == 22:
            theta, alpha, beta = -1.0 * np.pi/4, -2.0 * np.pi/4, 0.0 * np.pi/4
        if num == 23:
            theta, alpha, beta = 4.0 * np.pi/4, 4.0 * np.pi/4, 2.0 * np.pi/4
            
        qc = QuantumCircuit(1, name=f"Clifford-1Q({num})")
        qc = SX4_gates(qc, theta, alpha, beta)

        qc_default = QuantumCircuit(1, name=f"Clifford-1Q({num})")
        unpacked = cls._unpack_num(num, (2, 3, 4))
        i, j, p = unpacked[0], unpacked[1], unpacked[2]
        if i == 1:
            qc_default.h(0)
        if j == 1:
            qc_default.sxdg(0)
        if j == 2:
            qc_default.s(0)
        if p == 1:
            qc_default.x(0)
        if p == 2:
            qc_default.y(0)
        if p == 3:
            qc_default.z(0)
        ## END
        return CustomClifford(qc_default, qc)
        

    @classmethod
    @lru_cache(maxsize=11520)
    def clifford_2_qubit_circuit(cls, num):
        """Return the 2-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 11519.
        """
        vals = cls._unpack_num_multi_sigs(num, cls.CLIFFORD_2_QUBIT_SIGS)
        qc = QuantumCircuit(2, name=f"Clifford-2Q({num})")
        if vals[0] == 0 or vals[0] == 3:
            (form, i0, i1, j0, j1, p0, p1) = vals
        else:
            (form, i0, i1, j0, j1, k0, k1, p0, p1) = vals
        if i0 == 1:
            qc.h(0)
        if i1 == 1:
            qc.h(1)
        if j0 == 1:
            qc.sxdg(0)
        if j0 == 2:
            qc.s(0)
        if j1 == 1:
            qc.sxdg(1)
        if j1 == 2:
            qc.s(1)
        if form in (1, 2, 3):
            qc.cx(0, 1)
        if form in (2, 3):
            qc.cx(1, 0)
        if form == 3:
            qc.cx(0, 1)
        if form in (1, 2):
            if k0 == 1:  # V gate
                qc.sdg(0)
                qc.h(0)
            if k0 == 2:  # W gate
                qc.h(0)
                qc.s(0)
            if k1 == 1:  # V gate
                qc.sdg(1)
                qc.h(1)
            if k1 == 2:  # W gate
                qc.h(1)
                qc.s(1)
        if p0 == 1:
            qc.x(0)
        if p0 == 2:
            qc.y(0)
        if p0 == 3:
            qc.z(0)
        if p1 == 1:
            qc.x(1)
        if p1 == 2:
            qc.y(1)
        if p1 == 3:
            qc.z(1)

        return qc

    @staticmethod
    def _unpack_num(num, sig):
        r"""Returns a tuple :math:`(a_1, \ldots, a_n)` where
        :math:`0 \le a_i \le \sigma_i` where
        sig=:math:`(\sigma_1, \ldots, \sigma_n)` and num is the sequential
        number of the tuple
        """
        res = []
        for k in sig:
            res.append(num % k)
            num //= k
        return res

    @staticmethod
    def _unpack_num_multi_sigs(num, sigs):
        """Returns the result of `_unpack_num` on one of the
        signatures in `sigs`
        """
        for i, sig in enumerate(sigs):
            sig_size = 1
            for k in sig:
                sig_size *= k
            if num < sig_size:
                return [i] + CliffordUtils._unpack_num(num, sig)
            num -= sig_size
        return None
