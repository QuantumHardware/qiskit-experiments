from qiskit.quantum_info import Clifford, Operator

class CustomClifford:
    def __init__(self, qc_default, qc_custom):
        self.qc_default = qc_default
        self.qc_custom = qc_custom
        self.opeartor = Operator(self.qc_default)